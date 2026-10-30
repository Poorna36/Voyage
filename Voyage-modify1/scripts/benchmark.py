"""
scripts/benchmark.py
====================
S4 + S5 entry-point — correspondence matching and uniformity selection.

Orchestrates the matcher registry loop over a pair manifest, applies gates,
produces matches_raw.json and matches_selected.json + selection_stats.json.

Exit codes (PIPELINE.md §8 — applies to all pipeline scripts):
  0 — success (all pairs processed)
  1 — gate failures logged; non-failed pairs completed
  2 — config error
  3 — env error (missing package/dependency)
  4 — leakage audit failed

Usage
-----
  python scripts/benchmark.py \\
      --pair data/pairs/manifest.jsonl \\
      --matchers configs/matchers.yaml \\
      --config configs/ohrc_nac.yaml \\
      --out results/ \\
      [--splits train] [--parallel 1] [--resume] [--force] [-v]

References: ARCHITECTURE.md §3, PIPELINE.md S4-S5, FEATURES.md F14
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
import numpy as np
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# Ensure workspace root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Provenance (F25) ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src.provenance import build_provenance, set_global_seed
except ImportError:
    # Graceful degradation — provenance fields will be empty strings
    def build_provenance(config=None, matcher_params=None, seed=None, **kw):  # type: ignore
        return {"config_hash": "", "code_commit": "", "matcher_params_hash": "",
                "created_at": "", "seed": seed or 42}
    def set_global_seed(seed=42): return seed  # type: ignore

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("benchmark")

# GPU lock file — ensures M2 (LightGlue) and M3 (YOLOv9) are serialized
# across processes so that two workers never fight for VRAM at the same time.
_GPU_LOCK_FILE = Path(tempfile.gettempdir()) / "sih2026_gpu.lock"


@contextmanager
def _gpu_lock(timeout_s: float = 300.0) -> Generator[None, None, None]:
    """
    Cross-process GPU serialization lock.

    Uses an exclusive lock file so that only one GPU matcher runs at a time
    when --parallel > 1 is set.  Falls back to a no-op on Windows if
    file-locking is unavailable (single-process safety is guaranteed by the
    GIL on CPU-only paths).
    """
    lock_path = _GPU_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(lock_path, "w")
        # Attempt an exclusive lock (non-blocking first, then polling)
        deadline = time.time() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    # Windows: use msvcrt file locking
                    import msvcrt
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl as _fcntl
                    _fcntl.flock(fh, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                break
            except (OSError, IOError):
                if time.time() > deadline:
                    log.warning("GPU lock timeout after %.0f s — proceeding anyway", timeout_s)
                    break
                time.sleep(0.5)
        yield
    finally:
        try:
            fh.close()
        except Exception:
            pass


# Matcher IDs that require GPU serialization
_GPU_MATCHERS = frozenset({"lightglue", "crater", "crater_hough"})


# ── Matcher registry ─────────────────────────────────────────────────────────

def _build_registry(matchers_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Lazily import and instantiate only the matchers listed in config."""
    from src.matching.sift import SIFTMatcher
    from src.matching.rift import RIFT2Matcher
    from src.matching.lnift import LNIFTMatcher
    from src.matching.lightglue import LightGlueMatcher
    from src.matching.crater import CraterMatcher

    _classes = {
        "sift": SIFTMatcher,
        "rift2": RIFT2Matcher,
        "lnift": LNIFTMatcher,
        "lightglue": LightGlueMatcher,
        "crater": CraterMatcher,
    }

    registry: Dict[str, Any] = {}
    for mid, cls in _classes.items():
        cfg = matchers_cfg.get(mid, {})
        if cfg.get("enabled", True):
            try:
                registry[mid] = cls(config=cfg)
                log.debug("Registered matcher: %s", mid)
            except Exception as exc:
                log.warning("Could not load matcher %s: %s", mid, exc)
    return registry


# ── Output helpers ────────────────────────────────────────────────────────────

def _output_dir(out_root: Path, pair_id: str, matcher_id: str) -> Path:
    d = out_root / pair_id / matcher_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _output_exists(out_dir: Path) -> bool:
    return (out_dir / "matches_selected.json").exists()


# ── Selection pipeline (S5) ───────────────────────────────────────────────────

def _run_selection(
    result,        # MatchResult
    matcher_id: str,
    pair_meta: dict,
    cfg: dict,
    verbose: bool = False,
) -> tuple:
    """
    Apply L3 uniformity pipeline to raw MatchResult.
    Returns (src_xy, ref_xy, confidence, stats_dict).
    """
    from src.selection.spatial import (
        confidence_filter,
        grid_cap,
        coverage_greedy,
        one_to_one,
        selection_stats,
    )

    sel_cfg = cfg.get("selection", {})
    n = int(sel_cfg.get("n", 8))
    cap = int(sel_cfg.get("cap", 5))
    budget = int(sel_cfg.get("budget", 250))
    cov_min = float(sel_cfg.get("coverage_min", 0.60))

    src_w = pair_meta.get("src_width", None)
    src_h = pair_meta.get("src_height", None)
    img_shape = (src_h, src_w) if src_h and src_w else None

    sx, rx, conf = result.src_xy, result.ref_xy, result.confidence

    # Step 1: confidence filter
    sx, rx, conf = confidence_filter(sx, rx, conf, matcher_id)

    # Step 2: grid cap
    sx, rx, conf = grid_cap(sx, rx, conf, n=n, cap=cap, image_shape=img_shape)

    # Step 3: coverage-greedy bisection
    sx, rx, conf = coverage_greedy(sx, rx, conf, budget=budget,
                                    min_coverage=cov_min, n=n,
                                    image_shape=img_shape)

    # Step 4: one-to-one deduplication
    sx, rx, conf = one_to_one(sx, rx, conf)

    stats = selection_stats(
        result.src_xy, sx,
        result.confidence, conf,
        image_shape=img_shape, n=n,
    )
    if verbose:
        log.info(
            "  selection: %d -> %d | cov=%.2f | std=%.2f",
            stats["n_before"], stats["n_after"],
            stats["coverage_after"], stats["grid_density_std_after"],
        )
    return sx, rx, conf, stats


# ── Per-pair per-matcher runner ───────────────────────────────────────────────

def _run_pair_matcher(
    pair: dict,
    matcher,
    out_root: Path,
    cfg: dict,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run one matcher on one pair. Returns a summary dict.
    Writes outputs to results/<pair_id>/<matcher_id>/.
    Failures written to failures.jsonl (never crash the loop).
    """
    pair_id = pair["pair_id"]
    mid = matcher.matcher_id
    out_dir = _output_dir(out_root, pair_id, mid)

    # Checkpointing — S4/S5 re-runs only if output missing or --force
    if resume and not force and _output_exists(out_dir):
        log.info("  SKIP (checkpointed): %s / %s", pair_id, mid)
        return {"pair_id": pair_id, "matcher": mid, "status": "skipped_checkpoint"}

    t0 = time.time()

    # ── Load preprocessed images ──────────────────────────────────────────────
    src_path = pair.get("src_processed")
    if not src_path:
        p_src = Path("data/processed") / pair_id / "src.tif"
        if p_src.exists():
            src_path = p_src
        elif "src" in pair and isinstance(pair["src"], dict) and pair["src"].get("cub_path"):
            src_path = Path(pair["src"]["cub_path"])

    ref_path = pair.get("ref_processed")
    if not ref_path:
        p_ref = Path("data/processed") / pair_id / "ref.tif"
        if p_ref.exists():
            ref_path = p_ref
        elif "ref" in pair and isinstance(pair["ref"], dict) and pair["ref"].get("path"):
            ref_path = Path(pair["ref"]["path"])

    if not src_path or not ref_path:
        reason = "missing_processed_image_paths"
        _log_failure(out_root, pair_id, mid, "S4", reason)
        return {"pair_id": pair_id, "matcher": mid, "status": "failed", "reason": reason}

    try:
        import rasterio
        with rasterio.open(src_path) as ds:
            s_arr = ds.read(1).astype(np.float32)
        with rasterio.open(ref_path) as ds:
            r_arr = ds.read(1).astype(np.float32)

        def _to_u8(img):
            if img.dtype == np.uint8:
                return img
            v_min, v_max = img.min(), img.max()
            if v_max > v_min:
                norm = (img - v_min) / (v_max - v_min)
            else:
                norm = img
            return (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)

        src_img = _to_u8(s_arr)
        ref_img = _to_u8(r_arr)
    except Exception as exc_rast:
        try:
            import cv2
            src_img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            if src_img is None or ref_img is None:
                raise FileNotFoundError(f"Could not read processed images: {exc_rast}")
        except Exception as exc:
            reason = f"image_load_failed: {exc}"
            _log_failure(out_root, pair_id, mid, "S4", reason)
            return {"pair_id": pair_id, "matcher": mid, "status": "failed", "reason": reason}

    gsd_ratio = float(pair.get("gsd_ratio", 1.0))

    # ── M3 gate check BEFORE running (FEATURES.md F13) ───────────────────────
    if mid in ("crater", "crater_hough"):
        from src.matching.crater import CraterMatcher
        gate_ok, gate_reason = CraterMatcher.check_gate(
            density_src=float(pair.get("crater_density_per_km2", 0.0)),
            density_ref=float(pair.get("ref_crater_density", 0.0)),
            terrain_src=pair.get("terrain_class", "unknown"),
            terrain_ref=pair.get("ref_terrain_class", "unknown"),
            tau_c=matcher.tau_c,
            allowed_terrain=matcher.allowed_terrain,
        )
        if not gate_ok:
            prov = build_provenance(matcher_params={"matcher_id": mid})
            result_raw = {"gate_skip": True, "gate_reason": gate_reason,
                          "pair_id": pair_id, "matcher": mid, **prov}
            _write_json(out_dir / "matches_raw.json", result_raw)
            log.info("  M3 gate FAIL (%s): %s", pair_id, gate_reason)
            return {"pair_id": pair_id, "matcher": mid, "status": "gate_skip",
                    "reason": gate_reason}

    # ── Run matcher (GPU matchers serialized via lock file) ───────────────────
    match_kwargs: dict = {}
    # Check for preprocessed shadow/valid mask
    mask_path = Path("data/processed") / pair_id / "valid_mask.png"
    if mask_path.exists():
        try:
            import cv2
            mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_arr is not None:
                match_kwargs["mask_src"] = mask_arr
        except Exception:
            pass

    if mid in ("crater", "crater_hough"):
        match_kwargs = {
            "crater_density_src": float(pair.get("crater_density_per_km2", 0.0)),
            "crater_density_ref": float(pair.get("ref_crater_density", 0.0)),
            "terrain_src": pair.get("terrain_class", "unknown"),
            "terrain_ref": pair.get("ref_terrain_class", "unknown"),
        }

    if mid in _GPU_MATCHERS:
        with _gpu_lock():
            result = matcher.match(src_img, ref_img, gsd_ratio=gsd_ratio, **match_kwargs)
    else:
        result = matcher.match(src_img, ref_img, gsd_ratio=gsd_ratio, **match_kwargs)

    # Write matches_raw.json — embed F25 provenance fields
    prov = build_provenance(
        matcher_params=result.matcher_params,
    )
    raw_out = {
        "pair_id": pair_id,
        "matcher": mid,
        "n_raw": result.count,
        "runtime_s": round(result.runtime_s, 4),
        "src_xy": result.src_xy.tolist(),
        "ref_xy": result.ref_xy.tolist(),
        "confidence": result.confidence.tolist(),
        "matcher_params": result.matcher_params,
        **prov,
    }
    _write_json(out_dir / "matches_raw.json", raw_out)

    # ── Gate S4: >= 150 candidate matches ────────────────────────────────────
    if result.count < 150:
        reason = f"S4_gate_fail: {result.count} < 150 candidates"
        _log_failure(out_root, pair_id, mid, "S4", reason, n_raw=result.count)
        log.warning("  [S4 FAIL] %s / %s: %s", pair_id, mid, reason)
        return {"pair_id": pair_id, "matcher": mid, "status": "s4_fail", "reason": reason}

    # ── Selection pipeline (S5) ───────────────────────────────────────────────
    sel_cfg = cfg.get("selection", {})
    cov_min = float(sel_cfg.get("coverage_min", 0.60))
    n_grid = int(sel_cfg.get("n", 8))

    sx, rx, conf, stats = _run_selection(result, mid, pair, cfg, verbose)

    # ── Gate S5: coverage >= 0.60 AND >= 25 matches ──────────────────────────
    if stats["coverage_after"] < cov_min or stats["n_after"] < 25:
        # Relax cap once (double cap) and retry
        log.info("  [S5 soft] %s / %s: coverage=%.2f n=%d — relaxing cap",
                 pair_id, mid, stats["coverage_after"], stats["n_after"])
        from src.selection.spatial import grid_cap, one_to_one, selection_stats
        sx2, rx2, conf2 = grid_cap(result.src_xy, result.ref_xy, result.confidence,
                                    n=n_grid, cap=int(sel_cfg.get("cap", 5)) * 2)
        sx2, rx2, conf2 = one_to_one(sx2, rx2, conf2)
        stats2 = selection_stats(result.src_xy, sx2, result.confidence, conf2)

        if stats2["coverage_after"] < cov_min or stats2["n_after"] < 25:
            reason = (f"S5_gate_fail: coverage={stats2['coverage_after']:.2f} "
                      f"n={stats2['n_after']}")
            _log_failure(out_root, pair_id, mid, "S5", reason, **stats2)
            log.warning("  [S5 FAIL] %s / %s: %s", pair_id, mid, reason)
            return {"pair_id": pair_id, "matcher": mid, "status": "s5_fail",
                    "reason": reason}
        sx, rx, conf, stats = sx2, rx2, conf2, stats2

    # ── Write outputs — embed F25 provenance in matches_selected.json ─────────
    sel_prov = build_provenance(
        matcher_params=result.matcher_params,
    )
    selected_out = {
        "pair_id": pair_id,
        "matcher": mid,
        "n_selected": int(len(sx)),
        "src_xy": sx.tolist(),
        "ref_xy": rx.tolist(),
        "confidence": conf.tolist(),
        "selection_stats": stats,
        "gsd_ratio": gsd_ratio,
        **sel_prov,
    }
    _write_json(out_dir / "matches_selected.json", selected_out)
    _write_json(out_dir / "selection_stats.json", stats)

    elapsed = round(time.time() - t0, 3)
    log.info("  OK %s / %s: n=%d cov=%.2f (%.1fs)",
             pair_id, mid, len(sx), stats["coverage_after"], elapsed)
    return {"pair_id": pair_id, "matcher": mid, "status": "ok",
            "n_selected": int(len(sx)), "elapsed_s": elapsed}


def _log_failure(
    out_root: Path,
    pair_id: str,
    matcher: str,
    stage: str,
    reason: str,
    **extra,
) -> None:
    """Append one line to failures.jsonl (never overwrite)."""
    entry = {
        "pair_id": pair_id,
        "matcher": matcher,
        "stage": stage,
        "reason": reason,
        **extra,
    }
    failures_path = out_root / "failures.jsonl"
    with failures_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="benchmark.py",
        description="S4+S5: matcher registry loop + spatial uniformity selection",
    )
    p.add_argument("--pair", "--manifest", required=True,
                   help="Path to data/pairs/manifest.jsonl")
    p.add_argument("--matchers", default="configs/matchers.yaml",
                   help="Path to matchers.yaml config")
    p.add_argument("--config", default="configs/ohrc_nac.yaml",
                   help="Pipeline config YAML")
    p.add_argument("--out", default="results",
                   help="Output root directory")
    p.add_argument(
        "--mode",
        choices=["benchmark", "production", "msm"],
        default="benchmark",
        help=(
            "benchmark: run ALL matchers from registry (M0 always-on baseline + M1/M2/M3). "
            "production: apply heuristic arbitration policy. "
            "msm: apply Matcher Selection Model (L1.5/S4.5) to route optimal matcher pipeline. "
            "Default: benchmark."
        ),
    )
    p.add_argument("--msm-config", default=None,
                   help="Path to msm.yaml config (used when --mode msm)")
    p.add_argument("--splits", nargs="*", default=["train"],
                   help="Which manifest splits to process (train/test)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of parallel workers (1 = sequential)")
    p.add_argument("--resume", action="store_true",
                   help="Skip pairs where outputs already exist")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs exist")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        with open(args.matchers) as f:
            matchers_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        log.error("Config file not found: %s", exc)
        return 2
    except Exception as exc:
        log.error("Config parse error: %s", exc)
        return 2

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest_path = Path(args.pair)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return 2

    pairs: List[dict] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if not args.splits or rec.get("split") in args.splits:
                    pairs.append(rec)

    if not pairs:
        log.warning("No pairs found for splits %s in %s", args.splits, manifest_path)
        return 0

    log.info("Loaded %d pairs, splits=%s", len(pairs), args.splits)

    # ── Build matcher registry ────────────────────────────────────────────────
    try:
        registry = _build_registry(matchers_cfg)
    except ImportError as exc:
        log.error("Dependency missing: %s", exc)
        return 3

    log.info("Active matchers: %s", list(registry.keys()))
    log.info("Mode: %s", args.mode)

    # ── M0 (SIFT) must always be in registry ─────────────────────────────────
    if "sift" not in registry:
        log.error("M0 (sift) not in registry — check matchers.yaml")
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Matcher Selection Model (MSM) initialization if requested ─────────────
    selector = None
    if args.mode == "msm":
        try:
            from src.selector import MatcherSelector, extract_features
            msm_config_source = args.msm_config if args.msm_config else cfg
            selector = MatcherSelector(config=msm_config_source)
            log.info("MSM MatcherSelector initialized (version: %s)", selector.model_version)
        except Exception as exc:
            log.warning("Failed to initialize MatcherSelector: %s. Falling back to benchmark mode.", exc)
            args.mode = "benchmark"

    # ── Determine which matchers to run based on mode ─────────────────────────
    def _matchers_for_pair(pair: dict) -> Dict[str, Any]:
        """
        benchmark mode: all registered matchers (M0 always-on).
        production mode: apply arbitration policy to select one matcher per pair.
            Policy: M3 (if crater gate + detector_validated) → M2 → M1 (polar flag) → M0
        msm mode: evaluate MSM feature vector and route predicted matcher(s).
        """
        if args.mode == "benchmark":
            return registry

        if args.mode == "msm" and selector is not None:
            pair_id = pair.get("pair_id", "unknown")
            meta_path = Path("data/processed") / pair_id / "meta.json"
            meta_json = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta_json = json.load(fh)
                except Exception as exc:
                    log.warning("[%s] Failed to load meta.json for MSM: %s", pair_id, exc)

            features = extract_features(pair, meta_json)
            sel_result = selector.predict(features)
            selector.save_result(sel_result, out_root / pair_id / "selector.json")

            log.info(
                "  [MSM] %s -> selected=%s (conf=%.2f), fallback=%s, matchers=%s, reason=%s",
                pair_id, sel_result.selected_matcher, sel_result.confidence,
                sel_result.fallback_matcher, sel_result.matchers_to_run, sel_result.routing_reason,
            )

            ordered_msm: Dict[str, Any] = {}
            for mid in sel_result.matchers_to_run:
                if mid in registry:
                    ordered_msm[mid] = registry[mid]
            # Ensure at least SIFT fallback if none found
            if not ordered_msm and "sift" in registry:
                ordered_msm["sift"] = registry["sift"]
            return ordered_msm

        # production / arbitration policy
        ordered: Dict[str, Any] = {}

        # M3 — crater gate
        for mid in ("crater", "crater_hough"):
            if mid in registry:
                m = registry[mid]
                try:
                    from src.matching.crater import CraterMatcher
                    gate_ok, _ = CraterMatcher.check_gate(
                        density_src=float(pair.get("crater_density_per_km2", 0.0)),
                        density_ref=float(pair.get("ref_crater_density", 0.0)),
                        terrain_src=pair.get("terrain_class", "unknown"),
                        terrain_ref=pair.get("ref_terrain_class", "unknown"),
                        tau_c=m.tau_c,
                        allowed_terrain=m.allowed_terrain,
                    )
                    if gate_ok:
                        ordered[mid] = m
                        break
                except Exception:
                    pass

        # M2 — LightGlue (not gated on GPU in production)
        if "lightglue" in registry:
            ordered["lightglue"] = registry["lightglue"]

        # M1 — RIFT2 or LNIFT (flag if polar)
        for mid in ("rift2", "lnift"):
            if mid in registry:
                lat = abs(float(pair.get("latitude_center_deg", 0.0)))
                if lat > 55:
                    log.info("  [production] polar pair (lat=%.1f) — M1 flagged", lat)
                ordered[mid] = registry[mid]
                break

        # M0 — SIFT always as final fallback
        ordered["sift"] = registry["sift"]
        log.debug("  [production] matcher order: %s", list(ordered.keys()))
        return ordered

    # ── Main loop ─────────────────────────────────────────────────────────────
    n_ok = n_fail = n_skip = 0
    for pair in pairs:
        pair_id = pair.get("pair_id", "unknown")
        log.info("Pair: %s", pair_id)

        active_matchers = _matchers_for_pair(pair)
        pair_succeeded = False

        for mid, matcher in list(active_matchers.items()):
            result_summary = _run_pair_matcher(
                pair, matcher, out_root, cfg,
                resume=args.resume, force=args.force, verbose=args.verbose,
            )
            status = result_summary.get("status", "")
            if status == "ok":
                n_ok += 1
                pair_succeeded = True
            elif status == "skipped_checkpoint":
                n_skip += 1
                pair_succeeded = True
            else:
                n_fail += 1

            # In production or MSM mode, stop after first successful matcher
            if (args.mode in ("production", "msm")) and pair_succeeded:
                log.info("  [%s] %s succeeded — stopping pair pipeline", args.mode, mid)
                break

        # In MSM mode: on S4/S5 failure of routed matcher(s), escalate to baseline M0 (SIFT)
        if args.mode == "msm" and not pair_succeeded and "sift" in registry and "sift" not in active_matchers:
            log.warning("  [MSM ESCALATION] Routed matcher(s) failed — escalating to M0 (SIFT)")
            result_summary = _run_pair_matcher(
                pair, registry["sift"], out_root, cfg,
                resume=args.resume, force=args.force, verbose=args.verbose,
            )
            if result_summary.get("status") in ("ok", "skipped_checkpoint"):
                n_ok += 1
            else:
                n_fail += 1

    log.info("Done — ok=%d fail=%d skip=%d", n_ok, n_fail, n_skip)

    return 1 if n_fail > 0 else 0



if __name__ == "__main__":
    sys.exit(main())

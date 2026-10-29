"""
scripts/preprocess.py
======================
S3 — Preprocessing Pipeline Entry Point (L1 layer).

Reads ``data/pairs/manifest.jsonl``, runs the full L1 preprocessing pipeline
on each pair, and writes outputs under ``data/processed/<pair_id>/``:

  src.tif          — preprocessed source image (GeoTIFF)
  ref.tif          — preprocessed reference image (GeoTIFF)
  valid_mask.png   — shadow mask (white=valid, black=masked)
  tiles.geojson    — tile bounding boxes for reassembly
  meta.json        — provenance log (every transform + F25 fields)

Exit codes (per PIPELINE.md §8):
  0 — All pairs processed successfully
  1 — Gate failures logged; non-failed pairs completed
  2 — Configuration error (bad YAML, missing key)
  3 — Environment error (missing dependency)
  4 — Leakage audit failed

Usage
-----
  python scripts/preprocess.py \\
      --manifest data/pairs/manifest.jsonl \\
      --config   configs/ohrc_nac.yaml \\
      --out      data/processed \\
      [--force] [-v]

References:
  - PIPELINE.md §3 (S3 stage)
  - FEATURES.md F04-F08
  - PROGRESS.md §2.6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Dependency imports with graceful error messages
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    print("[FATAL] numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(3)

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    print("[FATAL] rasterio not installed. Run: pip install rasterio", file=sys.stderr)
    sys.exit(3)

try:
    import yaml
except ImportError:
    print("[FATAL] pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(3)

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.masks import shadow_mask, check_mask_fraction, save_mask_png
from src.preprocessing.normalize import percentile_clip, stat_transfer
from src.preprocessing.branches import apply_ohrc_nac, apply_tmc_wac, apply_minimal, select_branch
from src.preprocessing.resample import reconcile_gsd
from src.preprocessing.tiling import tile_image, write_tile_geojson
from src.preprocessing.stats import compute_texture_contrast, compute_mean_gradient
from src.provenance import build_provenance, set_global_seed
from src.failures import log_gate_failure as write_failure

logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config, resolving 'extends: default' inheritance."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    extends = cfg.pop("extends", None)
    if extends:
        base_path = config_path.parent / f"{extends}.yaml"
        if base_path.exists():
            with open(base_path, encoding="utf-8") as fh:
                base_cfg = yaml.safe_load(fh) or {}
            # Shallow merge: child keys override base
            merged = {**base_cfg, **cfg}
            return merged

    return cfg


def _get_preproc_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the preprocessing sub-config block."""
    return cfg.get("preprocessing", {})


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _read_image(path: Path) -> tuple:
    """
    Read a single-band GeoTIFF (or PNG/TIF) into a float32 numpy array.

    Returns (array_2d, profile_dict).
    """
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
    return data, profile


def _write_geotiff(
    array: np.ndarray,
    out_path: Path,
    profile: Optional[Dict] = None,
) -> Path:
    """Write a float32 2-D array to GeoTIFF with a minimal profile."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape

    if profile is None:
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": w,
            "height": h,
            "count": 1,
            "crs": CRS.from_epsg(4326),
            "transform": from_bounds(0, 0, 1, 1, w, h),
            "compress": "deflate",
        }
    else:
        profile.update({"dtype": "float32", "count": 1, "compress": "deflate"})

    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(array[np.newaxis, :, :])

    return out_path


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def _process_pair(
    pair: Dict[str, Any],
    cfg: Dict[str, Any],
    out_root: Path,
    failures_path: Path,
    force: bool,
) -> bool:
    """
    Run the full L1 pipeline for a single pair record.

    Returns True on success, False on gate failure.
    """
    pair_id = pair.get("pair_id", "unknown")
    out_dir = out_root / pair_id
    meta_path = out_dir / "meta.json"

    # Checkpointing — skip if already done and not forced
    if meta_path.exists() and not force:
        logger.info("[%s] Already processed (skip with --force to redo)", pair_id)
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    preproc_cfg = _get_preproc_cfg(cfg)
    shadow_cfg = preproc_cfg.get("shadow_mask", {})
    tiling_cfg = preproc_cfg.get("tiling", {})
    gsd_cfg = preproc_cfg.get("gsd", {})
    sensor_pair = cfg.get("sensor_pair", pair.get("sensor_pair", "OHRC-NAC"))

    provenance_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Step 0 — Load images
    # ------------------------------------------------------------------ #
    src_raw = pair.get("src_processed_path") or pair.get("src_path")
    if not src_raw and isinstance(pair.get("src"), dict):
        src_raw = pair["src"].get("cub_path") or pair["src"].get("path")
    if not src_raw and isinstance(pair.get("src"), str):
        src_raw = pair["src"]

    ref_raw = pair.get("ref_processed_path") or pair.get("ref_path")
    if not ref_raw and isinstance(pair.get("ref"), dict):
        ref_raw = pair["ref"].get("path") or pair["ref"].get("cub_path")
    if not ref_raw and isinstance(pair.get("ref"), str):
        ref_raw = pair["ref"]

    src_path = Path(src_raw or f"data/processed/{pair_id}/src.tif")
    ref_path = Path(ref_raw or f"data/processed/{pair_id}/ref.tif")

    if not src_path.exists() or not ref_path.exists():
        reason = f"Source or reference image not found: src={src_path}, ref={ref_path}"
        logger.warning("[%s] GATE FAIL: %s", pair_id, reason)
        write_failure(
            failures_path, stage="S3", pair_id=pair_id,
            reason=reason, fallback_taken="skip",
        )
        return False

    src_img, src_profile = _read_image(src_path)
    ref_img, ref_profile = _read_image(ref_path)

    src_gsd = float(pair.get("src_gsd_m", cfg.get("pair", {}).get("src_gsd_m", 0.31)))
    ref_gsd = float(pair.get("ref_gsd_m", cfg.get("pair", {}).get("ref_gsd_m", 0.50)))
    solar_incidence = float(pair.get("solar_incidence_deg", 45.0))

    # ------------------------------------------------------------------ #
    # Step 1 — Shadow masking
    # ------------------------------------------------------------------ #
    mask = shadow_mask(
        src_img,
        solar_incidence_deg=solar_incidence,
        incidence_threshold_deg=float(shadow_cfg.get("incidence_threshold_deg", 80.0)),
        local_variance_window=int(shadow_cfg.get("local_variance_window", 15)),
        flat_variance_threshold=float(shadow_cfg.get("flat_variance_threshold", 10.0)),
    )
    save_mask_png(mask, out_dir / "valid_mask.png")

    fraction, in_range = check_mask_fraction(
        mask,
        min_pct=float(shadow_cfg.get("mask_min_pct", 5.0)),
        max_pct=float(shadow_cfg.get("mask_max_pct", 30.0)),
    )
    provenance_log.append({
        "stage": "shadow_mask",
        "mask_fraction": round(fraction, 4),
        "in_range": in_range,
        "solar_incidence_deg": solar_incidence,
    })

    if not in_range:
        # Gate: flag pair, but continue processing on the unmasked area (per spec)
        logger.warning(
            "[%s] Mask fraction %.1f%% outside [%.0f%%, %.0f%%] — pair flagged, "
            "continuing on unmasked area.",
            pair_id, fraction * 100,
            shadow_cfg.get("mask_min_pct", 5.0),
            shadow_cfg.get("mask_max_pct", 30.0),
        )
        write_failure(
            failures_path, stage="S3", pair_id=pair_id,
            reason=f"mask_fraction={fraction:.3f} outside [{shadow_cfg.get('mask_min_pct', 5)/100:.3f}, "
                   f"{shadow_cfg.get('mask_max_pct', 30)/100:.3f}]",
            fallback_taken="proceed_on_unmasked_area",
        )

    # ------------------------------------------------------------------ #
    # Step 2 — Percentile clip (both images)
    # ------------------------------------------------------------------ #
    lo, hi = preproc_cfg.get("radiometric_norm", {}).get("percentile_clip", [2, 98])
    src_clipped = percentile_clip(src_img, lo=float(lo), hi=float(hi))
    ref_clipped = percentile_clip(ref_img, lo=float(lo), hi=float(hi))
    provenance_log.append({
        "stage": "percentile_clip",
        "lo_pct": lo, "hi_pct": hi,
    })

    # ------------------------------------------------------------------ #
    # Step 3 — Statistical transfer (src → ref statistics)
    # ------------------------------------------------------------------ #
    do_stat_transfer = preproc_cfg.get("radiometric_norm", {}).get("stat_transfer", True)
    if do_stat_transfer:
        src_normalized = stat_transfer(src_clipped, ref_clipped)
        provenance_log.append({"stage": "stat_transfer"})
    else:
        src_normalized = src_clipped

    # ------------------------------------------------------------------ #
    # Step 4 — Sensor branch (classical matchers: heavy processing)
    # ------------------------------------------------------------------ #
    # We process for the default/classical matcher context here.
    # benchmark.py will apply apply_minimal() before learned matchers.
    branch_name = select_branch(sensor_pair, matcher_id="sift", config=cfg)
    branch_cfg_key = branch_name  # e.g. "ohrc_to_nac" or "tmc_to_wac"
    branch_cfg = preproc_cfg.get(branch_cfg_key, {})

    if branch_name == "ohrc_to_nac":
        src_branched = apply_ohrc_nac(src_normalized, branch_cfg)
        branch_experimental = False
    elif branch_name == "tmc_to_wac":
        src_branched = apply_tmc_wac(src_normalized, ref_clipped, branch_cfg)
        branch_experimental = True
    else:
        src_branched = apply_minimal(src_normalized, branch_cfg)
        branch_experimental = False

    provenance_log.append({
        "stage": "sensor_branch",
        "branch": branch_name,
        "branch_experimental": branch_experimental,
    })

    # ------------------------------------------------------------------ #
    # Step 5 — GSD reconciliation
    # ------------------------------------------------------------------ #
    low_angle_threshold = float(
        gsd_cfg.get("low_angle_threshold_deg", 45.0)
    )
    src_resampled, gsd_meta = reconcile_gsd(
        src_branched,
        src_gsd=src_gsd,
        ref_gsd=ref_gsd,
        solar_incidence_deg=solar_incidence,
        low_angle_threshold_deg=low_angle_threshold,
    )
    provenance_log.append({
        "stage": "gsd_reconciliation",
        **gsd_meta,
    })

    # If ref is coarser (e.g. TMC-WAC), resample the reference instead
    if gsd_meta["which_resampled"] == "ref":
        ref_resampled, _ = reconcile_gsd(
            ref_clipped,
            src_gsd=ref_gsd,
            ref_gsd=src_gsd,
            solar_incidence_deg=solar_incidence,
            low_angle_threshold_deg=low_angle_threshold,
        )
    else:
        ref_resampled = ref_clipped

    # ------------------------------------------------------------------ #
    # Step 6 — Tiling
    # ------------------------------------------------------------------ #
    tile_size = int(tiling_cfg.get("size_px", 512))
    overlap_px = int(tiling_cfg.get("overlap_px", 64))
    min_fraction = float(tiling_cfg.get("min_tile_fraction", 0.5))

    # Resize mask to match resampled src if needed
    if src_resampled.shape != mask.shape:
        import cv2
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (src_resampled.shape[1], src_resampled.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_resized = mask

    tiles = tile_image(
        src_resampled,
        tile_size=tile_size,
        overlap_px=overlap_px,
        min_fraction=min_fraction,
        valid_mask=mask_resized,
    )
    tiles_geojson_path = write_tile_geojson(
        tiles, pair_id=pair_id,
        out_path=out_dir / "tiles.geojson",
        tile_size=tile_size,
        overlap_px=overlap_px,
    )
    provenance_log.append({
        "stage": "tiling",
        "tile_size_px": tile_size,
        "overlap_px": overlap_px,
        "min_tile_fraction": min_fraction,
        "n_tiles": len(tiles),
    })

    # ------------------------------------------------------------------ #
    # Step 7 — Write output images
    # ------------------------------------------------------------------ #
    _write_geotiff(src_resampled, out_dir / "src.tif", profile=src_profile.copy())
    _write_geotiff(ref_resampled, out_dir / "ref.tif", profile=ref_profile.copy())

    # ------------------------------------------------------------------ #
    # Step 8 — Feature Stats & Write meta.json (provenance)
    # ------------------------------------------------------------------ #
    src_texture_contrast = float(compute_texture_contrast(src_resampled, valid_mask=mask_resized))
    ref_texture_contrast = float(compute_texture_contrast(ref_resampled))
    src_mean_gradient = float(compute_mean_gradient(src_resampled, valid_mask=mask_resized))
    ref_mean_gradient = float(compute_mean_gradient(ref_resampled))
    tile_count = len(tiles)

    provenance_log.append({
        "stage": "feature_stats",
        "src_texture_contrast": round(src_texture_contrast, 4),
        "ref_texture_contrast": round(ref_texture_contrast, 4),
        "src_mean_gradient": round(src_mean_gradient, 4),
        "ref_mean_gradient": round(ref_mean_gradient, 4),
        "tile_count": tile_count,
    })

    prov = build_provenance(config=cfg)
    meta = {
        "pair_id": pair_id,
        "sensor_pair": sensor_pair,
        "src_gsd_m": src_gsd,
        "ref_gsd_m": ref_gsd,
        "solar_incidence_deg": solar_incidence,
        "mask_fraction": round(fraction, 4),
        "masked_fraction": round(fraction, 4),
        "mask_in_range": in_range,
        "n_tiles": tile_count,
        "tile_count": tile_count,
        "src_texture_contrast": round(src_texture_contrast, 4),
        "ref_texture_contrast": round(ref_texture_contrast, 4),
        "src_mean_gradient": round(src_mean_gradient, 4),
        "ref_mean_gradient": round(ref_mean_gradient, 4),
        "pipeline_steps": provenance_log,
        **prov,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "[%s] Done: %d tiles, mask=%.1f%%, contrast=(%.2f, %.2f), grad=(%.2f, %.2f), branch=%s",
        pair_id, tile_count, fraction * 100,
        src_texture_contrast, ref_texture_contrast,
        src_mean_gradient, ref_mean_gradient,
        branch_name,
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S3 — L1 Preprocessing pipeline entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest", required=True,
        help="Path to data/pairs/manifest.jsonl",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to sensor-pair config YAML (e.g. configs/ohrc_nac.yaml)",
    )
    p.add_argument(
        "--out", default="data/processed",
        help="Root output directory (default: data/processed)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-process pairs that already have outputs",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 2

    try:
        cfg = _load_config(config_path)
    except Exception as exc:
        logger.error("Failed to parse config: %s", exc)
        return 2

    # Set global seed
    seed = int(cfg.get("global", {}).get("seed", 42))
    set_global_seed(seed)

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 2

    pairs: List[Dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed manifest line: %s", exc)

    if not pairs:
        logger.error("No pairs found in manifest: %s", manifest_path)
        return 2

    logger.info("Loaded %d pair(s) from %s", len(pairs), manifest_path)

    # Prepare output and failure paths
    out_root = Path(args.out)
    failures_path = Path(cfg.get("data", {}).get("failures", "results/failures.jsonl"))
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    # Process each pair
    n_success = 0
    n_failed = 0
    for pair in pairs:
        try:
            ok = _process_pair(pair, cfg, out_root, failures_path, args.force)
            if ok:
                n_success += 1
            else:
                n_failed += 1
        except Exception as exc:
            pair_id = pair.get("pair_id", "unknown")
            logger.error("[%s] Unexpected error: %s", pair_id, exc)
            if args.verbose:
                traceback.print_exc()
            write_failure(
                failures_path, stage="S3", pair_id=pair_id,
                reason=f"Unexpected exception: {exc}", fallback_taken="skip",
            )
            n_failed += 1

    logger.info(
        "Preprocessing complete: %d succeeded, %d failed out of %d pairs",
        n_success, n_failed, len(pairs),
    )

    if n_failed > 0 and n_success == 0:
        return 1  # All pairs failed
    if n_failed > 0:
        return 1  # Some pairs failed — gate failures logged
    return 0


if __name__ == "__main__":
    sys.exit(main())

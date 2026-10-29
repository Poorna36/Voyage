#!/usr/bin/env python3
"""
scripts/generate_synthetic_benchmark.py — Synthetic GT Benchmark Generator

Generates synthetic image pairs with exact floating-point ground truth for the
Component-Wise Synthetic Benchmark (Phase 10, v3.0).

Usage:
    python scripts/generate_synthetic_benchmark.py \\
        --config configs/synthetic_benchmark.yaml \\
        --images data/raw/ \\
        --out data/synthetic/ \\
        --phase 1

Benchmark Phases:
    Phase 1: smoke test — 1 seed (42), 5 GT anchors, translation only
    Phase 2: geometric suite — N=10 seeds, scale + rotation + translation
    Phase 3: photometric suite — N=30 seeds, all transforms
    Phase 4: full benchmark — N=50 seeds, all transforms, 6 terrain strata

Outputs per pair:
    data/synthetic/<pair_id>_src.tif         — source image (float32 GeoTIFF)
    data/synthetic/<pair_id>_tgt.tif         — transformed synthetic target
    data/synthetic/gt/<pair_id>_gt.json      — hidden GT anchor coords (never
                                               loaded by matcher)
    data/synthetic/synthetic_manifest.jsonl  — manifest (append-only per pair)

Exit codes (per PIPELINE.md §8):
    0 — success (all pairs generated)
    1 — gate failures logged; remaining pairs completed
    2 — config error
    3 — environment error (no source images found)
    4 — leakage audit failed (GT file leaked outside gt_dir)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Project root on sys.path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import yaml

from src.synthetic.anchors import extract_anchors, AnchorSet
from src.synthetic.transforms import (
    build_transform_matrix,
    apply_transform,
    transform_gt_points,
    generate_synthetic_pair,
    TransformParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_synthetic_benchmark")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """Load YAML config, resolve 'extends' chain (single level)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    extends = cfg.pop("extends", None)
    if extends:
        parent_path = config_path.parent / f"{extends}.yaml"
        if parent_path.exists():
            with open(parent_path) as f:
                parent = yaml.safe_load(f)
            # Shallow merge: child overrides parent at top level
            merged = {**parent, **cfg}
            cfg = merged
    return cfg


def _hash_config(cfg: dict) -> str:
    """Deterministic MD5 hash of config for provenance."""
    return hashlib.md5(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _find_source_images(images_dir: Path, extensions: Tuple[str, ...] = (".tif", ".img", ".png", ".fits", ".tiff", ".TIF", ".IMG", ".PNG", ".TIFF")) -> List[Path]:
    """Recursively find candidate source images."""
    found = []
    seen = set()
    for ext in extensions:
        for p in sorted(images_dir.rglob(f"*{ext}")):
            if p not in seen:
                seen.add(p)
                found.append(p)
    return found


def _load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image as a single-channel float32 array normalised to [0, 1].

    Supports .tif (via rasterio), .png/.jpg (via cv2), generic binary (.img).
    Automatically crops very large satellite strips to a 2048x2048 window to
    stay within OpenCV's SHRT_MAX (32767) limit.
    Returns None on failure (logged as warning).
    """
    try:
        # Try rasterio first (preferred for GeoTIFF / scientific formats)
        try:
            import rasterio
            from rasterio.windows import Window
            with rasterio.open(path) as ds:
                h, w = ds.shape
                if h > 4096 or w > 4096:
                    max_dim = 2048
                    r0 = (h - max_dim) // 2 if h > max_dim else 0
                    c0 = (w - max_dim) // 2 if w > max_dim else 0
                    win = Window(c0, r0, min(w, max_dim), min(h, max_dim))
                    data = ds.read(1, window=win).astype(np.float32)
                else:
                    data = ds.read(1).astype(np.float32)
                vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
                if vmax > vmin:
                    data = (data - vmin) / (vmax - vmin)
                else:
                    data = np.zeros_like(data)
                return data
        except Exception:
            pass

        # Fallback: OpenCV
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape[0] > 4096 or img.shape[1] > 4096:
                max_dim = 2048
                r0 = (img.shape[0] - max_dim) // 2 if img.shape[0] > max_dim else 0
                c0 = (img.shape[1] - max_dim) // 2 if img.shape[1] > max_dim else 0
                img = img[r0 : r0 + max_dim, c0 : c0 + max_dim]
            vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
            if vmax > vmin:
                return (img.astype(np.float32) - vmin) / (vmax - vmin)
            return img.astype(np.float32) / 255.0

        logger.warning("Could not load image: %s", path)
        return None

    except Exception as exc:
        logger.warning("Error loading %s: %s", path, exc)
        return None


def _save_image_tif(image: np.ndarray, path: Path) -> None:
    """Save a float32 2D image as a single-band GeoTIFF (no CRS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rasterio
        from rasterio.transform import from_bounds
        h, w = image.shape[:2]
        transform = from_bounds(0, 0, w, h, w, h)
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=h, width=w,
            count=1,
            dtype="float32",
            crs=None,
            transform=transform,
        ) as ds:
            ds.write(image.astype(np.float32), 1)
    except Exception:
        # Fallback: save as uint8 PNG-style via OpenCV (normalise to 0-255)
        img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(path.with_suffix(".png")), img_u8)
        logger.warning("rasterio unavailable; saved %s as PNG.", path)


def _write_gt_json(
    gt_path: Path,
    pair_id: str,
    anchor_set: AnchorSet,
    tgt_pts: np.ndarray,
    params: TransformParams,
) -> None:
    """Write hidden GT file: source + target anchor coordinates.

    Schema per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §4.2:
      pair_id, points:[{id, src_x, src_y, tgt_x, tgt_y, feature_class}],
      transform_params, anchor_extraction_phase
    """
    points = []
    src_pts = anchor_set.as_numpy()  # (N, 2) col/row
    for i, anchor in enumerate(anchor_set.anchors):
        tgt_col, tgt_row = float(tgt_pts[i, 0]), float(tgt_pts[i, 1])
        points.append({
            "id": anchor.id,
            "src_x": anchor.src_x,
            "src_y": anchor.src_y,
            "tgt_x": tgt_col,
            "tgt_y": tgt_row,
            "feature_class": anchor.feature_class,
            "gradient_magnitude": anchor.gradient_magnitude,
        })

    gt_record = {
        "pair_id": pair_id,
        "points": points,
        "n_gt": len(points),
        "transform_params": params.to_dict(),
        "anchor_extraction_phase": anchor_set.extraction_phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(gt_record, f, indent=2)
    logger.debug("GT written: %s (%d points)", gt_path, len(points))


def _append_manifest(manifest_path: Path, record: dict) -> None:
    """Atomically append or update one record in the JSONL manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if d.get("pair_id") != record.get("pair_id"):
                            existing.append(line.strip())
                    except Exception:
                        existing.append(line.strip())
    existing.append(json.dumps(record, default=str))
    with open(manifest_path, "w") as f:
        for l in existing:
            f.write(l + "\n")


def _append_failure(failures_path: Path, pair_id: str, stage: str, reason: str) -> None:
    """Write a failure record to failures.jsonl (per PIPELINE.md §8)."""
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "pair_id": pair_id,
        "stage": stage,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(failures_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Phase-specific configuration builders
# ---------------------------------------------------------------------------

def _build_phase_config(base_cfg: dict, phase: int) -> dict:
    """Override transform enables for the given benchmark phase."""
    cfg = {k: v for k, v in base_cfg.items()}  # shallow copy top-level
    # Deep-copy transforms block
    import copy
    cfg = copy.deepcopy(base_cfg)

    tf = cfg.get("synthetic_benchmark", {}).get("transforms", {})
    phases_cfg = cfg.get("synthetic_benchmark", {}).get("phases", {})
    phase_key = f"phase_{phase}"
    phase_spec = phases_cfg.get(phase_key, {})

    enabled_transforms = phase_spec.get("transformations", ["translation"])

    # Enable/disable transforms per phase spec
    tf.get("translation", {})["enabled"] = "translation" in enabled_transforms
    tf.get("rotation", {})["enabled"] = "rotation" in enabled_transforms
    tf.get("scale", {})["enabled"] = "scale" in enabled_transforms
    tf.get("illumination", {})["enabled"] = "illumination" in enabled_transforms
    tf.get("sensor_simulation", {})["enabled"] = "sensor_simulation" in enabled_transforms

    # Phase 1: force exactly 5 GT anchors (smoke test)
    if phase == 1:
        cfg["synthetic_benchmark"]["anchors"]["target_count"] = 5
        cfg["synthetic_benchmark"]["anchors"]["min_count"] = 3

    return cfg


def _get_seeds(phase_spec: dict, global_seed: int) -> List[int]:
    """Return seed list for a phase."""
    seeds_val = phase_spec.get("seeds", [global_seed])
    if isinstance(seeds_val, int):
        # seeds is a count; generate deterministically from global seed
        rng = np.random.default_rng(global_seed)
        return [int(s) for s in rng.integers(0, 2**31, size=seeds_val)]
    return [int(s) for s in seeds_val]


# ---------------------------------------------------------------------------
# Core pair generator
# ---------------------------------------------------------------------------

def generate_pair(
    source_image: np.ndarray,
    source_path: Path,
    pair_id: str,
    seed: int,
    phase: int,
    phase_cfg: dict,
    out_dir: Path,
    gt_dir: Path,
    manifest_path: Path,
    failures_path: Path,
    config_hash: str,
    force: bool = False,
) -> bool:
    """Generate one synthetic pair (source + target + hidden GT).

    Returns True on success, False on gate failure (logged to failures.jsonl).
    """
    src_tif = out_dir / f"{pair_id}_src.tif"
    tgt_tif = out_dir / f"{pair_id}_tgt.tif"
    gt_json = gt_dir / f"{pair_id}_gt.json"

    # Checkpointing: skip if outputs already exist and --force not set
    if not force and src_tif.exists() and tgt_tif.exists() and gt_json.exists():
        logger.info("SKIP %s — already generated (use --force to regenerate).", pair_id)
        return True

    sb_cfg = phase_cfg.get("synthetic_benchmark", {})
    anchor_cfg = {**sb_cfg.get("anchors", {})}
    # Phase 1 override already applied in phase_cfg by _build_phase_config()
    config_for_anchors = {"anchors": anchor_cfg}

    rng = np.random.default_rng(seed)

    # --- Step 1: Extract GT anchors ---
    try:
        anchor_set = extract_anchors(
            image=source_image,
            pair_id=pair_id,
            config=config_for_anchors,
            rng=rng,
        )
    except RuntimeError as exc:
        logger.warning("Anchor gate failed for %s: %s", pair_id, exc)
        _append_failure(failures_path, pair_id, "S-ANCHOR", str(exc))
        return False

    # --- Step 2: Generate synthetic target image ---
    transforms_cfg = sb_cfg.get("transforms", {})
    full_config_for_transform = {"transforms": transforms_cfg}

    synthetic_img, params, M = generate_synthetic_pair(
        source_image=source_image,
        config=full_config_for_transform,
        pair_id=pair_id,
        seed=seed,
    )

    # --- Step 3: Map GT anchors to target space ---
    src_pts = anchor_set.as_numpy()  # (N, 2) col/row
    tgt_pts = transform_gt_points(src_pts, M)  # (N, 2) col/row in target

    # Validate all GT target points are within image bounds (with 1px margin)
    h, w = source_image.shape[:2]
    in_bounds = (
        (tgt_pts[:, 0] >= 0) & (tgt_pts[:, 0] < w) &
        (tgt_pts[:, 1] >= 0) & (tgt_pts[:, 1] < h)
    )
    valid_count = int(in_bounds.sum())
    if valid_count < anchor_set.as_numpy().shape[0] * 0.5:
        reason = (
            f"Only {valid_count}/{len(tgt_pts)} GT anchors remain in-bounds "
            "after transformation. Pair skipped."
        )
        logger.warning(reason)
        _append_failure(failures_path, pair_id, "S-GT-BOUNDS", reason)
        return False

    # Filter to in-bounds anchors only
    valid_anchors = [a for a, ok in zip(anchor_set.anchors, in_bounds.tolist()) if ok]
    tgt_pts_valid = tgt_pts[in_bounds]

    # Rebuild AnchorSet with valid anchors
    from src.synthetic.anchors import AnchorSet as _AnchorSet
    valid_anchor_set = _AnchorSet(
        pair_id=pair_id,
        image_shape=anchor_set.image_shape,
        anchors=valid_anchors,
        extraction_phase=anchor_set.extraction_phase,
        n_grid_cells=anchor_set.n_grid_cells,
    )

    # --- Step 4: Write outputs ---
    _save_image_tif(source_image, src_tif)
    _save_image_tif(synthetic_img, tgt_tif)
    _write_gt_json(gt_json, pair_id, valid_anchor_set, tgt_pts_valid, params)

    # --- Step 5: Append manifest record ---
    manifest_record = {
        "pair_id": pair_id,
        "base_image": str(source_path.relative_to(source_path.parent.parent) if source_path.parent.parent.exists() else source_path),
        "source_image": str(src_tif),
        "synthetic_image": str(tgt_tif),
        "src_processed": str(src_tif),
        "ref_processed": str(tgt_tif),
        "split": "train",
        "gt_points_file": str(gt_json),
        "benchmark_phase": phase,
        "random_seed": seed,
        "n_gt_points": len(valid_anchors),
        "anchor_phase": valid_anchor_set.extraction_phase,
        "parameters": {
            "scale_factor": params.scale_factor,
            "rotation_deg": params.rotation_deg,
            "translation_px": params.translation_px,
            "illumination_gamma": params.illumination_gamma,
            "mtf_blur_sigma": params.mtf_blur_sigma,
            "pushbroom_stripe_amplitude": params.pushbroom_stripe_amplitude,
            "shadow_extension_factor": params.shadow_extension_factor,
        },
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_manifest(manifest_path, manifest_record)

    logger.info(
        "Generated pair %s (phase=%d seed=%d n_gt=%d scale=%.3f rot=%.2f°).",
        pair_id, phase, seed, len(valid_anchors),
        params.scale_factor, params.rotation_deg,
    )
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GT benchmark pairs (Phase 10 v3.0)."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_benchmark.yaml"),
        help="Path to synthetic_benchmark.yaml config.",
    )
    parser.add_argument(
        "--images", type=Path, default=Path("data/raw/"),
        help="Directory containing source images (.tif, .img, .png).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/synthetic/"),
        help="Output directory for synthetic images.",
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], default=1,
        help="Benchmark phase to generate (1=smoke, 2=geometric, 3=photometric, 4=full).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate pairs even if output files already exist.",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Limit number of source images to process (useful for quick runs).",
    )
    parser.add_argument(
        "--synthetic-base", action="store_true",
        help="Synthesize a realistic lunar terrain base image if no source images are found.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Load config ---
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 2

    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 2

    sb_cfg = cfg.get("synthetic_benchmark", {})
    if not sb_cfg:
        logger.error("'synthetic_benchmark' block missing from config.")
        return 2

    # Check phase is enabled in config
    phase_key = f"phase_{args.phase}"
    phases_cfg = sb_cfg.get("phases", {})
    phase_spec = phases_cfg.get(phase_key, {})
    if not phase_spec.get("enabled", False) and args.phase != 1:
        logger.warning(
            "Phase %d is not enabled in config (phases.%s.enabled=false). "
            "Set enabled=true in synthetic_benchmark.yaml to run non-smoke phases.",
            args.phase, phase_key,
        )
        # Do not exit — allow override via CLI for development

    # --- Resolve paths ---
    out_dir = args.out
    gt_dir = out_dir / "gt"
    manifest_path = out_dir / "synthetic_manifest.jsonl"
    failures_path = out_dir / "failures.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Build phase-specific config (enables/disables transforms per phase)
    phase_cfg = _build_phase_config(cfg, args.phase)
    config_hash = _hash_config(phase_cfg)
    global_seed = cfg.get("global", {}).get("seed", 42)

    # --- Find source images ---
    source_paths = []
    if args.images.exists():
        source_paths = _find_source_images(args.images)

    if not source_paths:
        if args.synthetic_base or not args.images.exists() or len(list(args.images.iterdir())) == 0:
            logger.info("Synthesizing realistic lunar base scene (512x512) for synthetic benchmark...")
            base_dir = out_dir / "base_scenes"
            base_dir.mkdir(parents=True, exist_ok=True)
            synth_base_path = base_dir / "synthetic_lunar_base.tif"
            
            # Generate realistic multi-scale lunar scene
            rng = np.random.default_rng(global_seed)
            height, width = 512, 512
            y, x = np.mgrid[0:height, 0:width]
            terrain = (
                0.45
                + 0.12 * np.sin(x / 32.0) * np.cos(y / 32.0)
                + 0.08 * np.sin(x / 14.0 + y / 18.0)
                + 0.05 * rng.standard_normal((height, width))
            )
            craters = [
                (width * 0.35, height * 0.35, 45.0, 0.4),
                (width * 0.70, height * 0.60, 30.0, 0.5),
                (width * 0.25, height * 0.75, 20.0, 0.6),
                (width * 0.80, height * 0.25, 25.0, 0.5),
                (width * 0.50, height * 0.70, 15.0, 0.7),
                (width * 0.60, height * 0.40, 18.0, 0.6),
                (width * 0.15, height * 0.20, 12.0, 0.7),
            ]
            for cx, cy, r, depth in craters:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                bowl = dist < r
                terrain[bowl] -= depth * (1.0 - (dist[bowl] / r) ** 2)
                rim = (dist >= r) & (dist < r + 5.0)
                terrain[rim] += depth * 0.4 * (1.0 - (dist[rim] - r) / 5.0)

            p2, p98 = np.percentile(terrain, (1.0, 99.0))
            synth_img = np.clip((terrain - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
            _save_image_tif(synth_img, synth_base_path)
            source_paths = [synth_base_path]
        else:
            logger.error("No source images found in %s.", args.images)
            return 3

    if args.max_pairs is not None:
        source_paths = source_paths[: args.max_pairs]

    logger.info("Found %d source image(s). Generating phase %d pairs...", len(source_paths), args.phase)

    # --- Determine seeds ---
    seeds = _get_seeds(phase_spec, global_seed)
    if args.phase == 1:
        seeds = [42]  # Phase 1 always uses seed=42 per architecture doc

    logger.info("Seeds for phase %d: %s", args.phase, seeds)

    # --- Generate pairs ---
    n_success = 0
    n_failed = 0

    for src_path in source_paths:
        source_image = _load_image(src_path)
        if source_image is None:
            logger.warning("Skipping unreadable image: %s", src_path)
            continue

        # Minimum image size check
        h, w = source_image.shape[:2]
        if min(h, w) < 128:
            logger.warning("Image too small (%dx%d), skipping: %s", h, w, src_path)
            continue

        base_name = src_path.stem

        for seed in seeds:
            pair_id = f"synth_p{args.phase}_{base_name}_s{seed}"
            # Sanitise pair_id
            pair_id = pair_id.replace(" ", "_")[:128]

            success = generate_pair(
                source_image=source_image,
                source_path=src_path,
                pair_id=pair_id,
                seed=seed,
                phase=args.phase,
                phase_cfg=phase_cfg,
                out_dir=out_dir,
                gt_dir=gt_dir,
                manifest_path=manifest_path,
                failures_path=failures_path,
                config_hash=config_hash,
                force=args.force,
            )
            if success:
                n_success += 1
            else:
                n_failed += 1

    # --- Leakage audit: verify no GT file exists outside gt_dir ---
    leaked = []
    for gt_file in manifest_path.parent.rglob("*_gt.json"):
        if gt_dir not in gt_file.parents:
            leaked.append(gt_file)
    if leaked:
        logger.error(
            "LEAKAGE AUDIT FAILED: GT files found outside gt_dir (%s): %s",
            gt_dir, leaked,
        )
        return 4

    # --- Summary ---
    logger.info(
        "Phase %d generation complete: %d succeeded, %d failed. "
        "Manifest: %s",
        args.phase, n_success, n_failed, manifest_path,
    )

    if n_success == 0:
        logger.error("No pairs generated successfully.")
        return 1
    if n_failed > 0:
        logger.warning("%d pairs failed (see %s).", n_failed, failures_path)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

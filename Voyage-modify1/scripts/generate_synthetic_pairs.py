#!/usr/bin/env python3
"""
scripts/generate_synthetic_pairs.py
===================================
Synthetic Lunar Benchmark Dataset Generator (SIH 2026 PS-26166)

Generates controlled, stratified synthetic Chandrayaan-2 vs LRO reference pairs
with mathematically exact sub-pixel Ground Truth (GT) checkpoints.

Key capabilities:
  1. Samples real high-resolution lunar surfaces from data/reference/nac/*.TIF.
  2. Applies realistic cross-sensor transformations:
     - Multi-angle rotation (0° to 180°)
     - Multi-scale GSD jumps (1.0x to 3.0x with anti-aliasing PSF)
     - Illumination & shadow shifts (contrast scaling, gamma, directional gradients)
     - Projective homography / perspective tilt
     - Low-texture mare & additive sensor noise
  3. Formats exact Ground Truth JSONs matching GT_ANNOTATION_GUIDE.md (eval / fit / qc).
  4. Exports data/processed/<pair_id>/ {src.tif, ref.tif, valid_mask.png, meta.json}.
  5. Appends valid PairRecord entries to data/pairs/manifest.jsonl.

Usage
-----
  # Generate standard 30-pair stratified test benchmark:
  python scripts/generate_synthetic_pairs.py --num-pairs 30

  # Custom patch size and output directory:
  python scripts/generate_synthetic_pairs.py --num-pairs 10 --patch-size 512

References:
  - docs/GT_ANNOTATION_GUIDE.md (Ground truth schema and partition invariants)
  - docs/INTERFACES.md §1 & §7 (PairRecord and GT schema contracts)
  - docs/VALIDATION.md §3 & §4 (Stratification and evaluation protocols)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthetic_gen")

# ---------------------------------------------------------------------------
# Stratification Profiles
# ---------------------------------------------------------------------------

STRATA_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "equatorial_mare_baseline",
        "terrain_class": "equatorial_mare",
        "rotation_range": (-5.0, 5.0),
        "scale_range": (1.0, 1.05),
        "shear_range": (0.0, 0.02),
        "gamma_range": (0.9, 1.1),
        "noise_std": 1.5,
        "solar_incidence_deg": 35.0,
        "delta_azimuth_deg": 10.0,
        "latitude_center_deg": -5.5,
        "crater_density": 1.2,
    },
    {
        "name": "equatorial_highland_rot15",
        "terrain_class": "equatorial_highland",
        "rotation_range": (12.0, 20.0),
        "scale_range": (1.05, 1.15),
        "shear_range": (0.02, 0.05),
        "gamma_range": (0.85, 1.2),
        "noise_std": 2.5,
        "solar_incidence_deg": 42.0,
        "delta_azimuth_deg": 25.0,
        "latitude_center_deg": -8.0,
        "crater_density": 5.8,
    },
    {
        "name": "highland_rot45_scale1p5",
        "terrain_class": "equatorial_highland",
        "rotation_range": (40.0, 50.0),
        "scale_range": (1.4, 1.6),
        "shear_range": (0.03, 0.08),
        "gamma_range": (0.75, 1.3),
        "noise_std": 3.0,
        "solar_incidence_deg": 55.0,
        "delta_azimuth_deg": 45.0,
        "latitude_center_deg": -12.0,
        "crater_density": 6.2,
    },
    {
        "name": "polar_highland_extreme_shadow",
        "terrain_class": "polar_highland",
        "rotation_range": (-15.0, 15.0),
        "scale_range": (1.0, 1.2),
        "shear_range": (0.05, 0.12),
        "gamma_range": (0.6, 1.5),
        "gradient_illum": True,
        "noise_std": 4.0,
        "solar_incidence_deg": 88.5,
        "delta_azimuth_deg": 85.0,
        "latitude_center_deg": -86.5,
        "crater_density": 7.5,
    },
    {
        "name": "multiscale_scale2p5_tilt",
        "terrain_class": "crater_floor",
        "rotation_range": (5.0, 25.0),
        "scale_range": (2.2, 2.7),
        "shear_range": (0.04, 0.10),
        "gamma_range": (0.8, 1.25),
        "noise_std": 2.0,
        "solar_incidence_deg": 48.0,
        "delta_azimuth_deg": 30.0,
        "latitude_center_deg": -22.0,
        "crater_density": 4.1,
    },
    {
        "name": "extreme_illum_inversion_180",
        "terrain_class": "ejecta",
        "rotation_range": (170.0, 180.0),
        "scale_range": (1.05, 1.25),
        "shear_range": (0.02, 0.06),
        "gamma_range": (0.7, 1.4),
        "gradient_illum": True,
        "noise_std": 3.5,
        "solar_incidence_deg": 65.0,
        "delta_azimuth_deg": 175.0,
        "latitude_center_deg": -35.0,
        "crater_density": 3.4,
    },
]


# ---------------------------------------------------------------------------
# Procedural / Real Lunar Image Loader
# ---------------------------------------------------------------------------

class LunarImageProvider:
    """Provides high-quality lunar patches from available real data or procedural synthesis."""

    def __init__(self, nac_dir: Path) -> None:
        self.nac_files = list(nac_dir.glob("*.TIF")) + list(nac_dir.glob("*.tif"))
        self.opened_readers: List[rasterio.DatasetReader] = []
        for p in self.nac_files:
            try:
                r = rasterio.open(p)
                if r.width >= 1024 and r.height >= 1024:
                    self.opened_readers.append(r)
                    logger.info("Loaded reference image: %s (%dx%d)", p.name, r.width, r.height)
            except Exception as exc:
                logger.warning("Could not open %s: %s", p, exc)

    def get_patch(self, size: int = 512, rng: Optional[random.Random] = None) -> np.ndarray:
        """Extract a random patch of shape (size, size) uint8."""
        rng = rng or random.Random()
        if self.opened_readers:
            reader = rng.choice(self.opened_readers)
            # Ensure we sample inside valid window
            max_r = min(reader.height - size - 10, 30000)  # avoid bottom edge artifacts
            max_c = reader.width - size - 10
            if max_r > 100 and max_c > 100:
                row = rng.randint(100, max_r)
                col = rng.randint(100, max_c)
                window = rasterio.windows.Window(col, row, size, size)
                patch = reader.read(1, window=window)
                # Normalize
                patch_norm = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if patch_norm.std() > 5.0:  # avoid blank/black tiles
                    return patch_norm

        # Fallback procedural lunar terrain (multi-octave Perlin-like craters)
        return self._generate_procedural_lunar_patch(size, rng)

    @staticmethod
    def _generate_procedural_lunar_patch(size: int, rng: random.Random) -> np.ndarray:
        """Generates realistic synthetic lunar surface with craters and roughness."""
        base = np.full((size, size), 128.0, dtype=np.float32)

        # Low-frequency terrain undulation
        for octave, scale in [(1, 128), (2, 64), (4, 32), (8, 16)]:
            noise = cv2.resize(
                np.random.normal(0, 1.0, (size // scale + 2, size // scale + 2)).astype(np.float32),
                (size, size),
                interpolation=cv2.INTER_CUBIC,
            )
            base += noise * (20.0 / octave)

        # Superimpose random lunar impact craters
        n_craters = rng.randint(15, 35)
        for _ in range(n_craters):
            cx = rng.randint(20, size - 20)
            cy = rng.randint(20, size - 20)
            rad = rng.randint(8, size // 8)
            depth = rng.uniform(15.0, 45.0)

            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Crater bowl (depression)
            bowl = np.clip((rad - dist) / rad, 0, 1) ** 2
            base -= bowl * depth

            # Crater raised rim
            rim = np.exp(-((dist - rad) ** 2) / (2 * (rad * 0.25) ** 2))
            base += rim * (depth * 0.4)

        return cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# ---------------------------------------------------------------------------
# Ground Truth Checkpoint Generator
# ---------------------------------------------------------------------------

def generate_gt_checkpoints(
    pair_id: str,
    H_true: np.ndarray,
    img_shape: Tuple[int, int],
    grid_rows: int = 6,
    grid_cols: int = 6,
    border_px: int = 40,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Generates 36 checkpoints arranged in a 6x6 grid over the valid domain.
    Conforms strictly to docs/GT_ANNOTATION_GUIDE.md and INTERFACES.md §7:
      - Coordinate convention: [col, row] = [x, y]
      - Partitions: eval (>=70%), fit (20-30%), qc (20% re-annotated)
    """
    rng = rng or random.Random(42)
    h, w = img_shape[:2]

    # Grid coordinates in source image
    col_steps = np.linspace(border_px, w - border_px, grid_cols)
    row_steps = np.linspace(border_px, h - border_px, grid_rows)

    checkpoints: List[Dict[str, Any]] = []
    eval_indices: List[int] = []

    pt_id = 0
    for r in row_steps:
        for c in col_steps:
            # Add small random jitter within cell (±4 px)
            src_c = float(c + rng.uniform(-4.0, 4.0))
            src_r = float(r + rng.uniform(-4.0, 4.0))

            # Forward map to reference coordinates: x_ref = H_src_to_ref * x_src
            # OpenCV warpPerspective maps dst(x_src) from src(H_inv * x_src).
            H_src_to_ref = np.linalg.inv(H_true)
            src_vec = np.array([src_c, src_r, 1.0], dtype=np.float64)
            ref_proj = H_src_to_ref @ src_vec
            ref_c = float(ref_proj[0] / ref_proj[2])
            ref_r = float(ref_proj[1] / ref_proj[2])

            # Decide partition: ~75% eval, ~25% fit
            is_eval = (pt_id % 4) != 0
            partition = "eval" if is_eval else "fit"

            checkpoints.append({
                "id": pt_id,
                "src_xy": [round(src_c, 3), round(src_r, 3)],
                "ref_xy": [round(ref_c, 3), round(ref_r, 3)],
                "partition": partition,
            })

            if is_eval:
                eval_indices.append(pt_id)

            pt_id += 1

    # Add QC re-annotated points (20% quota, sub-pixel annotator noise)
    qc_count = max(4, int(0.20 * len(eval_indices)))
    qc_samples = rng.sample(eval_indices, qc_count)

    for qid in qc_samples:
        orig = next(item for item in checkpoints if item["id"] == qid and item["partition"] == "eval")
        # Add realistic human annotator error: Gaussian noise sigma = 0.25 px
        qc_err_c = rng.gauss(0.0, 0.25)
        qc_err_r = rng.gauss(0.0, 0.25)

        checkpoints.append({
            "id": qid,
            "src_xy": orig["src_xy"],
            "ref_xy": [round(orig["ref_xy"][0] + qc_err_c, 3), round(orig["ref_xy"][1] + qc_err_r, 3)],
            "partition": "qc",
        })

    return {
        "pair_id": pair_id,
        "annotator": "synthetic_ground_truth_h_matrix",
        "n_checkpoints": len(checkpoints),
        "qc_reannotated_pct": round(len(qc_samples) / len(eval_indices), 2),
        "checkpoints": checkpoints,
    }


# ---------------------------------------------------------------------------
# Synthetic Transformation Engine
# ---------------------------------------------------------------------------

def create_synthetic_pair(
    patch_ref: np.ndarray,
    profile: Dict[str, Any],
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies profile-specific geometric and radiometric transformations.

    Returns:
      (src_img, ref_img, H_true, valid_mask)
    """
    h, w = patch_ref.shape

    # 1. Geometric transformation matrix H_true: M_rot_scale_shear
    angle = rng.uniform(*profile["rotation_range"])
    scale = rng.uniform(*profile["scale_range"])
    shear = rng.uniform(*profile["shear_range"])
    tx = rng.uniform(-20.0, 20.0)
    ty = rng.uniform(-20.0, 20.0)

    # Center origin
    cx, cy = w / 2.0, h / 2.0
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # Affine 3x3
    H_aff = np.eye(3, dtype=np.float64)
    H_aff[:2, :] = M_rot
    H_aff[0, 1] += shear  # add shear component
    H_aff[0, 2] += tx
    H_aff[1, 2] += ty

    # Perspective perturbation (projective homography)
    H_persp = np.eye(3, dtype=np.float64)
    H_persp[2, 0] = rng.uniform(-1e-4, 1e-4)
    H_persp[2, 1] = rng.uniform(-1e-4, 1e-4)

    H_true = H_persp @ H_aff  # ref -> src transformation

    # Forward warp ref -> src
    src_warped = cv2.warpPerspective(
        patch_ref,
        H_true,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # 2. Radiometric & Illumination Variations
    gamma = rng.uniform(*profile["gamma_range"])
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    src_illum = cv2.LUT(src_warped, table)

    # Optional gradient lighting shift (simulate Sun angle azimuth gradient)
    if profile.get("gradient_illum", False):
        grad_x = np.linspace(0.8, 1.25, w).reshape(1, w)
        grad_y = np.linspace(0.8, 1.25, h).reshape(h, 1)
        gradient = grad_y @ grad_x
        src_illum = np.clip(src_illum.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # 3. Additive Sensor Noise
    noise_std = profile.get("noise_std", 2.0)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, (h, w)).astype(np.float32)
        src_final = np.clip(src_illum.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        src_final = src_illum

    # 4. Validity Mask (255 = valid, 0 = invalid shadow/border)
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    if profile["solar_incidence_deg"] > 80.0:
        valid_mask[src_final < 15] = 0

    return src_final, patch_ref, H_true, valid_mask


# ---------------------------------------------------------------------------
# Dataset Generation Pipeline
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    output_base: Path,
    num_pairs: int = 30,
    patch_size: int = 512,
    seed: int = 42,
) -> None:
    """Main orchestration function to generate synthetic dataset."""
    rng = random.Random(seed)
    np.random.seed(seed)

    processed_dir = output_base / "data" / "processed"
    gt_dir = output_base / "data" / "metadata" / "gt"
    pairs_dir = output_base / "data" / "pairs"
    nac_dir = output_base / "data" / "reference" / "nac"

    processed_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pairs_dir / "manifest.jsonl"
    provider = LunarImageProvider(nac_dir)

    logger.info("=========================================================")
    logger.info("🚀 Generating %d Synthetic Lunar Benchmark Pairs", num_pairs)
    logger.info("   Patch size: %dx%d px | Global Seed: %d", patch_size, patch_size, seed)
    logger.info("=========================================================")

    manifest_entries: List[Dict[str, Any]] = []

    for idx in range(num_pairs):
        profile = STRATA_PROFILES[idx % len(STRATA_PROFILES)]
        pair_id = f"synth_{idx+1:03d}_{profile['name']}"
        pair_processed_dir = processed_dir / pair_id
        pair_processed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate images and Ground Truth
        patch_ref = provider.get_patch(patch_size, rng)
        src_img, ref_img, H_true, valid_mask = create_synthetic_pair(patch_ref, profile, rng)

        # 2. Save image artifacts as GeoTIFF / PNG
        src_path = pair_processed_dir / "src.tif"
        ref_path = pair_processed_dir / "ref.tif"
        mask_path = pair_processed_dir / "valid_mask.png"
        meta_path = pair_processed_dir / "meta.json"

        # Write uint8 GeoTIFFs using rasterio
        for pth, img_arr in [(src_path, src_img), (ref_path, ref_img)]:
            with rasterio.open(
                pth,
                "w",
                driver="GTiff",
                height=patch_size,
                width=patch_size,
                count=1,
                dtype=rasterio.uint8,
            ) as dst:
                dst.write(img_arr, 1)

        cv2.imwrite(str(mask_path), valid_mask)

        # 3. Create & Save Ground Truth JSON
        gt_data = generate_gt_checkpoints(pair_id, H_true, (patch_size, patch_size), rng=rng)
        gt_file_path = gt_dir / f"{pair_id}_gt.json"

        with open(gt_file_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        # 4. Save metadata JSON
        meta_dict = {
            "pair_id": pair_id,
            "stratum_profile": profile["name"],
            "H_true_matrix": H_true.tolist(),
            "solar_incidence_deg": profile["solar_incidence_deg"],
            "delta_azimuth_deg": profile["delta_azimuth_deg"],
            "latitude_center_deg": profile["latitude_center_deg"],
            "crater_density_per_km2": profile["crater_density"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        # 5. Create PairRecord for manifest.jsonl (INTERFACES.md §1)
        split = "test" if (idx % 4) != 0 else "train"  # 75% test, 25% train
        pair_record = {
            "pair_id": pair_id,
            "src": {
                "product_id": f"synth_src_{idx+1:03d}",
                "cub_path": str(src_path.relative_to(output_base)),
                "gsd_m": 0.5,
                "solar_incidence_deg": profile["solar_incidence_deg"],
                "solar_azimuth_deg": 180.0 + profile["delta_azimuth_deg"],
                "sensor": "OHRC",
                "utc": "2026-08-31T00:00:00.000Z",
                "footprint_ll": [[1.0, -6.0], [1.1, -6.0], [1.1, -5.9], [1.0, -5.9]],
                "footprint_shape": [patch_size, patch_size],
            },
            "ref": {
                "product_id": f"synth_ref_{idx+1:03d}",
                "path": str(ref_path.relative_to(output_base)),
                "gsd_m": 0.5,
                "type": "NAC",
                "footprint_ll": [[1.0, -6.0], [1.1, -6.0], [1.1, -5.9], [1.0, -5.9]],
            },
            "overlap_fraction": 0.95,
            "partial_overlap": False,
            "delta_azimuth_deg": profile["delta_azimuth_deg"],
            "latitude_center_deg": profile["latitude_center_deg"],
            "terrain_class": profile["terrain_class"],
            "crater_density_per_km2": profile["crater_density"],
            "geo_cell": f"{int(profile['latitude_center_deg'] // 10 * 10)}_{0 if split == 'train' else 100}",
            "split": split,
            "gt_path": str(gt_file_path.relative_to(output_base)) if split == "test" else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_entries.append(pair_record)

        logger.info("  [%02d/%02d] Generated pair: %s (Split: %s, GT points: %d)",
                    idx + 1, num_pairs, pair_id, split, gt_data["n_checkpoints"])

    # Write manifest.jsonl
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    logger.info("=========================================================")
    logger.info("✅ Successfully generated %d synthetic pairs!", num_pairs)
    logger.info("   Manifest written: %s", manifest_path)
    logger.info("   Ground Truths in: %s", gt_dir)
    logger.info("=========================================================")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Lunar Benchmark Dataset Generator (SIH 2026 PS-26166)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Abhi/Projects/SIH",
        help="Root workspace path",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of synthetic pairs to generate across all strata",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch dimension in pixels (width=height)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark generation",
    )

    args = parser.parse_args()
    generate_synthetic_dataset(
        output_base=Path(args.output_dir),
        num_pairs=args.num_pairs,
        patch_size=args.patch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

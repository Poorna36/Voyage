#!/usr/bin/env python3
"""
scripts/synthetic_gt_check.py
=============================
Synthetic Ground Truth Sanity Check (VALIDATION.md §7).

Workflow:
  1. Synthesizes a realistic lunar terrain image with craters, ridges, and texture.
  2. Applies a known geometric ground truth transform T:
     - Rotation = 2.0 degrees
     - Scale = 1.05
     - Translation = (50.0, 50.0) pixels
  3. Executes the SIFT correspondence matching + L3 spatial selection +
     DEGENSAC geometric verification + L5 sub-pixel refinement pipeline.
  4. Evaluates recovered transform vs exact ground truth on a uniform 6x6 test grid.
  5. Asserts RMSE < 0.50 px (sub-pixel precision pass criteria).

Usage:
  python scripts/synthetic_gt_check.py [--out results/synthetic_gt_check]

Exit Codes:
  0: Pass — Recovered transform achieves RMSE < 0.5 px vs ground truth.
  1: Fail — Transform error exceeds 0.5 px or pipeline failure.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from src.evaluation.metrics import rmse
from src.matching.sift import SIFTMatcher
from src.provenance import build_provenance, set_global_seed
from src.refinement.local import refine_inliers
from src.registration.ladder import model_ladder
from src.selection.spatial import confidence_filter, coverage_greedy, grid_cap, one_to_one


def generate_synthetic_lunar_scene(
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Generate textured lunar scene with multi-scale craters and surface relief."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]

    # Multi-frequency undulating mare/highland terrain
    terrain = (
        0.45
        + 0.12 * np.sin(x / 32.0) * np.cos(y / 32.0)
        + 0.08 * np.sin(x / 14.0 + y / 18.0)
        + 0.05 * rng.standard_normal((height, width))
    )

    # Add craters of varying sizes
    crater_configs = [
        (width * 0.35, height * 0.35, 45.0, 0.4),
        (width * 0.70, height * 0.60, 30.0, 0.5),
        (width * 0.25, height * 0.75, 20.0, 0.6),
        (width * 0.80, height * 0.25, 25.0, 0.5),
        (width * 0.50, height * 0.70, 15.0, 0.7),
        (width * 0.60, height * 0.40, 18.0, 0.6),
        (width * 0.15, height * 0.20, 12.0, 0.7),
    ]

    for cx, cy, r, depth in crater_configs:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        bowl = dist < r
        terrain[bowl] -= depth * (1.0 - (dist[bowl] / r) ** 2)
        rim = (dist >= r) & (dist < r + 5.0)
        terrain[rim] += depth * 0.4 * (1.0 - (dist[rim] - r) / 5.0)

    # Normalize to [0.0, 1.0]
    p2, p98 = np.percentile(terrain, (1.0, 99.0))
    img = np.clip((terrain - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
    return img


def create_ground_truth_transform(
    rot_deg: float = 2.0,
    scale: float = 1.05,
    tx: float = 50.0,
    ty: float = 50.0,
    center: Tuple[float, float] = (256.0, 256.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build affine 3x3 transform matrix mapping source (col, row) -> reference (col, row).

    T = T_trans @ T_rot_scale
    """
    cx, cy = center
    rad = math.radians(rot_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Forward mapping: ref = T @ src
    # Matrix about center:
    # 1. Translate to origin (-cx, -cy)
    # 2. Scale & Rotate
    # 3. Translate back and add shift (cx + tx, cy + ty)
    T = np.array([
        [scale * cos_a, -scale * sin_a, cx + tx - scale * (cx * cos_a - cy * sin_a)],
        [scale * sin_a,  scale * cos_a, cy + ty - scale * (cx * sin_a + cy * cos_a)],
        [0.0,            0.0,           1.0],
    ], dtype=np.float64)

    # Affine 2x3 for cv2.warpAffine
    M_affine = T[:2, :]
    return T, M_affine


def run_synthetic_gt_check(out_dir: Optional[Path] = None, seed: int = 42) -> Tuple[bool, float, dict]:
    """
    Execute full synthetic sanity check.

    Returns:
      (passed: bool, rmse_px: float, summary_dict: dict)
    """
    set_global_seed(seed)
    start_time = time.perf_counter()

    # Step 1: Synthesize images
    src_img = generate_synthetic_lunar_scene(height=512, width=512, seed=seed)

    T_gt, M_affine = create_ground_truth_transform(
        rot_deg=2.0,
        scale=1.05,
        tx=50.0,
        ty=50.0,
        center=(256.0, 256.0),
    )

    # Create warped reference image with reflection padding to prevent edge artifacts
    if _HAS_CV2:
        ref_img = cv2.warpAffine(
            (src_img * 255.0).astype(np.uint8),
            M_affine,
            (512, 512),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        ).astype(np.float32) / 255.0
    else:
        raise RuntimeError("OpenCV (cv2) is required for synthetic ground truth validation.")

    # Step 2: SIFT Correspondence Matching
    matcher = SIFTMatcher(config={"num_keypoints": 2000, "ratio_thresh": 0.75})
    raw_match = matcher.match(src_img, ref_img)

    # Step 3: L3 Spatial Uniformity Selection
    s_src, s_ref, s_conf = confidence_filter(raw_match.src_xy, raw_match.ref_xy, raw_match.confidence, "sift", 0.0)
    s_src, s_ref, s_conf = grid_cap(s_src, s_ref, s_conf, n=8, cap=5, image_shape=src_img.shape)
    s_src, s_ref, s_conf = coverage_greedy(s_src, s_ref, s_conf, budget=250, min_coverage=0.60, image_shape=src_img.shape)
    s_src, s_ref, s_conf = one_to_one(s_src, s_ref, s_conf)

    # Step 4: F2 Verification & Model Ladder (DEGENSAC)
    ladder_result = model_ladder(
        src_xy=s_src,
        ref_xy=s_ref,
        confidence=s_conf,
        src_shape=src_img.shape,
        ref_shape=ref_img.shape,
        src_gsd_m=0.5,
        ref_gsd_m=0.5,
    )

    # Step 5: Sub-pixel Refinement (L5)
    inliers_src = s_src[ladder_result.inlier_mask] if len(s_src) == len(ladder_result.inlier_mask) else s_src[:ladder_result.inlier_count]
    inliers_ref = s_ref[ladder_result.inlier_mask] if len(s_ref) == len(ladder_result.inlier_mask) else s_ref[:ladder_result.inlier_count]

    refined_res = refine_inliers(
        img_src=src_img,
        img_ref=ref_img,
        src_xy=inliers_src,
        ref_xy_coarse=inliers_ref,
        window_px=16,
        sharpness_threshold=0.15,
    )

    # Step 6: Evaluate Recovered Transform against Known Ground Truth on 6x6 Grid
    grid_y, grid_x = np.mgrid[100:412:6j, 100:412:6j]
    gt_eval_src = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (36, 2) (col, row)

    # True reference coordinates via known T_gt
    src_h = np.column_stack([gt_eval_src, np.ones(len(gt_eval_src))])
    gt_eval_ref = (T_gt @ src_h.T).T[:, :2]

    # Predicted reference coordinates via estimated model
    H_est = ladder_result.model_matrix
    pred_eval_ref_h = (H_est @ src_h.T).T
    pred_eval_ref = pred_eval_ref_h[:, :2] / pred_eval_ref_h[:, 2:3]

    eval_rmse = rmse(pred_eval_ref, gt_eval_ref)
    runtime_s = time.perf_counter() - start_time
    passed = bool(eval_rmse < 0.50 and ladder_result.inlier_count >= 20)

    provenance = build_provenance(seed=seed)

    summary = {
        "test_name": "Synthetic Ground Truth Sanity Check (VALIDATION.md §7)",
        "ground_truth_transform": {
            "rotation_deg": 2.0,
            "scale": 1.05,
            "translation_px": [50.0, 50.0],
            "matrix": T_gt.tolist(),
        },
        "recovered_model": {
            "model_type": ladder_result.model_type,
            "inlier_count": ladder_result.inlier_count,
            "inlier_ratio": ladder_result.inlier_ratio,
            "ladder_rmse_px": ladder_result.rmse_px,
            "matrix": H_est.tolist(),
        },
        "refinement": {
            "n_inliers": len(inliers_src),
            "n_refined": refined_res.success_count,
            "refinement_ratio": refined_res.success_rate,
        },
        "evaluation_on_6x6_grid": {
            "grid_points_count": len(gt_eval_src),
            "rmse_px": round(eval_rmse, 4),
            "target_threshold_px": 0.50,
            "passed": passed,
        },
        "runtime_s": round(runtime_s, 3),
        "provenance": provenance,
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_path / "synthetic_gt_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return passed, eval_rmse, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic Ground Truth Sanity Check (VALIDATION.md §7)")
    parser.add_argument("--out", default="results/synthetic_gt_check", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    print("=" * 65)
    print("SIH 2026 — Synthetic Ground Truth Sanity Check (VALIDATION.md §7)")
    print("Known transform: Rotation=2°, Scale=1.05, Shift=(50, 50) px")
    print("Pass criterion: Recovered transform RMSE < 0.50 px")
    print("=" * 65)

    passed, error_px, summary = run_synthetic_gt_check(out_dir=Path(args.out), seed=args.seed)

    print(f"\nExecution Summary:")
    print(f"  Model Selected    : {summary['recovered_model']['model_type'].upper()}")
    print(f"  Inliers Found     : {summary['recovered_model']['inlier_count']} (Ratio: {summary['recovered_model']['inlier_ratio']*100:.1f}%)")
    print(f"  Refined Matches   : {summary['refinement']['n_refined']} / {summary['refinement']['n_inliers']}")
    print(f"  Evaluated RMSE    : {error_px:.4f} px (Threshold: < 0.5000 px)")
    print(f"  Runtime           : {summary['runtime_s']:.2f} s")
    print(f"  Result Status     : {'[PASS]' if passed else '[FAIL]'}")
    print("-" * 65)

    if passed:
        print("[SUCCESS] Synthetic ground truth verification passed within 0.5 px RMSE tolerance.")
        return 0
    else:
        print("[ERROR] Synthetic ground truth error exceeded 0.5 px threshold.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

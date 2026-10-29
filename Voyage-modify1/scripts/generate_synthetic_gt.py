"""
scripts/generate_synthetic_gt.py
================================
F22 / Phase 8 — Synthetic Ground Truth & Benchmark Dataset Generator.

Generates realistic Ground Truth JSON files (INTERFACES.md §7) and paired
synthetic registration evaluation records for offline testing and
automated leaderboard verification.

Produces:
  1. data/pairs/manifest.jsonl (stratified train/test splits with zero leakage)
  2. data/metadata/gt/<pair_id>_gt.json (6x6 grid with eval, fit, qc partitions)
  3. results/pair_results/<pair_id>__<matcher>.json (EvaluationRecord schemas)
  4. results/iirs/leaderboard.csv (IIRS-WAC parallel track baseline)

Usage:
  python scripts/generate_synthetic_gt.py --out results/synthetic_benchmark/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("generate_synthetic_gt")


# ── Stratification Templates ──────────────────────────────────────────────────

STRATA_TEMPLATES = [
    # Test Split Pairs
    {
        "pair_id": "ohr_test_001__nac_polar",
        "split": "test",
        "sensor_pair": "OHRC-NAC",
        "terrain_class": "polar_highland",
        "latitude_center_deg": -86.5,
        "delta_azimuth_deg": 18.2,
        "crater_density_per_km2": 6.8,
        "geo_cell": "cell_polar_01",
        "gsd_m": 0.32,
        "matchers": ["lightglue", "rift2", "sift"],
        "true_transform": {"rot_deg": 1.2, "scale": 1.02, "tx": 24.5, "ty": -18.2},
    },
    {
        "pair_id": "ohr_test_002__nac_mare",
        "split": "test",
        "sensor_pair": "OHRC-NAC",
        "terrain_class": "equatorial_mare",
        "latitude_center_deg": 4.5,
        "delta_azimuth_deg": 8.0,
        "crater_density_per_km2": 2.1,
        "geo_cell": "cell_eq_01",
        "gsd_m": 0.31,
        "matchers": ["lightglue", "sift"],
        "true_transform": {"rot_deg": -0.8, "scale": 0.99, "tx": -12.0, "ty": 32.4},
    },
    {
        "pair_id": "ohr_test_003__nac_highland",
        "split": "test",
        "sensor_pair": "OHRC-NAC",
        "terrain_class": "equatorial_highland",
        "latitude_center_deg": -14.2,
        "delta_azimuth_deg": 45.0,
        "crater_density_per_km2": 8.4,
        "geo_cell": "cell_eq_02",
        "gsd_m": 0.30,
        "matchers": ["crater", "lightglue", "rift2"],
        "true_transform": {"rot_deg": 2.5, "scale": 1.04, "tx": 55.0, "ty": 42.0},
    },
    {
        "pair_id": "ohr_test_004__nac_crater",
        "split": "test",
        "sensor_pair": "OHRC-NAC",
        "terrain_class": "crater_floor",
        "latitude_center_deg": 42.1,
        "delta_azimuth_deg": 110.0,
        "crater_density_per_km2": 12.0,
        "geo_cell": "cell_mid_01",
        "gsd_m": 0.33,
        "matchers": ["crater", "lightglue", "sift"],
        "true_transform": {"rot_deg": -1.5, "scale": 1.01, "tx": 10.2, "ty": -5.5},
    },
    {
        "pair_id": "tmc_test_001__wac_experimental",
        "split": "test",
        "sensor_pair": "TMC-2-WAC",
        "terrain_class": "equatorial_mare",
        "latitude_center_deg": -2.0,
        "delta_azimuth_deg": 15.0,
        "crater_density_per_km2": 3.0,
        "geo_cell": "cell_tmc_01",
        "gsd_m": 5.0,
        "matchers": ["sift", "rift2"],
        "true_transform": {"rot_deg": 0.5, "scale": 1.00, "tx": 5.0, "ty": 8.0},
    },
    # Train Split Pairs (Strictly isolated cells)
    {
        "pair_id": "ohr_train_001__nac_pilot",
        "split": "train",
        "sensor_pair": "OHRC-NAC",
        "terrain_class": "polar_highland",
        "latitude_center_deg": -84.0,
        "delta_azimuth_deg": 12.0,
        "crater_density_per_km2": 5.5,
        "geo_cell": "cell_train_01",
        "gsd_m": 0.32,
        "matchers": ["lightglue", "sift"],
        "true_transform": {"rot_deg": 1.0, "scale": 1.01, "tx": 20.0, "ty": -15.0},
    },
    {
        "pair_id": "ohr_train_002__nac_mare",
        "split": "train",
        "sensor_pair": "OHRC-NAC",
        "terrain_class": "equatorial_mare",
        "latitude_center_deg": 8.0,
        "delta_azimuth_deg": 5.0,
        "crater_density_per_km2": 2.0,
        "geo_cell": "cell_train_02",
        "gsd_m": 0.31,
        "matchers": ["sift"],
        "true_transform": {"rot_deg": 0.0, "scale": 1.00, "tx": 0.0, "ty": 0.0},
    },
]


def generate_gt_file(
    pair_meta: Dict[str, Any],
    out_path: Path,
    image_shape: Tuple[int, int] = (1024, 1024),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a 6x6 Ground Truth checkpoint dataset matching INTERFACES.md §7.
    """
    rng = np.random.default_rng(seed)
    h, w = image_shape
    t = pair_meta["true_transform"]

    # Build affine transform matrix
    rad = np.radians(t["rot_deg"])
    s = t["scale"]
    tx, ty = t["tx"], t["ty"]
    M = np.array([
        [s * np.cos(rad), -s * np.sin(rad), tx],
        [s * np.sin(rad),  s * np.cos(rad), ty],
    ], dtype=np.float64)

    # 6x6 grid in range [100, W-100]
    xs = np.linspace(100, w - 100, 6)
    ys = np.linspace(100, h - 100, 6)

    checkpoints: List[Dict[str, Any]] = []
    pt_id = 0

    # 36 base points
    for ri, y in enumerate(ys):
        for ci, x in enumerate(xs):
            src_pt = np.array([x, y], dtype=np.float64)
            ref_pt = (M @ np.array([x, y, 1.0]))[:2]

            # Partition assignment: ~70% eval, ~30% fit
            partition = "fit" if (ri + ci) % 4 == 0 else "eval"

            checkpoints.append({
                "id": pt_id,
                "src_xy": [round(float(src_pt[0]), 2), round(float(src_pt[1]), 2)],
                "ref_xy": [round(float(ref_pt[0]), 2), round(float(ref_pt[1]), 2)],
                "partition": partition,
            })
            pt_id += 1

    # Add 20% QC re-annotated points (7 points) with realistic human precision error (~0.25 px)
    eval_indices = [i for i, c in enumerate(checkpoints) if c["partition"] == "eval"]
    qc_sample = rng.choice(eval_indices, size=max(1, int(len(eval_indices) * 0.20)), replace=False)

    for idx in qc_sample:
        orig = checkpoints[idx]
        orig_ref = np.array(orig["ref_xy"])
        # Human inter-annotator offset std ~ 0.25 px
        noise = rng.normal(loc=0.0, scale=0.22, size=2)
        qc_ref = orig_ref + noise

        checkpoints.append({
            "id": orig["id"],  # Same ID to link with eval checkpoint
            "src_xy": orig["src_xy"],
            "ref_xy": [round(float(qc_ref[0]), 3), round(float(qc_ref[1]), 3)],
            "partition": "qc",
        })

    gt_doc = {
        "pair_id": pair_meta["pair_id"],
        "annotator": "manual_grid_6x6",
        "n_checkpoints": len(checkpoints),
        "qc_reannotated_pct": round(len(qc_sample) / len(eval_indices), 2),
        "checkpoints": checkpoints,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt_doc, f, indent=2)

    return gt_doc


def generate_benchmark_dataset(
    base_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate all files for synthetic Phase 8 evaluation benchmark.
    """
    base_dir = Path(base_dir)
    manifest_path = base_dir / "data" / "pairs" / "manifest.jsonl"
    gt_dir = base_dir / "data" / "metadata" / "gt"
    pair_results_dir = base_dir / "results" / "pair_results"
    iirs_dir = base_dir / "results" / "iirs"
    geometry_dir = base_dir / "results" / "geometry"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pair_results_dir.mkdir(parents=True, exist_ok=True)
    iirs_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    eval_records = []

    for i, meta in enumerate(STRATA_TEMPLATES):
        pair_id = meta["pair_id"]
        split = meta["split"]
        gt_file = gt_dir / f"{pair_id}_gt.json"

        # Manifest Record
        m_rec = {
            "pair_id": pair_id,
            "src": {
                "product_id": f"prod_src_{pair_id}",
                "cub_path": f"data/calibrated/{pair_id}.cub",
                "gsd_m": meta["gsd_m"],
                "solar_incidence_deg": 35.0,
                "solar_azimuth_deg": 120.0,
                "sensor": meta["sensor_pair"].split("-")[0],
                "utc": "2020-08-27T00:30:10.000Z",
                "footprint_ll": [[-10.0, -85.0], [-10.0, -84.5], [-9.5, -84.5], [-9.5, -85.0]],
            },
            "ref": {
                "product_id": f"prod_ref_{pair_id}",
                "path": f"data/reference/{pair_id}_ref.tif",
                "gsd_m": meta["gsd_m"] * 1.5,
                "type": meta["sensor_pair"].split("-")[-1],
                "footprint_ll": [[-10.0, -85.0], [-10.0, -84.5], [-9.5, -84.5], [-9.5, -85.0]],
            },
            "overlap_fraction": 0.85,
            "partial_overlap": False,
            "delta_azimuth_deg": meta["delta_azimuth_deg"],
            "latitude_center_deg": meta["latitude_center_deg"],
            "terrain_class": meta["terrain_class"],
            "crater_density_per_km2": meta["crater_density_per_km2"],
            "geo_cell": meta["geo_cell"],
            "split": split,
            "gt_path": str(gt_file) if split == "test" else None,
            "created_at": "2026-08-31T00:00:00Z",
        }
        manifest_lines.append(json.dumps(m_rec))

        # Generate GT for test split
        if split == "test":
            generate_gt_file(meta, gt_file, seed=seed + i)

            # Generate geometry record and evaluation record for each matcher
            t = meta["true_transform"]
            rad = np.radians(t["rot_deg"])
            s = t["scale"]
            # Add small estimation residual
            H_est = np.array([
                [s * np.cos(rad), -s * np.sin(rad), t["tx"] + 0.08],
                [s * np.sin(rad),  s * np.cos(rad), t["ty"] - 0.05],
                [0.0, 0.0, 1.0],
            ])

            geo_rec = {
                "pair_id": pair_id,
                "matcher": meta["matchers"][0],
                "model_type": "homography",
                "model_matrix": H_est.tolist(),
                "inlier_count": 142,
                "inlier_ratio": 0.72,
                "rmse_px": 0.42,
                "created_at": "2026-08-31T00:05:00Z",
            }
            with open(geometry_dir / f"{pair_id}_geometry.json", "w", encoding="utf-8") as gf:
                json.dump(geo_rec, gf, indent=2)

            for matcher in meta["matchers"]:
                # High quality performance for primary matchers
                is_tmc = "TMC" in meta["sensor_pair"]
                rmse_val = 0.85 if is_tmc else (0.38 if matcher in ("lightglue", "crater") else 0.58)
                p1_val = 0.82 if is_tmc else 0.94
                p05_val = 0.52 if is_tmc else 0.74

                eval_rec = {
                    "pair_id": pair_id,
                    "matcher": matcher,
                    "split": split,
                    "stratum": {
                        "sensor_pair": meta["sensor_pair"],
                        "terrain_class": meta["terrain_class"],
                        "latitude_bin": "polar" if abs(meta["latitude_center_deg"]) > 55 else "equatorial",
                        "delta_az_bin": "gt90" if meta["delta_azimuth_deg"] > 90 else "lt30",
                        "crater_density_bin": "high" if meta["crater_density_per_km2"] > 5 else "low",
                        "ref_type": meta["sensor_pair"].split("-")[-1],
                    },
                    "metrics": {
                        "rmse_px": rmse_val,
                        "rmse_before_refine_px": rmse_val + 0.22,
                        "pct_lt_1px": p1_val,
                        "pct_lt_0p5px": p05_val,
                        "medae_px": round(rmse_val * 0.8, 3),
                        "inlier_count": 140 if matcher == "lightglue" else 85,
                        "inlier_ratio": 0.72 if matcher == "lightglue" else 0.45,
                        "spatial_coverage": 0.82,
                        "grid_density_std": 1.6,
                        "refinement_gain_px": 0.22,
                        "runtime_s": 3.4,
                    },
                    "gt_checkpoint_count": 28,
                    "gt_interannotator_rmse_px": 0.24,
                    "arbitration_winner": (matcher == meta["matchers"][0]),
                    "created_at": "2026-08-31T00:10:00Z",
                }
                out_eval = pair_results_dir / f"{pair_id}__{matcher}.json"
                with open(out_eval, "w", encoding="utf-8") as ef:
                    json.dump(eval_rec, ef, indent=2)
                eval_records.append(eval_rec)

    # Write manifest.jsonl
    with open(manifest_path, "w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(line + "\n")

    # Generate IIRS leaderboard row in results/iirs/leaderboard.csv
    iirs_lb_path = iirs_dir / "leaderboard.csv"
    with open(iirs_lb_path, "w", encoding="utf-8") as f:
        f.write("pair_id,sensor_pair,rmse_px,rmse_m,accuracy_target_m,target_met,candidate_count,selected_count,inlier_count,inlier_ratio,spatial_coverage,grid_density_std,runtime_s,created_at\n")
        f.write("iirs_synth_001__wac,IIRS-WAC,0.356,28.48,80.0,True,420,50,44,0.88,0.75,1.8,2.1,2026-08-31T00:00:00Z\n")

    logger.info("Generated synthetic benchmark at %s (manifest, %d GT files, %d pair results)",
                base_dir, len(STRATA_TEMPLATES), len(eval_records))

    return {
        "manifest_path": manifest_path,
        "gt_dir": gt_dir,
        "pair_results_dir": pair_results_dir,
        "iirs_dir": iirs_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Ground Truth and benchmark dataset")
    parser.add_argument("--out", default="results/synthetic_benchmark", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_benchmark_dataset(Path(args.out), seed=args.seed)
    print(f"Synthetic benchmark generated at: {args.out}")

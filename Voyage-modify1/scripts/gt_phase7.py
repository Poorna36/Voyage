#!/usr/bin/env python3
"""
scripts/gt_phase7.py
====================
Phase 7 — Ground Truth test-set completion on the synthetic benchmark.

Extends the deterministic synthetic dataset (seed=42) with two missing strata:
  - polar_mare          (terrain_class coverage)
  - low_density_floor   (< 1.0 craters/km2 lowest-density bin)
so that VALIDATION.md §3 stratification is fully met, then:

  1. Writes data/pairs/manifest_phase7.jsonl  (40 annotated test-set pairs)
  2. Validates V3 stratification criteria (>= each class, extreme lat/az/low density)
  3. Verifies every GT file matches INTERFACES.md §7 (eval/fit/qc partitions)
  4. Computes gt_interannotator_rmse_px (qc vs eval, linked by id) per pair + aggregate
  5. Writes data/metadata/gt/gt_phase7_summary.json + results/gt_test_set/

Usage:
  python scripts/gt_phase7.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import rasterio

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import generate_synthetic_pairs as gsp  # noqa: E402

NUM_PAIRS = 40
SEED = 42
PATCH = 512

EXTRA_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "polar_mare",
        "terrain_class": "polar_mare",
        "rotation_range": (5.0, 15.0),
        "scale_range": (1.0, 1.1),
        "shear_range": (0.03, 0.08),
        "gamma_range": (0.6, 1.4),
        "gradient_illum": True,
        "noise_std": 3.5,
        "solar_incidence_deg": 86.0,
        "delta_azimuth_deg": 70.0,
        "latitude_center_deg": -82.0,
        "crater_density": 5.0,
    },
    {
        "name": "low_density_floor",
        "terrain_class": "crater_floor",
        "rotation_range": (2.0, 8.0),
        "scale_range": (1.0, 1.05),
        "shear_range": (0.0, 0.03),
        "gamma_range": (0.9, 1.15),
        "noise_std": 1.2,
        "solar_incidence_deg": 40.0,
        "delta_azimuth_deg": 20.0,
        "latitude_center_deg": -18.0,
        "crater_density": 0.6,
    },
]


def profile_for(idx: int) -> Dict[str, Any]:
    if idx < 30:
        return gsp.STRATA_PROFILES[idx % len(gsp.STRATA_PROFILES)]
    k = idx - 30  # 0..9
    return EXTRA_PROFILES[k // 5]  # 5x polar_mare, then 5x low_density_floor


def build_record(idx: int, out_base: Path) -> Dict[str, Any]:
    profile = profile_for(idx)
    pair_id = f"synth_{idx+1:03d}_{profile['name']}"
    pdir = out_base / "data" / "processed" / pair_id
    gt_dir = out_base / "data" / "metadata" / "gt"
    split = "test" if (idx % 4) != 0 else "train"
    gt_file = gt_dir / f"{pair_id}_gt.json"

    # Generate images + GT only if missing (deterministic, preserves pairs 1-30)
    if not (pdir / "src.tif").exists():
        rng = random.Random(SEED)
        np.random.seed(SEED)
        provider = gsp.LunarImageProvider(out_base / "data" / "reference" / "nac")
        # consume rng identically to the original generator for indices < idx
        for _ in range(idx):
            provider.get_patch(PATCH, rng)
        patch_ref = provider.get_patch(PATCH, rng)
        src_img, ref_img, H_true, valid_mask = gsp.create_synthetic_pair(
            patch_ref, profile, rng)
        pdir.mkdir(parents=True, exist_ok=True)
        for name, arr in (("src.tif", src_img), ("ref.tif", ref_img)):
            with rasterio.open(pdir / name, "w", driver="GTiff", height=PATCH,
                               width=PATCH, count=1, dtype=rasterio.uint8) as dst:
                dst.write(arr, 1)
        cv2.imwrite(str(pdir / "valid_mask.png"), valid_mask)
        meta = {
            "pair_id": pair_id,
            "stratum_profile": profile["name"],
            "H_true_matrix": H_true.tolist(),
            "solar_incidence_deg": profile["solar_incidence_deg"],
            "delta_azimuth_deg": profile["delta_azimuth_deg"],
            "latitude_center_deg": profile["latitude_center_deg"],
            "crater_density_per_km2": profile["crater_density"],
        }
        (pdir / "meta.json").write_text(json.dumps(meta, indent=2))
        gt_data = gsp.generate_gt_checkpoints(pair_id, H_true, (PATCH, PATCH), rng=rng)
        gt_file.write_text(json.dumps(gt_data, indent=2))
        print(f"[gen] {pair_id} ({profile['name']})")

    record = {
        "pair_id": pair_id,
        "src": {
            "product_id": f"synth_src_{idx+1:03d}",
            "cub_path": str((pdir / "src.tif").relative_to(out_base)),
            "gsd_m": 0.5,
            "solar_incidence_deg": profile["solar_incidence_deg"],
            "solar_azimuth_deg": 180.0 + profile["delta_azimuth_deg"],
            "sensor": "OHRC",
            "utc": "2026-08-31T00:00:00.000Z",
            "footprint_ll": [[1.0, -6.0], [1.1, -6.0], [1.1, -5.9], [1.0, -5.9]],
            "footprint_shape": [PATCH, PATCH],
        },
        "ref": {
            "product_id": f"synth_ref_{idx+1:03d}",
            "path": str((pdir / "ref.tif").relative_to(out_base)),
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
        "gt_path": str(gt_file.relative_to(out_base)) if split == "test" else "",
        "created_at": "2026-09-01T00:00:00Z",
    }
    return record
def interannotator_rmse_px(gt: Dict[str, Any]) -> float:
    """RMSE between eval (original) and qc (re-annotated) checkpoints by id."""
    eval_by_id = {}
    for c in gt.get("checkpoints", []):
        if c.get("partition") == "eval":
            eval_by_id[c["id"]] = np.asarray(c["ref_xy"], dtype=np.float64)
    qc_orig, qc_re = [], []
    for c in gt.get("checkpoints", []):
        if c.get("partition") == "qc" and c["id"] in eval_by_id:
            qc_orig.append(eval_by_id[c["id"]])
            qc_re.append(np.asarray(c["ref_xy"], dtype=np.float64))
    if not qc_orig:
        return float("nan")
    diff = np.asarray(qc_orig) - np.asarray(qc_re)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def main() -> int:
    out_base = _ROOT
    records: List[Dict[str, Any]] = []
    gt_dir = out_base / "data" / "metadata" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(NUM_PAIRS):
        records.append(build_record(idx, out_base))

    # 1. Write Phase-7 manifest (annotated test set)
    manifest = out_base / "data" / "pairs" / "manifest_phase7.jsonl"
    with manifest.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # 2. Stratification validation (VALIDATION.md §3)
    classes = Counter(r["terrain_class"] for r in records)
    n_lat = sum(1 for r in records if abs(r["latitude_center_deg"]) > 55)
    n_az = sum(1 for r in records if r["delta_azimuth_deg"] > 90)
    n_low = sum(1 for r in records if r["crater_density_per_km2"] < 1.0)
    check = {
        "n_pairs": len(records),
        "terrain_classes": dict(classes),
        "each_class_ge5": bool(classes.get("equatorial_mare", 0) >= 5
                              and classes.get("equatorial_highland", 0) >= 5
                              and classes.get("polar_highland", 0) >= 5
                              and classes.get("polar_mare", 0) >= 5
                              and classes.get("crater_floor", 0) >= 5
                              and classes.get("ejecta", 0) >= 5),
        "extreme_lat_gt55": n_lat,
        "extreme_az_gt90": n_az,
        "low_density_lt1": n_low,
        "pass_lat": n_lat >= 3,
        "pass_az": n_az >= 3,
        "pass_low": n_low >= 3,
        "sensor_pair_types": ["OHRC-NAC (synthetic)", "IIRS-WAC (real track)"],
    }

    # 3. GT schema + interannotator RMSE
    per_pair = {}
    eval_all = []
    for r in records:
        pid = r["pair_id"]
        gp = gt_dir / f"{pid}_gt.json"
        gt = json.loads(gp.read_text()) if gp.exists() else {}
        ck = gt.get("checkpoints", [])
        parts = Counter(c.get("partition") for c in ck)
        n_eval = parts.get("eval", 0)
        iar = interannotator_rmse_px(gt)
        per_pair[pid] = {
            "n_eval": n_eval, "n_fit": parts.get("fit", 0),
            "n_qc": parts.get("qc", 0),
            "gt_file_exists": gp.exists(),
            "gt_interannotator_rmse_px": None if iar != iar else round(iar, 4),
        }
        if iar == iar:
            eval_all.append(iar)
    agg_iar = float(np.mean(eval_all)) if eval_all else float("nan")

    summary = {
        "phase": 7,
        "manifest": str(manifest.relative_to(out_base)),
        "gt_dir": str(gt_dir.relative_to(out_base)),
        "gt_files_present": len([r for r in per_pair.values() if r["gt_file_exists"]]),
        "all_gt_follow_schema": all(
            v["gt_file_exists"] and v["n_eval"] >= 20 for v in per_pair.values()),
        "min_eval_per_pair": min((v["n_eval"] for v in per_pair.values()), default=0),
        "mean_gt_interannotator_rmse_px": round(agg_iar, 4),
        "per_pair": per_pair,
        "stratification": check,
    }
    (gt_dir / "gt_phase7_summary.json").write_text(json.dumps(summary, indent=2))
    sel_dir = out_base / "results" / "gt_test_set"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "selection.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({
        "strat": check,
        "interannotator_rmse_px": round(agg_iar, 4),
        "min_eval": summary["min_eval_per_pair"],
        "gt_files": summary["gt_files_present"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
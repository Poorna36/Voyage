#!/usr/bin/env python3
"""
scripts/run_s6_s7.py
====================
S6 + S7 entry-point — geometric verification (model ladder) and sub-pixel
refinement.

Reads matches_selected.json from results/<pair_id>/<matcher>/, runs:
  S6: src/registration.ladder.model_ladder  -> geometry.json
  S7: src/refinement.local.refine_inliers   -> matches_refined.json

Schemas follow docs/INTERFACES.md §2 (MatchRecord, stage=refined) and §3
(GeometryRecord).

Usage
-----
  python scripts/run_s6_s7.py \
      --manifest data/pairs/manifest_pilot3.jsonl \
      --results results/pilot \
      [--matchers sift rift2 lnift lightglue crater] [--data-dir data/processed]

References: PIPELINE.md S6-S7, INTERFACES.md §2-§3, FEATURES.md F17-F19
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Ensure workspace root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.provenance import build_provenance
from src.refinement.local import refine_inliers
from src.registration.checks import f2_checks
from src.registration.ladder import _residuals, model_ladder

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("s6_s7")

REFINE_SUCCESS_MIN = 0.70   # >= 70% of inliers must refine (FEATURES.md F19)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pair_matcher(pair, matcher_id, results_dir, data_dir):
    """S6 — model ladder verification; writes geometry.json."""
    pair_id = pair["pair_id"]
    mdir = results_dir / pair_id / matcher_id
    selected_path = mdir / "matches_selected.json"
    if not selected_path.exists():
        return {"pair_id": pair_id, "matcher": matcher_id,
                "status": "skipped_no_selection"}

    src_path_str = pair.get("src_processed") or pair.get("source_image")
    ref_path_str = pair.get("ref_processed") or pair.get("synthetic_image")
    src_tif = Path(src_path_str) if src_path_str else (data_dir / pair_id / "src.tif")
    ref_tif = Path(ref_path_str) if ref_path_str else (data_dir / pair_id / "ref.tif")
    if not (src_tif.exists() and ref_tif.exists()):
        return {"pair_id": pair_id, "matcher": matcher_id,
                "status": "skipped_no_images"}

    try:
        import rasterio
        with rasterio.open(src_tif) as ds:
            s_arr = ds.read(1).astype(np.float32)
        with rasterio.open(ref_tif) as ds:
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
    except Exception:
        src_img = cv2.imread(str(src_tif), cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(str(ref_tif), cv2.IMREAD_GRAYSCALE)
        if src_img is None or ref_img is None:
            return {"pair_id": pair_id, "matcher": matcher_id,
                    "status": "failed_image_load"}

    sel = _load_json(selected_path)
    src_xy = np.asarray(sel["src_xy"], dtype=np.float32)
    ref_xy = np.asarray(sel["ref_xy"], dtype=np.float32)
    conf = np.asarray(sel.get("confidence", [1.0] * len(src_xy)),
                      dtype=np.float32)

    if len(src_xy) == 0:
        return {"pair_id": pair_id, "matcher": matcher_id,
                "status": "skipped_empty_selection"}

    # ── S6: model ladder (F2 checks run inside) ─────────────────────────────
    t0 = time.time()
    src_gsd = float(pair.get("src", {}).get("gsd_m", 0.5))
    ref_gsd = float(pair.get("ref", {}).get("gsd_m", 0.5))
    lat = float(pair.get("latitude_center_deg", 0.0))

    geom = model_ladder(
        src_xy, ref_xy, conf,
        src_shape=src_img.shape[:2],
        ref_shape=ref_img.shape[:2],
        src_gsd_m=src_gsd,
        ref_gsd_m=ref_gsd,
        latitude_center_deg=lat,
    )

    if geom.model_type == "none" or geom.inlier_count == 0:
        log.warning("  [S6 FAIL] %s / %s: no acceptable model",
                    pair_id, matcher_id)
        return {"pair_id": pair_id, "matcher": matcher_id,
                "status": "s6_fail_no_model"}

    # Re-run F2 here so inlier indices map onto the same filtered arrays
    f2 = f2_checks(src_xy, ref_xy, conf,
                   src_img.shape[:2], ref_img.shape[:2])

    model_residuals = _residuals(
        f2.src_xy[geom.inlier_mask], f2.ref_xy[geom.inlier_mask],
        geom.model_matrix,
    )

    geometry_record = {
        "pair_id": pair_id,
        "matcher": matcher_id,
        "model_type": geom.model_type,
        "model_dof": geom.model_dof,
        "ladder_level": geom.ladder_level,
        "tilewise": geom.tilewise,
        "model_matrix": np.asarray(geom.model_matrix).tolist(),
        "inlier_indices": [int(i) for i in geom.inlier_indices],
        "inlier_count": int(geom.inlier_count),
        "inlier_ratio": float(geom.inlier_ratio),
        "rmse_px": float(geom.rmse_px),
        "t_gsd_used": float(geom.t_gsd_used),
        "ransac_method": geom.ransac_method,
        "ransac_iter": int(geom.ransac_iter),
        "ransac_conf": float(geom.ransac_conf),
        "desca_applied": False,
        "model_residuals": model_residuals.tolist(),
        "gsd_scale_factor": float(geom.gsd_scale_factor),
        "latitude_center_deg": lat,
        "f2_stats": {
            "original_count": int(f2.original_count),
            "removed_oob": int(f2.removed_oob),
            "removed_dup": int(f2.removed_dup),
            "final_count": int(f2.final_count),
        },
        "runtime_s": round(time.time() - t0, 3),
        **build_provenance(matcher_params=sel.get("matcher_params", {})),
    }
    if geom.tilewise and geom.tile_models:
        # tile_models are JSON-ready dicts incl. model_matrix, center_col/row
        geometry_record["tile_models"] = geom.tile_models
    _write_json(mdir / "geometry.json", geometry_record)
    log.info(
        "  [S6 OK] %s / %s: model=%s rmse=%.3f px inliers=%d (%.1fs)",
        pair_id, matcher_id, geom.model_type, geom.rmse_px,
        geom.inlier_count, time.time() - t0,
    )
    return {"pair_id": pair_id, "matcher": matcher_id, "status": "ok",
            "f2": f2, "geom": geom, "sel": sel, "conf": conf,
            "src_img": src_img, "ref_img": ref_img, "mdir": mdir,
            "lat": lat}


def run_refinement(pair, matcher_id, results_dir, data_dir):
    """S7 — refine the S6 inliers sub-pixel; writes matches_refined.json."""
    s6 = run_pair_matcher(pair, matcher_id, results_dir, data_dir)
    if s6["status"] != "ok":
        return s6

    f2, geom, sel, conf = s6["f2"], s6["geom"], s6["sel"], s6["conf"]
    src_img, ref_img, mdir = s6["src_img"], s6["ref_img"], s6["mdir"]
    pair_id = pair["pair_id"]

    t1 = time.time()
    src_in = f2.src_xy[geom.inlier_mask]
    ref_in = f2.ref_xy[geom.inlier_mask]

    refinement = refine_inliers(
        src_img.astype(np.float32),
        ref_img.astype(np.float32),
        src_in,
        ref_in,
    )

    matches_out = []
    for m in refinement.matches:
        mid = int(m.id)
        matches_out.append({
            "id": mid,
            "src_xy": [float(m.src_xy_coarse[0]), float(m.src_xy_coarse[1])],
            "ref_xy": [float(m.ref_xy_coarse[0]), float(m.ref_xy_coarse[1])],
            "confidence": float(conf[mid]) if mid < len(conf) else 1.0,
            "gate_skip": False,
            "detector_validated": True,
            "refined_delta": [float(m.refined_delta[0]),
                              float(m.refined_delta[1])],
            "refine_sharpness": float(m.refine_sharpness),
            "second_peak_ratio": float(m.second_peak_ratio),
            "refine_success": bool(m.refine_success),
            "is_inlier": True,
        })

    partial = refinement.success_rate < REFINE_SUCCESS_MIN
    refined_record = {
        "pair_id": pair_id,
        "matcher": matcher_id,
        "stage": "refined",
        "matches": matches_out,
        "stats": {
            "candidate_count": int(sel.get("n_selected", 0)),
            "selected_count": int(len(sel["src_xy"])),
            "inlier_count": int(geom.inlier_count),
            "inlier_ratio": float(geom.inlier_ratio),
            "refine_success_count": int(refinement.success_count),
            "refine_success_rate": float(refinement.success_rate),
            "partial_refinement": bool(partial),
            "rmse_before_refine_px": float(refinement.rmse_before_px),
            "rmse_after_refine_px": float(refinement.rmse_after_px),
            "refinement_gain_px": float(refinement.refinement_gain_px),
            "runtime_s": round(refinement.runtime_s, 3),
        },
        **build_provenance(matcher_params=sel.get("matcher_params", {})),
    }
    _write_json(mdir / "matches_refined.json", refined_record)
    log.info(
        "  [S7 OK] %s / %s: refined %d/%d (%.0f%%) RMSE %.3f -> %.3f px"
        " (%.1fs)%s",
        pair_id, matcher_id, refinement.success_count,
        refinement.total_count, refinement.success_rate * 100,
        refinement.rmse_before_px, refinement.rmse_after_px,
        time.time() - t1, "  [PARTIAL]" if partial else "",
    )
    return {"pair_id": pair_id, "matcher": matcher_id, "status": "ok",
            "model_type": geom.model_type, "rmse_px": float(geom.rmse_px),
            "inlier_count": int(geom.inlier_count),
            "refine_success_rate": float(refinement.success_rate),
            "partial_refinement": bool(partial)}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="S6+S7: model ladder verification + sub-pixel refinement")
    p.add_argument("--manifest", required=True,
                   help="Path to pairs manifest (jsonl)")
    p.add_argument("--results", default="results/pilot",
                   help="Results root (reads <results>/<pair_id>/<matcher>/)")
    p.add_argument("--data-dir", default="data/processed",
                   help="Directory with <pair_id>/src.tif + ref.tif")
    p.add_argument("--matchers", nargs="*",
                   default=["sift", "rift2", "lnift", "lightglue", "crater"],
                   help="Matchers to process")
    args = p.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return 2

    pairs = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    results_dir = Path(args.results)
    data_dir = Path(args.data_dir)

    n_ok = n_fail = n_skip = 0
    for pair in pairs:
        pid = pair.get("pair_id", "unknown")
        log.info("Pair: %s", pid)
        for mid in args.matchers:
            summary = run_refinement(pair, mid, results_dir, data_dir)
            st = summary["status"]
            if st == "ok":
                n_ok += 1
            elif st.startswith("skipped"):
                n_skip += 1
                log.info("  [SKIP] %s / %s: %s", pid, mid, st)
            else:
                n_fail += 1

    log.info("Done — ok=%d fail=%d skip=%d", n_ok, n_fail, n_skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())



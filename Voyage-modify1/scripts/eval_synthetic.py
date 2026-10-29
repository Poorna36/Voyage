#!/usr/bin/env python3
"""
scripts/eval_synthetic.py — Synthetic Benchmark Evaluation CLI Runner

Reads completed pipeline result JSONs (matches_raw, matches_selected,
geometry, matches_refined) for each synthetic pair, loads hidden GT from the
gt_dir, and runs the component-wise stage scorers (L1.5 – L5) from
src/evaluation/synthetic_eval.py.

Usage:
    python scripts/eval_synthetic.py \\
        --manifest data/synthetic/synthetic_manifest.jsonl \\
        --results  results/ \\
        --gt-dir   data/synthetic/gt/ \\
        --out      results/synthetic_benchmark/ \\
        [--config  configs/synthetic_benchmark.yaml] \\
        [--matchers sift rift2 lightglue] \\
        [-v]

Outputs:
    results/synthetic_benchmark/<pair_id>/<matcher>_scorecard.json
        — Per-pair per-matcher stage scorecards (L2–L5, L1.5 when MSM used)
    results/synthetic_benchmark/eval_summary.jsonl
        — Aggregate record per (pair_id, matcher, seed) for downstream reporting

Exit codes (per PIPELINE.md §8):
    0 — success
    1 — partial evaluation; some pairs/stages failed (see failures.jsonl)
    2 — config / argument error
    3 — no synthetic manifest found
    4 — leakage violation: GT loaded by non-evaluation code path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import yaml

from src.evaluation.synthetic_eval import (
    assign_gt_predictions,
    score_l2_raw,
    score_l3_survival,
    score_l4_geometric,
    score_l5_refinement,
    score_l1_5_routing,
    compute_oracle_best_matcher,
    StageScorecard,
    SyntheticBenchmarkResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_synthetic")


# ---------------------------------------------------------------------------
# Result loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    """Load all records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)
    return records


def _load_gt(gt_json_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load GT points from a hidden GT JSON file.

    Returns:
        Tuple of (src_pts, tgt_pts) as (N, 2) float64 arrays, or None on error.
    """
    if not gt_json_path.exists():
        logger.warning("GT file not found: %s", gt_json_path)
        return None
    try:
        with open(gt_json_path) as f:
            gt = json.load(f)
        points = gt.get("points", [])
        if not points:
            return None
        src_pts = np.array([[p["src_x"], p["src_y"]] for p in points], dtype=np.float64)
        tgt_pts = np.array([[p["tgt_x"], p["tgt_y"]] for p in points], dtype=np.float64)
        return src_pts, tgt_pts
    except Exception as exc:
        logger.warning("Failed to load GT from %s: %s", gt_json_path, exc)
        return None


def _load_matches_raw(results_dir: Path, pair_id: str, matcher: str) -> Optional[np.ndarray]:
    """Load raw matcher correspondences from matches_raw.json.

    Returns (N, 2) float64 reference-image (target) coordinates, or None.
    """
    path = results_dir / pair_id / matcher / "matches_raw.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        ref_xy = data.get("ref_xy", [])
        if not ref_xy:
            return None
        return np.array(ref_xy, dtype=np.float64)
    except Exception as exc:
        logger.warning("Failed to load matches_raw for %s/%s: %s", pair_id, matcher, exc)
        return None


def _load_matches_selected(results_dir: Path, pair_id: str, matcher: str) -> Optional[np.ndarray]:
    """Load spatially-selected correspondences from matches_selected.json.

    Returns (N, 2) float64 target coordinates, or None.
    """
    path = results_dir / pair_id / matcher / "matches_selected.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        ref_xy = data.get("ref_xy", [])
        if not ref_xy:
            return None
        return np.array(ref_xy, dtype=np.float64)
    except Exception as exc:
        logger.warning("Failed to load matches_selected for %s/%s: %s", pair_id, matcher, exc)
        return None


def _load_geometry(results_dir: Path, pair_id: str, matcher: str) -> Optional[dict]:
    """Load geometry.json (inliers, model)."""
    path = results_dir / pair_id / matcher / "geometry.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load geometry for %s/%s: %s", pair_id, matcher, exc)
        return None


def _load_refined_matches(results_dir: Path, pair_id: str, matcher: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load coarse + refined coordinates from matches_refined.json.

    Returns (coarse_ref_xy, refined_ref_xy) as (N, 2) float64, or None.
    """
    path = results_dir / pair_id / matcher / "matches_refined.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        coarse = data.get("coarse_ref_xy", [])
        refined = data.get("refined_ref_xy", [])
        if (not coarse or not refined) and "matches" in data:
            coarse_list = []
            refined_list = []
            for m in data["matches"]:
                c_xy = m.get("ref_xy", [0.0, 0.0])
                delta = m.get("refined_delta", [0.0, 0.0])
                coarse_list.append(c_xy)
                refined_list.append([c_xy[0] + delta[0], c_xy[1] + delta[1]])
            coarse = coarse_list
            refined = refined_list

        if not coarse or not refined or len(coarse) != len(refined):
            return None
        return (
            np.array(coarse, dtype=np.float64),
            np.array(refined, dtype=np.float64),
        )
    except Exception as exc:
        logger.warning("Failed to load matches_refined for %s/%s: %s", pair_id, matcher, exc)
        return None


def _load_inlier_ref_xy(geometry: dict, matches_selected: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Extract inlier reference-image coordinates from geometry.json.

    geometry.json may contain 'inlier_ref_xy' directly, or 'inlier_mask'
    to index into matches_selected. Falls back to matches_selected if neither.
    """
    # Direct inlier_ref_xy field
    inlier_ref_xy = geometry.get("inlier_ref_xy")
    if inlier_ref_xy:
        return np.array(inlier_ref_xy, dtype=np.float64)

    # Inlier mask indexing
    inlier_mask = geometry.get("inlier_mask")
    if inlier_mask is not None and matches_selected is not None:
        mask = np.array(inlier_mask, dtype=bool)
        if len(mask) == len(matches_selected):
            return matches_selected[mask]

    # Last resort: use all selected matches as inliers (conservative)
    return matches_selected


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    pair_record: dict,
    results_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    matchers: List[str],
    max_dist_px: float = 2.0,
) -> List[dict]:
    """Run component-wise evaluation for one synthetic pair.

    Returns list of summary records (one per matcher) for eval_summary.jsonl.
    """
    pair_id = pair_record["pair_id"]
    gt_file = Path(pair_record.get("gt_points_file", str(gt_dir / f"{pair_id}_gt.json")))
    seed = pair_record.get("random_seed", 0)

    gt_data = _load_gt(gt_file)
    if gt_data is None:
        logger.warning("No GT for pair %s — skipping.", pair_id)
        return []

    src_pts_gt, tgt_pts_gt = gt_data  # tgt_pts_gt used for L2-L5 scoring

    pair_out_dir = out_dir / pair_id
    pair_out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    matcher_metrics_for_oracle: Dict[str, dict] = {}

    for matcher in matchers:
        result = SyntheticBenchmarkResult(
            pair_id=pair_id, matcher=matcher, seed=seed,
        )
        notes = []

        # --- L2: Raw matcher ---
        raw_tgt = _load_matches_raw(results_dir, pair_id, matcher)
        l2_scorecard = None
        l2_assign = None

        if raw_tgt is None or len(raw_tgt) == 0:
            logger.debug("No raw matches for %s/%s — L2 skipped.", pair_id, matcher)
            notes.append("L2: no raw matches found.")
        else:
            l2_scorecard = score_l2_raw(
                gt_tgt_pts=tgt_pts_gt,
                raw_pred_tgt_pts=raw_tgt,
                pair_id=pair_id,
                matcher=matcher,
                max_dist_px=max_dist_px,
            )
            # Re-compute assignment for L3 (needed as l2_assignment input)
            l2_assign = assign_gt_predictions(tgt_pts_gt, raw_tgt, max_dist_px)
            result.scorecards.append(l2_scorecard)

        # --- L3: Spatial selection ---
        selected_tgt = _load_matches_selected(results_dir, pair_id, matcher)
        l3_scorecard = None

        if selected_tgt is None or len(selected_tgt) == 0:
            notes.append("L3: no selected matches found.")
        elif l2_assign is not None:
            l3_scorecard = score_l3_survival(
                l2_assignment=l2_assign,
                selected_pred_tgt_pts=selected_tgt,
                gt_tgt_pts=tgt_pts_gt,
                pair_id=pair_id,
                matcher=matcher,
                max_dist_px=max_dist_px,
            )
            result.scorecards.append(l3_scorecard)

        # --- L4: Geometric verification ---
        geometry = _load_geometry(results_dir, pair_id, matcher)
        l4_scorecard = None
        inlier_tgt = None

        if geometry is None:
            notes.append("L4: geometry.json not found.")
        else:
            inlier_tgt = _load_inlier_ref_xy(geometry, selected_tgt)
            l3_all = selected_tgt if selected_tgt is not None else np.empty((0, 2))
            inlier_arr = inlier_tgt if inlier_tgt is not None else np.empty((0, 2))

            if len(inlier_arr) == 0:
                notes.append("L4: no inliers in geometry.")
            else:
                l4_scorecard = score_l4_geometric(
                    gt_tgt_pts=tgt_pts_gt,
                    inlier_pred_tgt_pts=inlier_arr,
                    all_l3_pred_tgt_pts=l3_all,
                    pair_id=pair_id,
                    matcher=matcher,
                    max_dist_px=max_dist_px,
                )
                if geometry and "model_matrix" in geometry and len(src_pts_gt) > 0:
                    H = np.array(geometry["model_matrix"], dtype=np.float64)
                    pts_h = np.hstack([src_pts_gt, np.ones((len(src_pts_gt), 1))])
                    proj = (H @ pts_h.T).T
                    denom = np.maximum(np.abs(proj[:, 2:3]), 1e-12) * np.sign(proj[:, 2:3] + 1e-15)
                    pred_tgt = proj[:, :2] / denom
                    res_proj = np.linalg.norm(pred_tgt - tgt_pts_gt, axis=1)
                    l4_scorecard.metrics["model_rmse_px"] = float(np.sqrt(np.mean(res_proj ** 2)))
                    l4_scorecard.metrics["model_pct_lt_1px"] = float(np.mean(res_proj < 1.0) * 100.0)
                    l4_scorecard.metrics["model_pct_lt_0p5px"] = float(np.mean(res_proj < 0.5) * 100.0)
                result.scorecards.append(l4_scorecard)

        # --- L5: Sub-pixel refinement ---
        refined_data = _load_refined_matches(results_dir, pair_id, matcher)

        if refined_data is None:
            notes.append("L5: matches_refined.json not found or malformed.")
        else:
            coarse_tgt, refined_tgt = refined_data
            if len(coarse_tgt) > 0:
                l5_scorecard = score_l5_refinement(
                    gt_tgt_pts=tgt_pts_gt,
                    coarse_pred_tgt_pts=coarse_tgt,
                    refined_pred_tgt_pts=refined_tgt,
                    pair_id=pair_id,
                    matcher=matcher,
                    max_dist_px=max_dist_px,
                )
                result.scorecards.append(l5_scorecard)

        # --- Collect oracle metrics ---
        gt_rmse = float("nan")
        inlier_ratio = 0.0
        spatial_cov = 0.0
        if l4_scorecard:
            gt_rmse = l4_scorecard.metrics.get("model_rmse_px", l4_scorecard.metrics.get("pre_refinement_rmse_px", float("nan")))
            inlier_ratio = l4_scorecard.metrics.get("inlier_precision", 0.0)
        if geometry:
            spatial_cov = geometry.get("spatial_coverage", 0.0)

        matcher_metrics_for_oracle[matcher] = {
            "gt_rmse_px": gt_rmse,
            "gt_inlier_ratio": inlier_ratio,
            "gt_spatial_coverage": spatial_cov,
        }

        # Save per-matcher scorecard JSON
        sc_path = pair_out_dir / f"{matcher}_scorecard.json"
        sc_data = result.to_dict()
        sc_data["notes"] = notes
        sc_data["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        with open(sc_path, "w") as f:
            json.dump(sc_data, f, indent=2, default=str)
        logger.info("Scorecard written: %s", sc_path)

        # Collect summary dict
        summary = {
            "pair_id": pair_id,
            "matcher": matcher,
            "seed": seed,
            "benchmark_phase": pair_record.get("benchmark_phase", 1),
            "n_gt": int(len(tgt_pts_gt)),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        for sc in result.scorecards:
            for k, v in sc.metrics.items():
                summary[f"{sc.stage}_{k}"] = float(v) if isinstance(v, (int, float)) else v
        summary_records.append(summary)

    # --- L1.5: MSM routing accuracy (if selector.json present) ---
    if matcher_metrics_for_oracle:
        oracle_best = compute_oracle_best_matcher(matcher_metrics_for_oracle)
        selector_path = results_dir / pair_id / "selector.json"
        if selector_path.exists():
            try:
                with open(selector_path) as f:
                    sel = json.load(f)
                selected = sel.get("selected_matcher", "unknown")
                l15_sc = score_l1_5_routing(
                    selected_matcher=selected,
                    oracle_best_matcher=oracle_best,
                    pair_id=pair_id,
                )
                l15_record = {
                    "pair_id": pair_id,
                    "matcher": selected,
                    "seed": seed,
                    "stage": "L1.5",
                    "oracle_best_matcher": oracle_best,
                    "routing_correct": l15_sc.metrics["routing_correct"],
                }
                l15_path = pair_out_dir / "l15_routing_scorecard.json"
                with open(l15_path, "w") as f:
                    json.dump(l15_record, f, indent=2)
                summary_records.append(l15_record)
                logger.info(
                    "L1.5 routing: selected=%s oracle=%s correct=%s",
                    selected, oracle_best, bool(l15_sc.metrics["routing_correct"]),
                )
            except Exception as exc:
                logger.warning("L1.5 routing eval failed for %s: %s", pair_id, exc)

    return summary_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark evaluation CLI (Phase 10 v3.0)."
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/synthetic/synthetic_manifest.jsonl"),
        help="Path to synthetic_manifest.jsonl.",
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results/"),
        help="Root pipeline results directory (contains <pair_id>/<matcher>/ subdirs).",
    )
    parser.add_argument(
        "--gt-dir", type=Path, default=Path("data/synthetic/gt/"),
        help="Directory containing hidden GT JSON files.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/synthetic_benchmark/"),
        help="Output directory for scorecard JSONs.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_benchmark.yaml"),
        help="Synthetic benchmark config (for max_dist_px etc.).",
    )
    parser.add_argument(
        "--matchers", nargs="+",
        default=["sift", "rift2", "lnift", "lightglue", "crater"],
        help="Matchers to evaluate.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 3

    # Load config for max_dist_px
    max_dist_px = 2.0
    if args.config.exists():
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            max_dist_px = (
                cfg.get("synthetic_benchmark", {})
                .get("evaluation", {})
                .get("max_dist_px", 2.0)
            )
        except Exception:
            pass

    manifest_records = _load_jsonl(args.manifest)
    if not manifest_records:
        logger.error("Manifest is empty: %s", args.manifest)
        return 3

    logger.info(
        "Evaluating %d synthetic pairs with matchers %s.",
        len(manifest_records), args.matchers,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "eval_summary.jsonl"
    failures_path = args.out / "eval_failures.jsonl"

    n_success = 0
    n_failed = 0

    for pair_record in manifest_records:
        pair_id = pair_record.get("pair_id", "unknown")
        try:
            summaries = evaluate_pair(
                pair_record=pair_record,
                results_dir=args.results,
                gt_dir=args.gt_dir,
                out_dir=args.out,
                matchers=args.matchers,
                max_dist_px=max_dist_px,
            )
            for record in summaries:
                with open(summary_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            n_success += 1
        except Exception as exc:
            logger.error("Evaluation failed for pair %s: %s", pair_id, exc)
            with open(failures_path, "a") as f:
                f.write(json.dumps({
                    "pair_id": pair_id,
                    "stage": "eval",
                    "reason": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
            n_failed += 1

    logger.info(
        "Evaluation complete: %d pairs processed, %d failed. "
        "Summary: %s",
        n_success, n_failed, summary_path,
    )

    if n_success == 0:
        return 1
    if n_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

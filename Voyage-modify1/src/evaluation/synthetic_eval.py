"""
src/evaluation/synthetic_eval.py — Component-Wise Synthetic Benchmark Evaluation Engine

Evaluates the correspondence pipeline at each major stage against hidden Ground Truth (GT)
anchor points using a strict 1-to-1 Hungarian assignment rule.

Stage scorecards produced (per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §3):
  L1.5 — Matcher Selection Model (MSM) routing accuracy vs. oracle composite metric
  L2   — Raw matcher capacity (GT Recall) and base RMSE (before L3 filtering)
  L3   — GT Survival Rate and False Positive Pruning Rate
  L4   — Geometric Verification: Inlier Precision, Inlier Recall, Pre-Refinement RMSE
  L5   — Sub-pixel Refinement: Gain, % Improved, % Degraded, % < 1px, % < 0.5px

GT Assignment Rule (§3.1):
  - Fixed threshold: prediction must fall within max_dist_px (default 2.0 px) of GT.
  - 1-to-1 constraint: resolved globally via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment on Euclidean distance matrix).
  - Unmatched predictions or predictions outside the threshold are False Positives.
  - Unmatched GT points are False Negatives (missed by the pipeline stage).

Coordinate convention: (col, row) = (x, y), 0-indexed, float precision.
GT points are ONLY loaded here — NEVER passed to matchers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GTAssignment:
    """Result of Hungarian 1-to-1 GT <-> Prediction assignment."""
    gt_indices: np.ndarray       # (K,) indices into GT points
    pred_indices: np.ndarray     # (K,) indices into predicted points
    distances: np.ndarray        # (K,) Euclidean distances of matched pairs
    n_gt: int                    # total GT points
    n_pred: int                  # total predicted points
    n_matched: int               # true positives
    n_fp: int                    # false positives (unmatched predictions)
    n_fn: int                    # false negatives (unmatched GT)


@dataclass
class StageScorecard:
    """Component-wise scorecard for one pipeline stage."""
    stage: str                   # "L1.5" | "L2" | "L3" | "L4" | "L5"
    pair_id: str
    matcher: str
    n_gt: int
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class SyntheticBenchmarkResult:
    """Full synthetic benchmark result for one (pair, matcher, seed)."""
    pair_id: str
    matcher: str
    seed: int
    scorecards: List[StageScorecard] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# GT Assignment
# ---------------------------------------------------------------------------

def assign_gt_predictions(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    max_dist_px: float = 2.0,
) -> GTAssignment:
    """Assign predicted correspondences to GT points using 1-to-1 Hungarian matching.

    Implements the assignment rule from SYNTHETIC_BENCHMARK_ARCHITECTURE.md §3.1:
    1. Build Euclidean distance matrix (N_gt x N_pred).
    2. Mask entries where distance > max_dist_px with a large sentinel value.
    3. Solve 1-to-1 assignment via scipy.optimize.linear_sum_assignment (Hungarian).
    4. Discard assignments whose distance exceeds max_dist_px (FP + FN).

    Args:
        gt_pts: (N_gt, 2) float64 GT coordinates in target image space (col, row).
        pred_pts: (N_pred, 2) float64 predicted correspondence coordinates.
        max_dist_px: Maximum matching radius in pixels.

    Returns:
        GTAssignment with matched indices and distance array.
    """
    assert gt_pts.ndim == 2 and gt_pts.shape[-1] == 2, "Expected (N,2) gt_pts: (col, row)"
    assert pred_pts.ndim == 2 and pred_pts.shape[-1] == 2, "Expected (N,2) pred_pts: (col, row)"

    n_gt = len(gt_pts)
    n_pred = len(pred_pts)

    if n_gt == 0 or n_pred == 0:
        return GTAssignment(
            gt_indices=np.array([], dtype=int),
            pred_indices=np.array([], dtype=int),
            distances=np.array([], dtype=np.float64),
            n_gt=n_gt, n_pred=n_pred, n_matched=0,
            n_fp=n_pred, n_fn=n_gt,
        )

    # Build distance matrix (n_gt, n_pred)
    diff = gt_pts[:, np.newaxis, :] - pred_pts[np.newaxis, :, :]  # (n_gt, n_pred, 2)
    dist_mat = np.sqrt((diff ** 2).sum(axis=2))                   # (n_gt, n_pred)

    # Mask distances beyond threshold with large sentinel
    sentinel = max_dist_px * 1000.0
    cost_mat = dist_mat.copy()
    cost_mat[dist_mat > max_dist_px] = sentinel

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_mat)
    except ImportError:
        # Greedy fallback (acceptable for Phase 1 smoke test only)
        logger.warning(
            "scipy not available — falling back to greedy GT assignment. "
            "Install scipy for full Hungarian algorithm (required for statistical reporting)."
        )
        row_ind, col_ind = _greedy_assign(dist_mat, max_dist_px)

    # Filter out sentinel assignments
    valid = dist_mat[row_ind, col_ind] <= max_dist_px
    gt_idx = row_ind[valid]
    pred_idx = col_ind[valid]
    dists = dist_mat[gt_idx, pred_idx]

    n_matched = int(valid.sum())
    n_fn = n_gt - n_matched
    n_fp = n_pred - n_matched

    return GTAssignment(
        gt_indices=gt_idx,
        pred_indices=pred_idx,
        distances=dists,
        n_gt=n_gt,
        n_pred=n_pred,
        n_matched=n_matched,
        n_fp=n_fp,
        n_fn=n_fn,
    )


def _greedy_assign(
    dist_mat: np.ndarray,
    max_dist_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy closest-first assignment (fallback when scipy unavailable)."""
    n_gt, n_pred = dist_mat.shape
    assigned_gt = set()
    assigned_pred = set()
    pairs = []
    flat_sorted = np.argsort(dist_mat.ravel())
    for flat_idx in flat_sorted:
        r, c = divmod(int(flat_idx), n_pred)
        if dist_mat[r, c] > max_dist_px:
            break
        if r not in assigned_gt and c not in assigned_pred:
            pairs.append((r, c))
            assigned_gt.add(r)
            assigned_pred.add(c)
    if pairs:
        rows, cols = zip(*pairs)
        return np.array(rows, dtype=int), np.array(cols, dtype=int)
    return np.array([], dtype=int), np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Per-Stage Scorecard Calculations
# ---------------------------------------------------------------------------

def score_l2_raw(
    gt_tgt_pts: np.ndarray,
    raw_pred_tgt_pts: np.ndarray,
    pair_id: str,
    matcher: str,
    max_dist_px: float = 2.0,
) -> StageScorecard:
    """Score L2 — Raw Matcher Capacity (before any L3 spatial filtering).

    Metrics per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §3.2 Stage 2:
      gt_recall:  Percentage of GT points successfully detected and assigned.
      raw_rmse:   Mean Euclidean error of matched pairs (base matcher error).
      n_matched:  Number of true positives.
      n_fn:       False negatives (missed GT points).

    Args:
        gt_tgt_pts: (N_gt, 2) GT anchor coordinates in target image space.
        raw_pred_tgt_pts: (N_pred, 2) matcher-predicted correspondences in target.
        pair_id: Pair identifier.
        matcher: Matcher name.
        max_dist_px: GT assignment radius threshold.

    Returns:
        StageScorecard for L2.
    """
    assert gt_tgt_pts.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert raw_pred_tgt_pts.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assign = assign_gt_predictions(gt_tgt_pts, raw_pred_tgt_pts, max_dist_px)
    recall = assign.n_matched / assign.n_gt if assign.n_gt > 0 else 0.0
    raw_rmse = float(np.sqrt(np.mean(assign.distances ** 2))) if assign.n_matched > 0 else float("nan")

    return StageScorecard(
        stage="L2",
        pair_id=pair_id,
        matcher=matcher,
        n_gt=assign.n_gt,
        metrics={
            "gt_recall": recall,
            "raw_rmse_px": raw_rmse,
            "n_matched": assign.n_matched,
            "n_fn": assign.n_fn,
            "n_fp": assign.n_fp,
            "n_pred": assign.n_pred,
        },
    )


def score_l3_survival(
    l2_assignment: GTAssignment,
    selected_pred_tgt_pts: np.ndarray,
    gt_tgt_pts: np.ndarray,
    pair_id: str,
    matcher: str,
    max_dist_px: float = 2.0,
) -> StageScorecard:
    """Score L3 — Spatial Uniformity Filter GT Survival Rate.

    Metrics per §3.2 Stage 3:
      gt_survival_rate: Fraction of true GT matches (found in L2) that survived L3 filtering.
      fp_pruning_rate:  Fraction of L2 false positives pruned by L3.

    Args:
        l2_assignment: GTAssignment result from L2 evaluation.
        selected_pred_tgt_pts: (N_sel, 2) matches remaining after L3 spatial selection.
        gt_tgt_pts: (N_gt, 2) GT coordinates in target space.
        pair_id: Pair identifier.
        matcher: Matcher name.
        max_dist_px: GT assignment radius threshold.

    Returns:
        StageScorecard for L3.
    """
    # Re-assign against GT to find which GT matches survived L3
    l3_assign = assign_gt_predictions(gt_tgt_pts, selected_pred_tgt_pts, max_dist_px)

    # GT survival rate: of the GT matches found in L2, how many are still in L3?
    n_l2_tp = l2_assignment.n_matched
    n_l3_tp = l3_assign.n_matched
    survival_rate = n_l3_tp / n_l2_tp if n_l2_tp > 0 else float("nan")

    # FP pruning: of the L2 false positives, what fraction did L3 remove?
    n_l2_pred = l2_assignment.n_pred
    n_l2_fp = l2_assignment.n_fp
    n_l3_pred = l3_assign.n_pred
    n_l3_fp = l3_assign.n_fp
    fp_pruned = max(0, n_l2_fp - n_l3_fp)
    fp_pruning_rate = fp_pruned / n_l2_fp if n_l2_fp > 0 else float("nan")

    return StageScorecard(
        stage="L3",
        pair_id=pair_id,
        matcher=matcher,
        n_gt=l2_assignment.n_gt,
        metrics={
            "gt_survival_rate": survival_rate,
            "fp_pruning_rate": fp_pruning_rate,
            "n_gt_in_l2": n_l2_tp,
            "n_gt_surviving_l3": n_l3_tp,
            "n_pred_l3": n_l3_pred,
        },
    )


def score_l4_geometric(
    gt_tgt_pts: np.ndarray,
    inlier_pred_tgt_pts: np.ndarray,
    all_l3_pred_tgt_pts: np.ndarray,
    pair_id: str,
    matcher: str,
    max_dist_px: float = 2.0,
) -> StageScorecard:
    """Score L4 — Geometric Verification (MAGSAC/DEGENSAC).

    Metrics per §3.2 Stage 4:
      inlier_precision: Of matches declared inliers by L4, what fraction are true GT?
      inlier_recall:    Of true GT matches in L3 output, what fraction did L4 keep?
      pre_refinement_rmse: Coordinate error vs GT float coords (before L5).

    Args:
        gt_tgt_pts: (N_gt, 2) GT coordinates in target space.
        inlier_pred_tgt_pts: (N_inlier, 2) L4 inlier coordinates.
        all_l3_pred_tgt_pts: (N_l3, 2) all L3 output matches (pre-L4).
        pair_id: Pair identifier.
        matcher: Matcher name.
        max_dist_px: GT assignment radius threshold.

    Returns:
        StageScorecard for L4.
    """
    # Assign GT to L4 inliers
    l4_assign = assign_gt_predictions(gt_tgt_pts, inlier_pred_tgt_pts, max_dist_px)
    # Assign GT to L3 output (to compute recall denominator)
    l3_assign = assign_gt_predictions(gt_tgt_pts, all_l3_pred_tgt_pts, max_dist_px)

    n_l4_tp = l4_assign.n_matched
    n_l4_inliers = l4_assign.n_pred
    n_l3_tp = l3_assign.n_matched

    precision = n_l4_tp / n_l4_inliers if n_l4_inliers > 0 else float("nan")
    recall = n_l4_tp / n_l3_tp if n_l3_tp > 0 else float("nan")
    pre_rmse = (
        float(np.sqrt(np.mean(l4_assign.distances ** 2)))
        if l4_assign.n_matched > 0 else float("nan")
    )

    return StageScorecard(
        stage="L4",
        pair_id=pair_id,
        matcher=matcher,
        n_gt=l4_assign.n_gt,
        metrics={
            "inlier_precision": precision,
            "inlier_recall": recall,
            "pre_refinement_rmse_px": pre_rmse,
            "n_inliers": n_l4_inliers,
            "n_l4_tp": n_l4_tp,
            "n_l3_tp": n_l3_tp,
        },
    )


def score_l5_refinement(
    gt_tgt_pts: np.ndarray,
    coarse_pred_tgt_pts: np.ndarray,
    refined_pred_tgt_pts: np.ndarray,
    pair_id: str,
    matcher: str,
    max_dist_px: float = 2.0,
) -> StageScorecard:
    """Score L5 — Sub-Pixel Refinement quality.

    Metrics per §3.2 Stage 5:
      refinement_gain:  Mean(L4_error - L5_error) in pixels (positive = improved).
      pct_improved:     Percentage of inliers that moved closer to GT after refinement.
      pct_degraded:     Percentage of inliers that moved further from GT after refinement.
      pct_lt_1px:       Percentage of final (L5) errors < 1.0 px.
      pct_lt_0p5px:     Percentage of final (L5) errors < 0.5 px.

    Args:
        gt_tgt_pts: (N_gt, 2) GT coordinates in target space.
        coarse_pred_tgt_pts: (N_inlier, 2) L4 inlier coordinates (pre-refinement).
        refined_pred_tgt_pts: (N_inlier, 2) L5 refined coordinates.
        pair_id: Pair identifier.
        matcher: Matcher name.
        max_dist_px: GT assignment radius threshold.

    Returns:
        StageScorecard for L5.

    Note:
        coarse_pred_tgt_pts and refined_pred_tgt_pts must be aligned arrays
        (i.e., index i in both refer to the same inlier correspondence).
    """
    assert len(coarse_pred_tgt_pts) == len(refined_pred_tgt_pts), (
        "Coarse and refined prediction arrays must have same length (aligned per inlier)."
    )

    # Assign GT to coarse predictions (L4 → L5 carry-over)
    coarse_assign = assign_gt_predictions(gt_tgt_pts, coarse_pred_tgt_pts, max_dist_px)

    if coarse_assign.n_matched == 0:
        return StageScorecard(
            stage="L5",
            pair_id=pair_id,
            matcher=matcher,
            n_gt=coarse_assign.n_gt,
            metrics={
                "refinement_gain_px": float("nan"),
                "pct_improved": float("nan"),
                "pct_degraded": float("nan"),
                "pct_lt_1px": float("nan"),
                "pct_lt_0p5px": float("nan"),
                "n_inliers_evaluated": 0,
            },
            notes=["No GT matches in L4 output; L5 scores undefined."],
        )

    # For matched pairs: compute coarse and refined errors
    gt_matched = gt_tgt_pts[coarse_assign.gt_indices]       # (K, 2)
    pred_matched_idx = coarse_assign.pred_indices
    coarse_matched = coarse_pred_tgt_pts[pred_matched_idx]  # (K, 2)
    refined_matched = refined_pred_tgt_pts[pred_matched_idx]  # (K, 2)

    coarse_errs = np.linalg.norm(gt_matched - coarse_matched, axis=1)  # (K,)
    refined_errs = np.linalg.norm(gt_matched - refined_matched, axis=1)  # (K,)

    gains = coarse_errs - refined_errs  # positive = improved
    refinement_gain = float(np.mean(gains))
    pct_improved = float(np.mean(gains > 0))
    pct_degraded = float(np.mean(gains < 0))
    pct_lt_1px = float(np.mean(refined_errs < 1.0))
    pct_lt_0p5px = float(np.mean(refined_errs < 0.5))

    return StageScorecard(
        stage="L5",
        pair_id=pair_id,
        matcher=matcher,
        n_gt=coarse_assign.n_gt,
        metrics={
            "refinement_gain_px": refinement_gain,
            "pct_improved": pct_improved,
            "pct_degraded": pct_degraded,
            "pct_lt_1px": pct_lt_1px,
            "pct_lt_0p5px": pct_lt_0p5px,
            "n_inliers_evaluated": coarse_assign.n_matched,
            "l4_rmse_px": float(np.sqrt(np.mean(coarse_errs ** 2))),
            "l5_rmse_px": float(np.sqrt(np.mean(refined_errs ** 2))),
        },
    )


def score_l1_5_routing(
    selected_matcher: str,
    oracle_best_matcher: str,
    pair_id: str,
    all_matcher_metrics: Optional[Dict[str, Dict]] = None,
) -> StageScorecard:
    """Score L1.5 — Matcher Selection Model routing accuracy vs oracle.

    The oracle best matcher is defined a posteriori using the exact composite metric:
      argmax(0.5 * (1/GT_RMSE_norm) + 0.25 * GT_inlier_ratio + 0.25 * GT_spatial_coverage)

    Leakage constraint: the oracle is strictly a reference metric for a posteriori analysis.
    The MSM MUST NOT have access to this during inference (no test data leakage).

    Args:
        selected_matcher: The matcher the MSM selected at inference time.
        oracle_best_matcher: The oracle best matcher (computed from GT metrics).
        pair_id: Pair identifier.
        all_matcher_metrics: Optional dict mapping matcher_name -> metric dict
            (for computing detailed oracle score breakdown).

    Returns:
        StageScorecard for L1.5.
    """
    routing_correct = int(selected_matcher == oracle_best_matcher)

    return StageScorecard(
        stage="L1.5",
        pair_id=pair_id,
        matcher=selected_matcher,
        n_gt=0,  # N/A for routing accuracy
        metrics={
            "routing_correct": routing_correct,
            "oracle_best_matcher": oracle_best_matcher,
            "selected_matcher": selected_matcher,
        },
        notes=[] if routing_correct else [
            f"MSM selected '{selected_matcher}' but oracle preferred '{oracle_best_matcher}'."
        ],
    )


def compute_oracle_best_matcher(
    matcher_metrics: Dict[str, Dict],
    oracle_weights: Optional[Dict[str, float]] = None,
) -> str:
    """Compute the oracle best matcher from GT metrics using the composite score.

    Oracle formula (per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §3.2 Stage 1):
      score = w_rmse * (1/GT_RMSE_norm) + w_inlier * GT_inlier_ratio + w_cov * GT_spatial_coverage
      oracle_best = argmax(score)

    GT_RMSE_norm is the RMSE normalized to [0, 1] across all matchers.

    Args:
        matcher_metrics: Dict {matcher_name: {gt_rmse_px, gt_inlier_ratio, gt_spatial_coverage}}.
        oracle_weights: Optional override for composite weights. Defaults to
            {gt_rmse_norm: 0.5, gt_inlier_ratio: 0.25, gt_spatial_coverage: 0.25}.

    Returns:
        Name of the oracle best matcher.
    """
    weights = oracle_weights or {
        "gt_rmse_norm": 0.50,
        "gt_inlier_ratio": 0.25,
        "gt_spatial_coverage": 0.25,
    }

    # Collect valid matchers (those with a finite RMSE)
    valid = {
        m: v for m, v in matcher_metrics.items()
        if isinstance(v.get("gt_rmse_px"), float) and np.isfinite(v["gt_rmse_px"])
    }
    if not valid:
        logger.warning("compute_oracle_best_matcher: No valid matchers with finite RMSE. Returning fallback 'sift'.")
        return "sift"

    rmse_vals = np.array([v["gt_rmse_px"] for v in valid.values()])
    rmse_max = float(rmse_vals.max()) + 1e-8
    rmse_min = float(rmse_vals.min())

    scores: Dict[str, float] = {}
    for matcher, v in valid.items():
        rmse_norm = (v["gt_rmse_px"] - rmse_min) / (rmse_max - rmse_min + 1e-8)
        rmse_inv_norm = 1.0 - rmse_norm   # higher = better (lower RMSE)
        inlier_ratio = float(v.get("gt_inlier_ratio", 0.0))
        spatial_cov = float(v.get("gt_spatial_coverage", 0.0))
        scores[matcher] = (
            weights["gt_rmse_norm"] * rmse_inv_norm
            + weights["gt_inlier_ratio"] * inlier_ratio
            + weights["gt_spatial_coverage"] * spatial_cov
        )

    return max(scores, key=lambda m: scores[m])


# ---------------------------------------------------------------------------
# Statistical Reporting (N=50, 95% CI)
# ---------------------------------------------------------------------------

def aggregate_scorecards(
    scorecards_per_seed: List[StageScorecard],
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Aggregate per-seed scorecards into mean ± CI statistics.

    Per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §5 Phase 4:
    Each specific condition must be run across N=50 independent seeds.
    Reports mean and 95% confidence interval for every metric.

    Args:
        scorecards_per_seed: List of StageScorecard objects (one per seed).
        confidence_level: Confidence level for CI (default 0.95 → ±1.96 SE).

    Returns:
        Dict with {metric_mean, metric_ci_low, metric_ci_high} for each metric.
    """
    if not scorecards_per_seed:
        return {}

    z = 1.96 if confidence_level == 0.95 else 1.645  # z-score for CI
    # Collect all metric names
    all_keys = set()
    for sc in scorecards_per_seed:
        all_keys.update(sc.metrics.keys())

    summary: Dict[str, float] = {}
    for key in all_keys:
        vals = np.array([
            sc.metrics.get(key, float("nan")) for sc in scorecards_per_seed
        ], dtype=np.float64)
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_ci_low"] = float("nan")
            summary[f"{key}_ci_high"] = float("nan")
            summary[f"{key}_n"] = 0
            continue
        mean = float(np.mean(valid))
        se = float(np.std(valid, ddof=1)) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
        summary[f"{key}_mean"] = mean
        summary[f"{key}_ci_low"] = mean - z * se
        summary[f"{key}_ci_high"] = mean + z * se
        summary[f"{key}_n"] = len(valid)

    return summary

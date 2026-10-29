"""
src/evaluation/metrics.py
--------------------------
F22 — Evaluation Metrics (ARCHITECTURE.md L7)

All metrics are computed ONLY on GT checkpoints with partition="eval".
Inserting a "fit" or "qc" partition point MUST NOT change RMSE.

Per VALIDATION.md §4 — metric definitions used verbatim.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def rmse(predicted_ref_xy: np.ndarray, gt_ref_xy: np.ndarray) -> float:
    """
    RMSE of predicted reference coordinates vs GT checkpoints.

    Parameters
    ----------
    predicted_ref_xy : (N, 2) float — pipeline-predicted ref coords (col, row)
    gt_ref_xy        : (N, 2) float — GT annotated ref coords (col, row)

    Returns
    -------
    RMSE in pixels (Euclidean distance).
    """
    assert predicted_ref_xy.shape == gt_ref_xy.shape, \
        "predicted and GT must have same shape"
    assert predicted_ref_xy.ndim == 2, "Expected 2-D (N,2) array: (col, row)"
    assert predicted_ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    residuals = np.linalg.norm(predicted_ref_xy - gt_ref_xy, axis=1)
    return float(np.sqrt(np.mean(residuals ** 2)))


def pct_lt_1px(predicted_ref_xy: np.ndarray, gt_ref_xy: np.ndarray) -> float:
    """Fraction of GT checkpoints with residual < 1.0 px."""
    assert predicted_ref_xy.shape == gt_ref_xy.shape, "predicted and GT must have same shape"
    assert predicted_ref_xy.ndim == 2 and predicted_ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    residuals = np.linalg.norm(predicted_ref_xy - gt_ref_xy, axis=1)
    return float(np.mean(residuals < 1.0))


def pct_lt_0p5px(predicted_ref_xy: np.ndarray, gt_ref_xy: np.ndarray) -> float:
    """Fraction of GT checkpoints with residual < 0.5 px (sub-pixel precision)."""
    assert predicted_ref_xy.shape == gt_ref_xy.shape, "predicted and GT must have same shape"
    assert predicted_ref_xy.ndim == 2 and predicted_ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    residuals = np.linalg.norm(predicted_ref_xy - gt_ref_xy, axis=1)
    return float(np.mean(residuals < 0.5))


def medae(predicted_ref_xy: np.ndarray, gt_ref_xy: np.ndarray) -> float:
    """Median absolute error in pixels. Robust to outlier GT errors."""
    assert predicted_ref_xy.shape == gt_ref_xy.shape, "predicted and GT must have same shape"
    assert predicted_ref_xy.ndim == 2 and predicted_ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    residuals = np.linalg.norm(predicted_ref_xy - gt_ref_xy, axis=1)
    return float(np.median(residuals))


def refinement_gain(rmse_coarse: float, rmse_refined: float) -> float:
    """
    Refinement gain = RMSE_coarse - RMSE_refined.
    Positive = refinement helped. Negative = refinement hurt.
    """
    return rmse_coarse - rmse_refined


# ---------------------------------------------------------------------------
# Spatial coverage metrics
# ---------------------------------------------------------------------------

def spatial_coverage(
    match_xy: np.ndarray,
    image_shape: Tuple[int, int],
    n: int = 8,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Coverage = occupied_cells / valid_cells over an NxN grid.

    valid_cells = grid cells with mask_fraction < 0.5 (per VALIDATION.md §4).
    If no valid_mask is provided, all cells are considered valid.

    Parameters
    ----------
    match_xy    : (M, 2) float — match coordinates in source image (col, row)
    image_shape : (H, W)
    n           : grid size (default 8 → 8×8 = 64 cells)
    valid_mask  : (H, W) bool — True = INVALID (masked out). Optional.

    Returns
    -------
    coverage in [0, 1]
    """
    assert match_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    h, w = image_shape
    cell_h = h / n
    cell_w = w / n

    occupied = 0
    valid_cells = 0

    for ri in range(n):
        for ci in range(n):
            # Cell bounds
            r0, r1 = ri * cell_h, (ri + 1) * cell_h
            c0, c1 = ci * cell_w, (ci + 1) * cell_w

            # Check if cell is valid (mask_fraction < 0.5)
            if valid_mask is not None:
                r0i, r1i = int(r0), int(r1)
                c0i, c1i = int(c0), int(c1)
                cell_mask = valid_mask[r0i:r1i, c0i:c1i]
                if cell_mask.size > 0:
                    mask_frac = cell_mask.mean()
                    if mask_frac >= 0.5:
                        continue  # invalid cell — skip entirely
            valid_cells += 1

            # Check if any match falls in this cell
            in_cell = (
                (match_xy[:, 0] >= c0) & (match_xy[:, 0] < c1) &
                (match_xy[:, 1] >= r0) & (match_xy[:, 1] < r1)
            )
            if in_cell.any():
                occupied += 1

    if valid_cells == 0:
        return 0.0
    return occupied / valid_cells


def grid_density_std(
    match_xy: np.ndarray,
    image_shape: Tuple[int, int],
    n: int = 8,
) -> float:
    """
    Std-dev of match count per cell over the NxN grid.
    Lower = more uniform distribution.
    """
    assert match_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    h, w = image_shape
    cell_h = h / n
    cell_w = w / n

    counts = np.zeros((n, n), dtype=int)
    for m in match_xy:
        ci = int(min(m[0] / cell_w, n - 1))
        ri = int(min(m[1] / cell_h, n - 1))
        counts[ri, ci] += 1

    return float(np.std(counts))


# ---------------------------------------------------------------------------
# Precision / Recall / Matching Score (where GT match set is available)
# ---------------------------------------------------------------------------

def precision_recall_matching_score(
    predicted_ref_xy: np.ndarray,
    gt_ref_xy: np.ndarray,
    threshold_px: float = 3.0,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and matching_score.

    TP = predicted match within threshold_px of GT match.
    precision = TP / (TP + FP) = TP / len(predicted)
    recall    = TP / total_GT_matches
    matching_score = (precision + recall) / 2

    Parameters
    ----------
    predicted_ref_xy : (N, 2) float — predicted reference coords
    gt_ref_xy        : (M, 2) float — GT reference coords
    threshold_px     : distance threshold for TP (default 3 px per VALIDATION.md)

    Returns
    -------
    (precision, recall, matching_score)
    """
    assert predicted_ref_xy.ndim == 2 and predicted_ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert gt_ref_xy.ndim == 2 and gt_ref_xy.shape[-1] == 2, "Expected (M,2) array: (col, row)"
    n_pred = len(predicted_ref_xy)
    n_gt = len(gt_ref_xy)

    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0

    # For each predicted point, find nearest GT and check threshold
    tp = 0
    for pred_pt in predicted_ref_xy:
        dists = np.linalg.norm(gt_ref_xy - pred_pt, axis=1)
        if dists.min() < threshold_px:
            tp += 1

    precision = tp / n_pred
    recall = tp / n_gt
    ms = (precision + recall) / 2.0
    return precision, recall, ms


# ---------------------------------------------------------------------------
# GT inter-annotator RMSE (mandatory to report with every RMSE claim)
# ---------------------------------------------------------------------------

def gt_interannotator_rmse(
    original_xy: np.ndarray,
    reannotated_xy: np.ndarray,
) -> float:
    """
    Compute inter-annotator RMSE from the "qc" partition.

    This is the demonstrated precision of the GT itself.
    Per VALIDATION.md §4: no algorithmic accuracy claim may be presented
    as meaningful if claimed precision < gt_interannotator_rmse_px.

    Parameters
    ----------
    original_xy    : (N, 2) — original annotation (col, row)
    reannotated_xy : (N, 2) — re-annotation of same points

    Returns
    -------
    inter-annotator RMSE in pixels
    """
    assert original_xy.shape == reannotated_xy.shape, "original and reannotated must have same shape"
    assert original_xy.ndim == 2 and original_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    return rmse(original_xy, reannotated_xy)


# ---------------------------------------------------------------------------
# Composite: compute all metrics from GT file + predicted coords
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predicted_ref_xy: np.ndarray,
    gt_checkpoints: List[dict],
    rmse_coarse_px: Optional[float] = None,
    rmse_refined_px: Optional[float] = None,
    match_xy_for_coverage: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    runtime_s: float = 0.0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
) -> dict:
    """
    Compute the full metric suite from a GT checkpoint list.

    Only uses checkpoints with partition="eval" for RMSE computation.
    "fit" and "qc" checkpoints do NOT affect RMSE (per INTERFACES.md §7).

    Parameters
    ----------
    predicted_ref_xy : (N, 2) — predicted ref coords matching eval checkpoints
    gt_checkpoints : list of dicts with fields: id, src_xy, ref_xy, partition
    rmse_coarse_px, rmse_refined_px : optional — from refinement stage
    match_xy_for_coverage : all selected match coords for coverage computation
    image_shape : (H, W) — for spatial coverage
    runtime_s : wall-clock matching+refinement time

    Returns
    -------
    dict with all metric keys matching EvaluationRecord.metrics schema
    """
    # Filter to eval partition only
    assert predicted_ref_xy.ndim == 2 and predicted_ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    if match_xy_for_coverage is not None:
        assert match_xy_for_coverage.ndim == 2 and match_xy_for_coverage.shape[-1] == 2, "Expected (M,2) array: (col, row)"
    eval_pts = [c for c in gt_checkpoints if c.get("partition") == "eval"]
    if not eval_pts:
        logger.warning("No 'eval' partition GT checkpoints found — RMSE not computed")
        return {}

    gt_eval = np.array([[c["ref_xy"][0], c["ref_xy"][1]] for c in eval_pts], dtype=np.float64)
    n_eval = len(eval_pts)

    if len(predicted_ref_xy) != n_eval:
        logger.error(
            "predicted_ref_xy length (%d) != eval GT count (%d)",
            len(predicted_ref_xy), n_eval,
        )
        return {}

    res = np.linalg.norm(predicted_ref_xy - gt_eval, axis=1)
    rmse_val = float(np.sqrt(np.mean(res ** 2)))
    p1 = float(np.mean(res < 1.0))
    p05 = float(np.mean(res < 0.5))
    med = float(np.median(res))

    gain = refinement_gain(rmse_coarse_px, rmse_val) if rmse_coarse_px is not None else None

    coverage = grid_density = None
    if match_xy_for_coverage is not None and image_shape is not None:
        coverage = spatial_coverage(match_xy_for_coverage, image_shape)
        grid_density = grid_density_std(match_xy_for_coverage, image_shape)

    metrics = {
        "rmse_px": rmse_val,
        "rmse_before_refine_px": rmse_coarse_px,
        "pct_lt_1px": p1,
        "pct_lt_0p5px": p05,
        "medae_px": med,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "spatial_coverage": coverage,
        "grid_density_std": grid_density,
        "refinement_gain_px": gain,
        "runtime_s": runtime_s,
        "precision": None,
        "recall": None,
        "matching_score": None,
        "gt_checkpoint_count": n_eval,
    }
    return metrics

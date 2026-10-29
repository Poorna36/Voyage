"""
src/registration/declustering.py
---------------------------------
F18 — GCP Declustering and Z-Score Filtering (ARCHITECTURE.md L4)

After DEGENSAC, enforces GSD-scaled minimum spacing between inlier GCPs,
then removes residual outliers via Z-score filtering.

Coordinate convention: all coords are (col, row) = (x, y), 0-indexed.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Base GSD for min_spacing scaling (NAC at 0.5 m/px — per FEATURES.md F18)
BASE_GSD_M: float = 0.5
DEFAULT_MIN_SPACING_PX: float = 20.0   # at NAC 0.5 m/px reference
DEFAULT_ZSCORE_THRESHOLD: float = 3.0
MIN_GCPS_FOR_ZSCORE: int = 20


def decluster(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    residuals: np.ndarray,
    ref_gsd_m: float,
    min_spacing_px: float = DEFAULT_MIN_SPACING_PX,
    base_gsd_m: float = BASE_GSD_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Enforce GSD-scaled minimum spacing between GCP inliers.

    Uses a greedy grid-nearest-centre strategy:
      - Sort inliers by residual (ascending — keep best first)
      - Accept a point only if no already-accepted point is within
        min_spacing_px_scaled of it (Euclidean distance in ref image pixels)

    GSD scaling (MANDATORY per FEATURES.md F18):
      min_spacing_px_scaled = min_spacing_px * (ref_gsd_m / base_gsd_m)
      Examples:
        NAC ref (0.5 m)   → 20 px
        TMC ref (5 m)     → ~200 px
        IIRS ref (80 m)   → ~3200 px

    Parameters
    ----------
    src_xy : (N, 2) float  — source inlier coords (col, row)
    ref_xy : (N, 2) float  — reference inlier coords (col, row)
    residuals : (N,) float — per-inlier reprojection residual (px)
    ref_gsd_m : float      — reference image GSD in metres/pixel
    min_spacing_px : float — baseline minimum spacing (at base_gsd_m)
    base_gsd_m : float     — baseline GSD (default 0.5 m for NAC)

    Returns
    -------
    src_xy_out, ref_xy_out, residuals_out — after spacing filter
    gsd_scale_factor — the applied scale factor (for geometry.json)
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert src_xy.shape[0] == ref_xy.shape[0] == residuals.shape[0]

    gsd_scale_factor = ref_gsd_m / base_gsd_m
    min_spacing_scaled = min_spacing_px * gsd_scale_factor

    logger.debug(
        "Declustering: ref_gsd=%.2f m, scale=%.2fx → min_spacing=%.1f px",
        ref_gsd_m, gsd_scale_factor, min_spacing_scaled,
    )

    n = len(src_xy)
    if n == 0:
        return src_xy, ref_xy, residuals, gsd_scale_factor

    # Sort by residual ascending — keep lowest residual points first
    order = np.argsort(residuals)
    src_s = src_xy[order]
    ref_s = ref_xy[order]
    res_s = residuals[order]

    keep = np.zeros(n, dtype=bool)
    accepted_ref: list[np.ndarray] = []

    for i in range(n):
        pt = ref_s[i]  # use ref coords for spacing (physical ground spacing)
        if len(accepted_ref) == 0:
            keep[i] = True
            accepted_ref.append(pt)
        else:
            accepted_arr = np.array(accepted_ref)               # (M, 2)
            dists = np.linalg.norm(accepted_arr - pt, axis=1)  # (M,)
            if np.min(dists) >= min_spacing_scaled:
                keep[i] = True
                accepted_ref.append(pt)

    removed = int((~keep).sum())
    if removed > 0:
        logger.debug("Declustering spacing filter: removed %d / %d GCPs", removed, n)

    return src_s[keep], ref_s[keep], res_s[keep], gsd_scale_factor


def zscore_filter(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    residuals: np.ndarray,
    threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    min_gcps: int = MIN_GCPS_FOR_ZSCORE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove residual outliers using Z-score filtering.

    Only runs when the number of GCPs exceeds min_gcps (default 20).
    If fewer GCPs are available, returns inputs unchanged.

    Z-score = (residual - mean(residuals)) / std(residuals)
    Points with |Z-score| > threshold are rejected.

    Parameters
    ----------
    src_xy : (N, 2) float — source coords (col, row)
    ref_xy : (N, 2) float — reference coords (col, row)
    residuals : (N,) float — per-point reprojection residual in pixels
    threshold : float — Z-score cutoff (default 3.0)
    min_gcps : int — minimum count required to run filter (default 20)

    Returns
    -------
    src_xy_out, ref_xy_out, residuals_out
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    n = len(residuals)
    if n < min_gcps:
        logger.debug(
            "Z-score filter skipped: only %d GCPs (need >= %d)", n, min_gcps
        )
        return src_xy, ref_xy, residuals

    mean_res = np.mean(residuals)
    std_res = np.std(residuals)

    if std_res < 1e-9:
        logger.debug("Z-score filter: zero std — all residuals identical, keeping all")
        return src_xy, ref_xy, residuals

    zscores = np.abs((residuals - mean_res) / std_res)
    keep = zscores <= threshold

    removed = int((~keep).sum())
    if removed > 0:
        logger.debug(
            "Z-score filter (thr=%.1f): removed %d / %d GCPs", threshold, removed, n
        )

    return src_xy[keep], ref_xy[keep], residuals[keep]


def decluster_and_filter(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    residuals: np.ndarray,
    ref_gsd_m: float,
    min_spacing_px: float = DEFAULT_MIN_SPACING_PX,
    base_gsd_m: float = BASE_GSD_M,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    min_gcps_for_zscore: int = MIN_GCPS_FOR_ZSCORE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Combined pipeline: spacing decluster → Z-score filter.

    Returns
    -------
    src_xy_out, ref_xy_out, residuals_out, gsd_scale_factor, final_gcp_count
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    src_xy, ref_xy, residuals, gsd_scale_factor = decluster(
        src_xy, ref_xy, residuals,
        ref_gsd_m=ref_gsd_m,
        min_spacing_px=min_spacing_px,
        base_gsd_m=base_gsd_m,
    )
    src_xy, ref_xy, residuals = zscore_filter(
        src_xy, ref_xy, residuals,
        threshold=zscore_threshold,
        min_gcps=min_gcps_for_zscore,
    )

    logger.info(
        "GCP declustering complete: %d GCPs remaining (gsd_scale=%.2f)",
        len(src_xy), gsd_scale_factor,
    )
    return src_xy, ref_xy, residuals, gsd_scale_factor, len(src_xy)

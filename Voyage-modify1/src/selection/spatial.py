"""
src/selection/spatial.py
========================
Spatial uniformity selection pipeline (L3 — S5 gate).

Functions
---------
confidence_filter  : threshold-based per-matcher confidence cut
grid_cap           : NxN grid, per-cell maximum cap
coverage_greedy    : bisection to meet coverage_min >= 0.60
one_to_one         : deduplicate, keep highest confidence
selection_stats    : coverage + grid_density_std before & after

Configuration (CONFIGURATION.md §5):
  n             : 8        (grid NxN)
  cap           : 5        (per-cell cap)
  budget        : 250      (total match budget)
  coverage_min  : 0.60     (S5 gate threshold)

Per-matcher confidence thresholds (CONFIGURATION.md §5):
  SIFT      : 0.0   (no filtering — all SIFT matches pass)
  RIFT2     : 0.0
  LightGlue : 0.2   (native confidence from LightGlue model)
  Crater    : 0.65  (high threshold for topological matches)

References: ARCHITECTURE.md §3, FEATURES.md F14, CONFIGURATION.md §5
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Per-matcher confidence thresholds (CONFIGURATION.md §5)
CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "sift": 0.0,
    "rift2": 0.0,
    "lnift": 0.0,
    "lightglue": 0.2,
    "crater": 0.65,
    "crater_hough": 0.65,
}

# ── Public API ────────────────────────────────────────────────────────────────


def confidence_filter(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    matcher_id: str,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter matches below the per-matcher confidence threshold.

    Parameters
    ----------
    src_xy, ref_xy : (N, 2) float32 — (col, row) coordinates
    confidence     : (N,)   float32
    matcher_id     : used to look up default threshold if not provided
    threshold      : override threshold; None = use CONFIDENCE_THRESHOLDS

    Returns filtered (src_xy, ref_xy, confidence).
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    tau = threshold if threshold is not None else CONFIDENCE_THRESHOLDS.get(matcher_id, 0.0)
    keep = confidence >= tau
    return src_xy[keep], ref_xy[keep], confidence[keep]


def grid_cap(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    n: int = 8,
    cap: int = 5,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NxN grid cap — keep at most *cap* highest-confidence matches per cell.

    Parameters
    ----------
    src_xy       : (N, 2) float32 — (col, row) — used to assign grid cells
    n            : grid side length (cells per axis)
    cap          : max matches per grid cell
    image_shape  : (H, W) for grid normalization; inferred from coords if None

    Returns filtered (src_xy, ref_xy, confidence).
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    if len(src_xy) == 0:
        return src_xy, ref_xy, confidence

    if image_shape is not None:
        H, W = image_shape
    else:
        H = float(src_xy[:, 1].max()) + 1.0   # row max
        W = float(src_xy[:, 0].max()) + 1.0   # col max

    # Assign grid cell — (col / W * n, row / H * n) clipped to [0, n-1]
    cell_x = np.clip((src_xy[:, 0] / W * n).astype(int), 0, n - 1)  # col -> x cell
    cell_y = np.clip((src_xy[:, 1] / H * n).astype(int), 0, n - 1)  # row -> y cell
    cell_id = cell_y * n + cell_x    # linear cell index

    keep_mask = np.zeros(len(src_xy), dtype=bool)
    sort_order = np.argsort(-confidence)   # highest confidence first

    cell_count: Dict[int, int] = {}
    for idx in sort_order:
        cid = int(cell_id[idx])
        if cell_count.get(cid, 0) < cap:
            keep_mask[idx] = True
            cell_count[cid] = cell_count.get(cid, 0) + 1

    return src_xy[keep_mask], ref_xy[keep_mask], confidence[keep_mask]


def coverage_greedy(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    budget: int = 250,
    min_coverage: float = 0.60,
    n: int = 8,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select up to *budget* matches via bisection on confidence threshold,
    targeting spatial coverage >= min_coverage on an NxN grid.

    Algorithm:
      Binary-search on confidence threshold tau in [0, 1].
      Accept the smallest tau giving:
        - coverage >= min_coverage   AND
        - match count <= budget

    If coverage_min cannot be met, return best-effort result.

    Returns filtered (src_xy, ref_xy, confidence).
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    if len(src_xy) == 0:
        return src_xy, ref_xy, confidence

    if image_shape is not None:
        H, W = float(image_shape[0]), float(image_shape[1])
    else:
        H = float(src_xy[:, 1].max()) + 1.0
        W = float(src_xy[:, 0].max()) + 1.0

    def _coverage(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        pts = src_xy[mask]
        cx = np.clip((pts[:, 0] / W * n).astype(int), 0, n - 1)
        cy = np.clip((pts[:, 1] / H * n).astype(int), 0, n - 1)
        occupied = len(set(zip(cx.tolist(), cy.tolist())))
        return occupied / (n * n)

    # Binary search on threshold
    lo, hi = 0.0, 1.0
    best_mask = np.ones(len(confidence), dtype=bool)

    for _ in range(50):
        mid = (lo + hi) / 2.0
        mask = confidence >= mid
        if mask.sum() == 0:
            hi = mid
            continue
        cov = _coverage(mask)
        count = mask.sum()
        if cov >= min_coverage and count <= budget:
            best_mask = mask
            hi = mid    # try stricter threshold (fewer, higher-quality)
        else:
            lo = mid    # relax threshold
        if hi - lo < 1e-4:
            break

    # Final cap to budget
    if best_mask.sum() > budget:
        idxs = np.where(best_mask)[0]
        top = idxs[np.argsort(-confidence[idxs])[:budget]]
        best_mask = np.zeros(len(confidence), dtype=bool)
        best_mask[top] = True

    return src_xy[best_mask], ref_xy[best_mask], confidence[best_mask]


def one_to_one(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    tol_px: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce one-to-one constraint: for duplicate source or reference coords
    (within tol_px), keep the match with highest confidence.

    Returns deduplicated (src_xy, ref_xy, confidence).
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    if len(src_xy) <= 1:
        return src_xy, ref_xy, confidence

    # Sort by descending confidence; greedily accept if no collision
    order = np.argsort(-confidence)
    accepted_src: list = []
    accepted_ref: list = []
    accepted_conf: list = []
    used_src: list = []   # list of (col, row) already assigned
    used_ref: list = []

    for idx in order:
        sc = src_xy[idx]
        rc = ref_xy[idx]
        # Check collisions
        src_collision = any(
            abs(float(sc[0]) - float(u[0])) < tol_px and
            abs(float(sc[1]) - float(u[1])) < tol_px
            for u in used_src
        )
        ref_collision = any(
            abs(float(rc[0]) - float(u[0])) < tol_px and
            abs(float(rc[1]) - float(u[1])) < tol_px
            for u in used_ref
        )
        if not src_collision and not ref_collision:
            accepted_src.append(sc)
            accepted_ref.append(rc)
            accepted_conf.append(confidence[idx])
            used_src.append(sc)
            used_ref.append(rc)

    if not accepted_src:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.float32))

    return (
        np.array(accepted_src, dtype=np.float32),
        np.array(accepted_ref, dtype=np.float32),
        np.array(accepted_conf, dtype=np.float32),
    )


def selection_stats(
    src_xy_before: np.ndarray,
    src_xy_after: np.ndarray,
    confidence_before: np.ndarray,
    confidence_after: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    n: int = 8,
) -> Dict[str, Any]:
    """
    Compute coverage and grid_density_std before and after selection.

    Returns dict with keys:
      n_before, n_after, coverage_before, coverage_after,
      grid_density_std_before, grid_density_std_after
    """
    assert src_xy_before.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert src_xy_after.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    def _stats(pts: np.ndarray) -> Tuple[float, float]:
        if len(pts) == 0:
            return 0.0, 0.0
        assert pts.shape[-1] == 2, "Expected (N,2) array: (col, row)"
        if image_shape is not None:
            H, W = float(image_shape[0]), float(image_shape[1])
        else:
            H = float(pts[:, 1].max()) + 1.0
            W = float(pts[:, 0].max()) + 1.0
        cx = np.clip((pts[:, 0] / W * n).astype(int), 0, n - 1)
        cy = np.clip((pts[:, 1] / H * n).astype(int), 0, n - 1)
        grid = np.zeros((n, n), dtype=int)
        for x, y in zip(cx, cy):
            grid[y, x] += 1
        total_cells = n * n
        occupied = int((grid > 0).sum())
        coverage = occupied / total_cells
        density_std = float(grid.flatten().std())
        return coverage, density_std

    cov_b, std_b = _stats(src_xy_before)
    cov_a, std_a = _stats(src_xy_after)

    return {
        "n_before": int(len(src_xy_before)),
        "n_after": int(len(src_xy_after)),
        "coverage_before": round(cov_b, 4),
        "coverage_after": round(cov_a, 4),
        "grid_density_std_before": round(std_b, 4),
        "grid_density_std_after": round(std_a, 4),
        "grid_n": n,
    }

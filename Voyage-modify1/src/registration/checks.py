"""
src/registration/checks.py
--------------------------
F2 Mandatory Checks (Feature F15) — ARCHITECTURE.md L4

Must be called on every match set BEFORE any RANSAC/DEGENSAC step.
Applies to all matchers, but is especially critical for M2 (LightGlue)
and M3 (Crater) which can produce out-of-bounds or duplicate coordinates.

Coordinate convention: all coords are (col, row) = (x, y), 0-indexed.
NEVER (row, col). See INTERFACES.md §8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class F2CheckResult:
    """Result of F2 mandatory checks."""
    src_xy: np.ndarray          # (N, 2) float32 — filtered matches source coords
    ref_xy: np.ndarray          # (N, 2) float32 — filtered matches reference coords
    confidence: np.ndarray      # (N,)   float32 — per-match confidence
    original_count: int         # number of matches before F2
    removed_oob: int            # removed: out-of-bounds (source or reference)
    removed_dup: int            # removed: duplicate source or reference coord
    final_count: int            # number of matches after F2


def f2_checks(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    src_shape: Tuple[int, int],
    ref_shape: Tuple[int, int],
    buffer_px: int = 10,
) -> F2CheckResult:
    """
    Apply mandatory F2 checks to a match set.

    Steps (in order):
      1. In-domain bounds check — reject matches where src or ref coords
         fall outside the image bounds (with a buffer_px margin).
      2. One-to-one constraint — for any src coord that appears more than
         once, keep only the highest-confidence match. Same for ref coords.

    Parameters
    ----------
    src_xy : np.ndarray, shape (N, 2), dtype float
        Source image coordinates, (col, row) convention.
    ref_xy : np.ndarray, shape (N, 2), dtype float
        Reference image coordinates, (col, row) convention.
    confidence : np.ndarray, shape (N,), dtype float
        Per-match confidence score (higher = better).
    src_shape : (height, width)
        Source image shape in pixels.
    ref_shape : (height, width)
        Reference image shape in pixels.
    buffer_px : int
        Extra margin beyond image boundary that is still considered valid.
        Positive buffer → slightly allows coords just outside the edge.
        Default 10 px (per FEATURES.md F15).

    Returns
    -------
    F2CheckResult
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert src_xy.shape[0] == ref_xy.shape[0] == confidence.shape[0], \
        "src_xy, ref_xy and confidence must have the same length"

    src_xy = src_xy.astype(np.float32)
    ref_xy = ref_xy.astype(np.float32)
    confidence = confidence.astype(np.float32)
    original_count = len(src_xy)

    # ------------------------------------------------------------------
    # Step 1: in-domain bounds check
    # src_shape / ref_shape are (H, W); coords are (col, row) → (x, y)
    # ------------------------------------------------------------------
    src_h, src_w = src_shape
    ref_h, ref_w = ref_shape

    src_col, src_row = src_xy[:, 0], src_xy[:, 1]
    ref_col, ref_row = ref_xy[:, 0], ref_xy[:, 1]

    in_src = (
        (src_col >= -buffer_px) & (src_col < src_w + buffer_px) &
        (src_row >= -buffer_px) & (src_row < src_h + buffer_px)
    )
    in_ref = (
        (ref_col >= -buffer_px) & (ref_col < ref_w + buffer_px) &
        (ref_row >= -buffer_px) & (ref_row < ref_h + buffer_px)
    )
    in_bounds = in_src & in_ref

    removed_oob = int((~in_bounds).sum())
    src_xy = src_xy[in_bounds]
    ref_xy = ref_xy[in_bounds]
    confidence = confidence[in_bounds]

    if removed_oob > 0:
        logger.debug("F2 bounds check: removed %d out-of-bounds matches", removed_oob)

    # ------------------------------------------------------------------
    # Step 2: one-to-one constraint
    # For each src coord appearing more than once, keep highest confidence.
    # Same for ref coords.
    # Strategy: sort by confidence descending, then deduplicate by rounded
    # coordinates (sub-pixel duplicates treated as same point at 0.5 px).
    # ------------------------------------------------------------------
    n = len(src_xy)
    if n == 0:
        return F2CheckResult(
            src_xy=src_xy, ref_xy=ref_xy, confidence=confidence,
            original_count=original_count,
            removed_oob=removed_oob, removed_dup=0, final_count=0,
        )

    # Sort descending by confidence so greedy keep-first = keep-best
    order = np.argsort(-confidence)
    src_xy = src_xy[order]
    ref_xy = ref_xy[order]
    confidence = confidence[order]

    # Round to 0.5 px resolution for dedup key
    src_keys = np.round(src_xy * 2).astype(np.int32)   # (N, 2)
    ref_keys = np.round(ref_xy * 2).astype(np.int32)   # (N, 2)

    keep = np.ones(n, dtype=bool)
    seen_src: set = set()
    seen_ref: set = set()

    for i in range(n):
        sk = (src_keys[i, 0], src_keys[i, 1])
        rk = (ref_keys[i, 0], ref_keys[i, 1])
        if sk in seen_src or rk in seen_ref:
            keep[i] = False
        else:
            seen_src.add(sk)
            seen_ref.add(rk)

    removed_dup = int((~keep).sum())
    src_xy = src_xy[keep]
    ref_xy = ref_xy[keep]
    confidence = confidence[keep]

    if removed_dup > 0:
        logger.debug("F2 one-to-one: removed %d duplicate matches", removed_dup)

    final_count = len(src_xy)
    logger.info(
        "F2 checks complete: %d → %d matches (oob=%d, dup=%d)",
        original_count, final_count, removed_oob, removed_dup,
    )

    return F2CheckResult(
        src_xy=src_xy,
        ref_xy=ref_xy,
        confidence=confidence,
        original_count=original_count,
        removed_oob=removed_oob,
        removed_dup=removed_dup,
        final_count=final_count,
    )

"""
src/selection/anms.py
=====================
Adaptive Non-Maximal Suppression via Square Covering (SSC).

Reference: Bailo et al. 2018 — "Efficient Adaptive Non-Maximal Suppression
           Algorithms for Homogeneous Spatial Keypoint Distribution"

Time complexity: O(n log n) — grid binary-search approach.

Validation criterion (VALIDATION.md T05):
  - No two output keypoints within computed suppression radius
  - Output budget within +/-5% of target num_points
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np


def anms_ssc(
    keypoints,
    num_points: int,
    image_shape: Tuple[int, int],
    robust_coeff: float = 1.11,
) -> list:
    """
    Reduce *keypoints* to *num_points* with maximum spatial uniformity.

    Parameters
    ----------
    keypoints   : list of cv2.KeyPoint OR (N, 3) ndarray of (x, y, strength).
    num_points  : target output count.
    image_shape : (height, width) of the image being processed.
    robust_coeff: slight over-expansion factor to offset grid boundary losses
                  (Bailo 2018 default: 1.11).

    Returns
    -------
    Filtered list of the same type as input.
    """
    N = len(keypoints)
    if N == 0 or num_points >= N:
        return list(keypoints)

    if isinstance(keypoints, np.ndarray):
        assert keypoints.shape[-1] >= 2, "Expected (N, 2+) array: (col, row, ...)"

    h, w = image_shape

    # ── Unpack to (x, y, response) ─────────────────────────────────────────
    try:
        import cv2 as _cv2  # noqa: F401
        _is_cv2 = hasattr(keypoints[0], "pt")
    except ImportError:
        _is_cv2 = False

    if _is_cv2:
        xy = np.array([kp.pt for kp in keypoints], dtype=np.float64)        # (N,2) x,y
        resp = np.array([kp.response for kp in keypoints], dtype=np.float64) # (N,)
    else:
        arr = np.asarray(keypoints, dtype=np.float64)
        xy = arr[:, :2]
        resp = arr[:, 2]

    # ── Sort by response descending ─────────────────────────────────────────
    order = np.argsort(-resp)
    xy_s = xy[order]   # sorted positions

    # ── Grid-based SSC: binary search over suppression radius r ────────────
    # For a given r, divide image into cells of size r × r.
    # Iterate keypoints strongest-first; accept if its cell (and 8 neighbours)
    # are all empty, then mark cell occupied.
    # This is O(n × 9) per binary-search step, O(n log D) total.

    def count_accepted(r: float) -> List[int]:
        """Return list of sorted indices accepted at radius r."""
        if r < 1.0:
            return list(range(N))
        grid: dict = {}
        acc: List[int] = []
        for i in range(N):
            cx = int(xy_s[i, 0] / r)
            cy = int(xy_s[i, 1] / r)
            suppressed = any(
                (cx + dx, cy + dy) in grid
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            )
            if not suppressed:
                grid[(cx, cy)] = True
                acc.append(i)
        return acc

    # Binary search: find the smallest r where accepted count ~= num_points
    tol = max(1, int(0.05 * num_points))   # ±5% tolerance (VALIDATION.md T05)
    r_lo, r_hi = 1.0, float(max(w, h))
    best_acc: List[int] = []

    for _ in range(50):
        r_mid = (r_lo + r_hi) / 2.0
        acc = count_accepted(r_mid)
        n_acc = len(acc)
        if abs(n_acc - num_points) <= tol:
            best_acc = acc
            break
        if n_acc > num_points:
            r_lo = r_mid
            best_acc = acc   # over-full: trim later, but store as fallback
        else:
            r_hi = r_mid
            # Under-full: keep as fallback if it's the closest we've seen
            if not best_acc or abs(n_acc - num_points) < abs(len(best_acc) - num_points):
                best_acc = acc
        if r_hi - r_lo < 0.5:
            break

    if not best_acc:
        # Fallback: take top-N by response
        best_acc = list(range(min(num_points, N)))

    # Apply robust coefficient (expand radius slightly, shrink accepted set)
    r_final = (r_lo + r_hi) / 2.0 * robust_coeff
    final_acc = count_accepted(r_final)

    # Floor guard: if robust_coeff over-shrinks below budget, use best_acc
    if len(final_acc) < int(num_points * 0.90) and best_acc:
        final_acc = best_acc

    # Clip to budget if over-accepted
    if len(final_acc) > int(num_points * 1.05):
        final_acc = final_acc[:num_points]

    # ── Map sorted indices → original indices ───────────────────────────────
    original_indices = [int(order[i]) for i in final_acc]

    if _is_cv2:
        return [keypoints[i] for i in original_indices]
    else:
        return [keypoints[i] for i in original_indices]


def keypoints_to_array(keypoints) -> np.ndarray:
    """
    Convert list of cv2.KeyPoint to (N, 3) ndarray of (x, y, response).
    No-op if already an ndarray.
    """
    if isinstance(keypoints, np.ndarray):
        assert keypoints.shape[-1] >= 2, "Expected (N, 2+) array: (col, row, ...)"
        return keypoints
    return np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in keypoints],
                    dtype=np.float32)


def array_to_keypoints(arr: np.ndarray):
    """Convert (N, 3) array of (x, y, response) back to cv2.KeyPoint list."""
    import cv2
    return [cv2.KeyPoint(x=float(r[0]), y=float(r[1]), size=1.0,
                         response=float(r[2])) for r in arr]

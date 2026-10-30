"""
src/refinement/local.py
------------------------
F19 — Sub-pixel Refinement per Inlier (ARCHITECTURE.md L5)

For each DEGENSAC inlier, extract a local patch, apply NCC or phase-only
correlation, and fit a 2D paraboloid to achieve sub-pixel accuracy.

Critical requirements (from FEATURES.md F19 and ARCHITECTURE.md L5):
  - Apodization: ONLY Tukey or Gaussian. NEVER Blackman (demonstrated worst).
  - Second-peak rejection: MANDATORY (lunar repetitive crater texture).
  - Sharpness threshold tau_q: tunable (default 0.15; v1.1 used 0.45).
  - Report RMSE before AND after refinement as separate metrics.
  - >= 70% of inliers must refine successfully; else flag partial_refinement.

Coordinate convention: (col, row) = (x, y), 0-indexed. NEVER (row, col).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import numpy as np
from scipy.signal import windows as scipy_windows

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters (see CONFIGURATION.md — marked TUNE on real data)
# ---------------------------------------------------------------------------
WINDOW_PX: int = 32
PYRAMID_LEVELS: int = 3
SHARPNESS_THRESHOLD: float = 0.10   # tau_q — tuned for lunar regolith cross-correlation
SECOND_PEAK_RATIO: float = 0.80     # reject if 2nd peak > 0.80 × primary
VARIANCE_THRESHOLD: float = 1e-4    # tau_v — flat patch rejection
APODIZATION: Literal["tukey", "gaussian"] = "tukey"


# ---------------------------------------------------------------------------
# Apodization windows
# ---------------------------------------------------------------------------

def _make_2d_window(size: int, method: str) -> np.ndarray:
    """
    Create a 2D apodization window of shape (size, size).

    Allowed: 'tukey' or 'gaussian'. NEVER 'blackman' — see FEATURES.md F19.
    """
    method = method.lower()
    if method == "blackman":
        raise ValueError(
            "Blackman apodization is FORBIDDEN (demonstrated worst choice "
            "per HybridPhaseCorrelation paper). Use 'tukey' or 'gaussian'."
        )
    if method == "tukey":
        w1d = scipy_windows.tukey(size, alpha=0.5).astype(np.float32)
    elif method == "gaussian":
        w1d = scipy_windows.gaussian(size, std=size / 6).astype(np.float32)
    else:
        raise ValueError(f"Unknown apodization method: {method!r}. Use 'tukey' or 'gaussian'.")

    return np.outer(w1d, w1d).astype(np.float32)


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _extract_patch(
    image: np.ndarray,
    col: float,
    row: float,
    half: int,
) -> Tuple[np.ndarray, bool]:
    """
    Extract a (2*half x 2*half) patch centred at (col, row).

    Returns (patch, valid) where valid=False if the patch would go
    out of bounds.
    """
    h, w = image.shape[:2]
    c = int(round(col))
    r = int(round(row))

    r0, r1 = r - half, r + half
    c0, c1 = c - half, c + half

    if r0 < 0 or r1 > h or c0 < 0 or c1 > w:
        return np.zeros((2 * half, 2 * half), dtype=np.float32), False

    patch = image[r0:r1, c0:c1].astype(np.float32)
    if patch.ndim == 3:
        patch = patch.mean(axis=2)  # collapse to greyscale if needed
    return patch, True


# ---------------------------------------------------------------------------
# 2D paraboloid sub-pixel peak fitting
# ---------------------------------------------------------------------------

def _paraboloid_peak(corr: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a 2D paraboloid to the 3×3 neighbourhood of the correlation peak.

    Returns (dx, dy, sharpness) where:
      dx, dy   — sub-pixel displacement from integer peak (col, row)
      sharpness — peak[0,0] / sum(3x3 neighbourhood); higher = sharper
    """
    r_peak, c_peak = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[r_peak, c_peak])

    h, w = corr.shape
    # Guard: peak must not be on the border
    if r_peak == 0 or r_peak == h - 1 or c_peak == 0 or c_peak == w - 1:
        return 0.0, 0.0, 0.0

    # 3×3 neighbourhood
    nbr = corr[r_peak - 1:r_peak + 2, c_peak - 1:c_peak + 2].astype(np.float64)
    sharpness = float(peak_val / (nbr.sum() + 1e-12))

    # Paraboloid fit: dx = (f(-1) - f(+1)) / (2*(f(-1) - 2*f(0) + f(+1)))
    # Applied separately for col and row directions
    fx_m = float(nbr[1, 0])   # col-1
    fx_0 = float(nbr[1, 1])   # col centre
    fx_p = float(nbr[1, 2])   # col+1

    fy_m = float(nbr[0, 1])   # row-1
    fy_0 = float(nbr[1, 1])   # row centre
    fy_p = float(nbr[2, 1])   # row+1

    denom_c = 2.0 * (fx_m - 2.0 * fx_0 + fx_p)
    denom_r = 2.0 * (fy_m - 2.0 * fy_0 + fy_p)

    dx = (fx_m - fx_p) / denom_c if abs(denom_c) > 1e-12 else 0.0
    dy = (fy_m - fy_p) / denom_r if abs(denom_r) > 1e-12 else 0.0

    # Clamp to ±1 px (sanity guard)
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))

    return dx, dy, sharpness


# ---------------------------------------------------------------------------
# Second-peak rejection
# ---------------------------------------------------------------------------

def _second_peak_check(corr: np.ndarray, ratio_threshold: float = SECOND_PEAK_RATIO) -> bool:
    """
    Return True (reject) if a strong second peak exists.

    Finds primary peak, masks out its 5×5 neighbourhood, finds second peak.
    Rejects if second_peak > ratio_threshold × primary_peak.

    This is a MANDATORY check for lunar repetitive crater texture.
    """
    r_peak, c_peak = np.unravel_index(np.argmax(corr), corr.shape)
    primary_val = float(corr[r_peak, c_peak])

    if primary_val < 1e-12:
        return True  # reject flat correlation surface

    masked = corr.copy()
    r0 = max(0, r_peak - 2)
    r1 = min(corr.shape[0], r_peak + 3)
    c0 = max(0, c_peak - 2)
    c1 = min(corr.shape[1], c_peak + 3)
    masked[r0:r1, c0:c1] = 0.0

    second_val = float(masked.max())
    return second_val > ratio_threshold * primary_val


# ---------------------------------------------------------------------------
# NCC correlation
# ---------------------------------------------------------------------------

def _ncc_correlation(patch_src: np.ndarray, patch_ref: np.ndarray) -> np.ndarray:
    """
    Compute normalised cross-correlation surface via FFT.

    Pads the reference patch to allow ±search_half shift detection.
    Returns correlation surface of same shape as patch_ref.
    """
    # Zero-mean both patches
    ps = patch_src - patch_src.mean()
    pr = patch_ref - patch_ref.mean()

    # FFT-based cross-correlation
    size = np.array(pr.shape) + np.array(ps.shape) - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)

    F_ref = np.fft.rfft2(pr, s=fsize)
    F_src = np.fft.rfft2(ps, s=fsize)
    corr_full = np.fft.irfft2(F_ref * np.conj(F_src))

    # Normalise by std product
    norm = (np.std(ps) * np.std(pr) * ps.size)
    if norm < 1e-12:
        return np.zeros_like(patch_ref, dtype=np.float32)

    corr_full /= norm

    # Center zero-lag peak at (h//2, w//2) and crop to match patch_ref shape
    corr_shifted = np.fft.fftshift(corr_full)
    cr, cc = fsize[0] // 2, fsize[1] // 2
    h, w = patch_ref.shape
    corr = corr_shifted[cr - h // 2 : cr + h // 2, cc - w // 2 : cc + w // 2].astype(np.float32)
    return corr


# ---------------------------------------------------------------------------
# Pyramid coarse-to-fine
# ---------------------------------------------------------------------------

def _coarse_to_fine_refine(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    src_col: float,
    src_row: float,
    ref_col: float,
    ref_row: float,
    half: int,
    levels: int,
    apodization: str,
    sharpness_threshold: float,
    second_peak_ratio: float,
    variance_threshold: float,
) -> Tuple[float, float, float, float, bool]:
    """
    Pyramid coarse-to-fine NCC refinement.

    Returns (refined_dcol, refined_drow, sharpness, second_peak_ratio_val, success)
    """
    dcol_total = 0.0
    drow_total = 0.0

    for level in range(levels - 1, -1, -1):
        scale = 2 ** level
        # Downsample images for this pyramid level
        if scale > 1:
            src_ds = img_src[::scale, ::scale]
            ref_ds = img_ref[::scale, ::scale]
            sc = src_col / scale
            sr = src_row / scale
            rc = (ref_col + dcol_total) / scale
            rr = (ref_row + drow_total) / scale
        else:
            src_ds = img_src
            ref_ds = img_ref
            sc = src_col
            sr = src_row
            rc = ref_col + dcol_total
            rr = ref_row + drow_total

        # Extract patches at current level
        patch_s, valid_s = _extract_patch(src_ds, sc, sr, half // scale or 1)
        patch_r, valid_r = _extract_patch(ref_ds, rc, rr, half // scale or 1)

        if not valid_s or not valid_r:
            return dcol_total, drow_total, 0.0, 0.0, False

        # Variance check (flat patch rejection) — only at finest level
        if level == 0:
            if patch_s.var() < variance_threshold or patch_r.var() < variance_threshold:
                return dcol_total, drow_total, 0.0, 0.0, False

        # Apodization
        win = _make_2d_window(patch_s.shape[0], apodization)
        patch_s = patch_s * win
        patch_r = patch_r * win

        # NCC correlation
        corr = _ncc_correlation(patch_s, patch_r)

        # At finest level: second-peak check
        if level == 0:
            if _second_peak_check(corr, second_peak_ratio):
                return dcol_total, drow_total, 0.0, 0.0, False

        # Sub-pixel peak
        dx, dy, sharpness = _paraboloid_peak(corr)

        if level == 0 and sharpness < sharpness_threshold:
            return dcol_total, drow_total, sharpness, 0.0, False

        # Accumulate displacement (scaled back to full resolution)
        dcol_total += dx * scale
        drow_total += dy * scale

    # Compute final second_peak_ratio at full resolution for reporting
    patch_s_f, _ = _extract_patch(img_src, src_col, src_row, half)
    patch_r_f, _ = _extract_patch(img_ref, ref_col + dcol_total, ref_row + drow_total, half)
    win_f = _make_2d_window(patch_s_f.shape[0], apodization)
    corr_f = _ncc_correlation(patch_s_f * win_f, patch_r_f * win_f)
    r_peak, c_peak = np.unravel_index(np.argmax(corr_f), corr_f.shape)
    masked = corr_f.copy()
    masked[max(0, r_peak-2):r_peak+3, max(0, c_peak-2):c_peak+3] = 0
    primary = float(corr_f[r_peak, c_peak])
    spr = float(masked.max()) / (primary + 1e-12)
    _, _, final_sharpness = _paraboloid_peak(corr_f)

    return dcol_total, drow_total, final_sharpness, spr, True


# ---------------------------------------------------------------------------
# Per-match refinement result
# ---------------------------------------------------------------------------

@dataclass
class RefineMatch:
    id: int
    src_xy_coarse: Tuple[float, float]     # original (col, row)
    ref_xy_coarse: Tuple[float, float]     # original (col, row)
    ref_xy_refined: Tuple[float, float]    # after refinement (col, row)
    refined_delta: Tuple[float, float]     # (dcol, drow) shift applied
    refine_sharpness: float
    second_peak_ratio: float
    refine_success: bool


@dataclass
class RefinementResult:
    matches: List[RefineMatch] = field(default_factory=list)
    rmse_before_px: float = 0.0
    rmse_after_px: float = 0.0
    refinement_gain_px: float = 0.0
    success_count: int = 0
    total_count: int = 0
    success_rate: float = 0.0
    partial_refinement: bool = False
    runtime_s: float = 0.0


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def refine_inliers(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    src_xy: np.ndarray,
    ref_xy_coarse: np.ndarray,
    gt_ref_xy: np.ndarray | None = None,
    window_px: int = WINDOW_PX,
    pyramid_levels: int = PYRAMID_LEVELS,
    apodization: str = APODIZATION,
    sharpness_threshold: float = SHARPNESS_THRESHOLD,
    second_peak_ratio_thresh: float = SECOND_PEAK_RATIO,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> RefinementResult:
    """
    Sub-pixel refine each inlier match using local NCC + paraboloid fitting.

    Parameters
    ----------
    img_src : (H, W) or (H, W, C) — source image (float or uint8)
    img_ref : (H, W) or (H, W, C) — reference image
    src_xy  : (N, 2) float — source coords (col, row)
    ref_xy_coarse : (N, 2) float — coarse reference coords to refine
    gt_ref_xy : (N, 2) float or None — GT for RMSE before/after computation
    window_px : patch half-size in pixels (full patch = 2*window_px)
    pyramid_levels : number of pyramid levels (coarse → fine)
    apodization : 'tukey' or 'gaussian' (NEVER 'blackman')
    sharpness_threshold : tau_q (TUNE on pilot data)
    second_peak_ratio_thresh : reject if 2nd peak > this × primary
    variance_threshold : tau_v — reject flat patches

    Returns
    -------
    RefinementResult with per-match details and RMSE before/after
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy_coarse.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert apodization.lower() != "blackman", \
        "Blackman apodization is FORBIDDEN — use 'tukey' or 'gaussian'"

    t0 = time.time()
    n = len(src_xy)
    half = window_px // 2

    # Ensure float32 greyscale
    def to_grey_f32(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if img.ndim == 3:
            img = img.mean(axis=2)
        return img

    src_f = to_grey_f32(img_src)
    ref_f = to_grey_f32(img_ref)

    matches: List[RefineMatch] = []
    refined_ref_xy = ref_xy_coarse.copy().astype(np.float64)

    for i in range(n):
        sc, sr = float(src_xy[i, 0]), float(src_xy[i, 1])
        rc, rr = float(ref_xy_coarse[i, 0]), float(ref_xy_coarse[i, 1])

        dcol, drow, sharpness, spr, success = _coarse_to_fine_refine(
            src_f, ref_f, sc, sr, rc, rr,
            half=half,
            levels=pyramid_levels,
            apodization=apodization,
            sharpness_threshold=sharpness_threshold,
            second_peak_ratio=second_peak_ratio_thresh,
            variance_threshold=variance_threshold,
        )

        if success:
            refined_ref_xy[i, 0] = rc + dcol
            refined_ref_xy[i, 1] = rr + drow

        matches.append(RefineMatch(
            id=i,
            src_xy_coarse=(sc, sr),
            ref_xy_coarse=(rc, rr),
            ref_xy_refined=(refined_ref_xy[i, 0], refined_ref_xy[i, 1]),
            refined_delta=(dcol if success else 0.0, drow if success else 0.0),
            refine_sharpness=sharpness,
            second_peak_ratio=spr,
            refine_success=success,
        ))

    success_count = sum(1 for m in matches if m.refine_success)
    success_rate = success_count / n if n > 0 else 0.0
    partial_refinement = success_rate < 0.70

    if partial_refinement:
        logger.warning(
            "Partial refinement: only %.1f%% of inliers refined (< 70%% threshold)",
            success_rate * 100,
        )

    # RMSE before/after (using GT if available, else coarse vs refined)
    rmse_before = rmse_after = gain = 0.0
    if gt_ref_xy is not None and len(gt_ref_xy) == n:
        res_before = np.linalg.norm(ref_xy_coarse - gt_ref_xy, axis=1)
        res_after  = np.linalg.norm(refined_ref_xy - gt_ref_xy, axis=1)
        rmse_before = float(np.sqrt(np.mean(res_before ** 2)))
        rmse_after  = float(np.sqrt(np.mean(res_after ** 2)))
        gain = rmse_before - rmse_after
    else:
        # Report displacement magnitude as proxy
        deltas = np.array([
            (m.refined_delta[0] ** 2 + m.refined_delta[1] ** 2) ** 0.5
            for m in matches if m.refine_success
        ])
        rmse_before = float(np.sqrt(np.mean(deltas ** 2))) if len(deltas) > 0 else 0.0

    runtime = time.time() - t0
    logger.info(
        "Refinement: %d/%d success (%.1f%%), RMSE before=%.3f after=%.3f gain=%.3f, time=%.1fs",
        success_count, n, success_rate * 100,
        rmse_before, rmse_after, gain, runtime,
    )

    return RefinementResult(
        matches=matches,
        rmse_before_px=rmse_before,
        rmse_after_px=rmse_after,
        refinement_gain_px=gain,
        success_count=success_count,
        total_count=n,
        success_rate=success_rate,
        partial_refinement=partial_refinement,
        runtime_s=runtime,
    )

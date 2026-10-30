"""
src/preprocessing/normalize.py
================================
F05 — Radiometric Normalization for lunar image pairs.

Two-stage normalization pipeline:
  1. Percentile clipping — removes sensor noise extremes (P2–P98 by default)
  2. Statistical transfer — aligns mean and std of source to reference

Both functions operate on float32 arrays in the range [0, 1].  Inputs of
any numeric dtype are accepted and converted internally.

References:
  - FEATURES.md F05 (Radiometric Normalization)
  - CONFIGURATION.md §3 (radiometric_norm block)
  - PROGRESS.md §2.2
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def percentile_clip(
    image: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
) -> np.ndarray:
    """
    Clip image intensities to [P_lo, P_hi] and rescale to float32 [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image (any numeric dtype, any shape).
    lo : float
        Lower percentile for clipping (default 2).
    hi : float
        Upper percentile for clipping (default 98).

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1] with the same spatial shape as *image*.
    """
    img = image.astype(np.float32)
    p_lo = float(np.percentile(img, lo))
    p_hi = float(np.percentile(img, hi))

    if p_hi <= p_lo:
        logger.warning(
            "percentile_clip: P%.0f (%.4f) >= P%.0f (%.4f) — returning zeros",
            lo, p_lo, hi, p_hi,
        )
        return np.zeros_like(img, dtype=np.float32)

    clipped = np.clip(img, p_lo, p_hi)
    rescaled = (clipped - p_lo) / (p_hi - p_lo)
    result = rescaled.astype(np.float32)

    logger.debug(
        "percentile_clip: P%.0f=%.4f P%.0f=%.4f → output [%.4f, %.4f]",
        lo, p_lo, hi, p_hi, float(result.min()), float(result.max()),
    )
    return result


def stat_transfer(
    src: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """
    Transfer the mean and standard deviation of *ref* onto *src*.

    Formula:
        out = (src - src.mean()) / src.std() * ref.std() + ref.mean()

    The output is clipped to [0, 1].  If *src* has zero standard deviation
    (flat image), the source is returned unchanged after clipping.

    Parameters
    ----------
    src : np.ndarray
        Source image to transform (float32 preferred, any shape).
    ref : np.ndarray
        Reference image whose statistics are the target.

    Returns
    -------
    np.ndarray
        Float32 array in [0, 1] with the same shape as *src*.
    """
    src_f = src.astype(np.float32)
    ref_f = ref.astype(np.float32)

    src_mean = float(np.mean(src_f))
    src_std = float(np.std(src_f))
    ref_mean = float(np.mean(ref_f))
    ref_std = float(np.std(ref_f))

    if src_std < 1e-8:
        logger.warning(
            "stat_transfer: source std ≈ 0 (flat image) — returning clipped source"
        )
        return np.clip(src_f, 0.0, 1.0).astype(np.float32)

    transferred = (src_f - src_mean) / src_std * ref_std + ref_mean
    result = np.clip(transferred, 0.0, 1.0).astype(np.float32)

    logger.debug(
        "stat_transfer: src μ=%.4f σ=%.4f → ref μ=%.4f σ=%.4f | out μ=%.4f σ=%.4f",
        src_mean, src_std, ref_mean, ref_std,
        float(np.mean(result)), float(np.std(result)),
    )
    return result

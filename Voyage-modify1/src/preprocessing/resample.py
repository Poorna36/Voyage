"""
src/preprocessing/resample.py
==============================
F07 — GSD Reconciliation for mismatched-resolution image pairs.

Core rules (from FEATURES.md F07 and CONFIGURATION.md §3):
  - ONLY the coarser-GSD image is resampled.  The higher-GSD (reference)
    image is NEVER touched.
  - Interpolation method depends on solar incidence angle:
      * bilinear  when solar_incidence >= low_angle_threshold_deg (high shadow, default 45°)
      * bicubic   when solar_incidence <  low_angle_threshold_deg (crisp detail, default 45°)
  - GSD ratio and interpolation method are recorded in the returned metadata dict.

References:
  - FEATURES.md F07 (GSD Reconciliation)
  - CONFIGURATION.md §3 (gsd block)
  - PROGRESS.md §2.4
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenCV interpolation flag mapping
_INTERP_MAP = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "lanczos": cv2.INTER_LANCZOS4,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Maximum output dimension per side (prevents OOM on extreme GSD ratios).
# At 4096×4096 float32 = 64MB, safely within typical 8GB RAM budgets.
MAX_OUTPUT_PX: int = 4096


def reconcile_gsd(
    src: np.ndarray,
    src_gsd: float,
    ref_gsd: float,
    solar_incidence_deg: float,
    low_angle_threshold_deg: float = 45.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resample the coarser-resolution image to match the finer-resolution image.

    Only the image with the LARGER GSD value is resampled.  The finer-GSD
    image is always returned unchanged.

    Parameters
    ----------
    src : np.ndarray
        Source (OHRC / TMC-2) image, 2-D float32 in [0, 1].
    src_gsd : float
        Ground sampling distance of *src* in metres/pixel (e.g. 0.31 for OHRC).
    ref_gsd : float
        Ground sampling distance of the reference image in metres/pixel
        (e.g. 0.50 for LRO NAC).
    solar_incidence_deg : float
        Solar incidence angle of the acquisition in degrees.
    low_angle_threshold_deg : float
        Threshold separating bilinear (high incidence) from bicubic (low
        incidence) interpolation (default 45°, from CONFIGURATION.md §3).

    Returns
    -------
    resampled : np.ndarray
        The coarser image resampled to the finer image's scale.
        If both GSD values are equal no resampling is performed.
    meta : dict
        Provenance metadata with keys:
          - ``gsd_ratio``            float   coarser / finer
          - ``interpolation_method`` str     "bilinear" or "bicubic"
          - ``which_resampled``      str     "src" | "ref" | "none"
          - ``src_gsd_m``            float
          - ``ref_gsd_m``            float
          - ``solar_incidence_deg``  float
    """
    if src_gsd <= 0 or ref_gsd <= 0:
        raise ValueError(f"GSD values must be positive. Got src={src_gsd}, ref={ref_gsd}")

    # Select interpolation method based on solar angle
    if solar_incidence_deg >= low_angle_threshold_deg:
        interp_name = "bilinear"
    else:
        interp_name = "bicubic"
    interp_flag = _INTERP_MAP[interp_name]

    meta: Dict[str, Any] = {
        "src_gsd_m": src_gsd,
        "ref_gsd_m": ref_gsd,
        "solar_incidence_deg": solar_incidence_deg,
        "interpolation_method": interp_name,
    }

    # No resampling needed if GSD values are equal
    if abs(src_gsd - ref_gsd) < 1e-9:
        meta["gsd_ratio"] = 1.0
        meta["which_resampled"] = "none"
        logger.debug("reconcile_gsd: GSD equal (%.4f), no resampling.", src_gsd)
        return src.copy(), meta

    if src_gsd > ref_gsd:
        # Source is coarser — downsample src to ref resolution
        # scale_factor < 1 (shrink) — src covers more ground per pixel
        # We want src to represent the same spatial extent at finer GSD
        gsd_ratio = src_gsd / ref_gsd
        # New shape: src spatial coverage / ref_gsd
        # In pixel terms: src.shape * (src_gsd / ref_gsd) = src.shape * gsd_ratio
        new_h = max(1, int(round(src.shape[0] * gsd_ratio)))
        new_w = max(1, int(round(src.shape[1] * gsd_ratio)))

        # Guard: cap output size to prevent OOM on extreme GSD ratios
        if new_h > MAX_OUTPUT_PX or new_w > MAX_OUTPUT_PX:
            scale_cap = min(MAX_OUTPUT_PX / new_h, MAX_OUTPUT_PX / new_w)
            new_h = max(1, int(new_h * scale_cap))
            new_w = max(1, int(new_w * scale_cap))
            logger.warning(
                "reconcile_gsd: output capped to %dx%d (gsd_ratio=%.3f would produce "
                "image exceeding MAX_OUTPUT_PX=%d — this GSD ratio is unrealistically large "
                "for real sensor pairs).",
                new_h, new_w, gsd_ratio, MAX_OUTPUT_PX,
            )
        resampled = cv2.resize(
            src.astype(np.float32), (new_w, new_h), interpolation=interp_flag
        )
        meta["gsd_ratio"] = gsd_ratio
        meta["which_resampled"] = "src"
        logger.info(
            "reconcile_gsd: src resampled (%.2f→%.2f m/px, ratio=%.3f) "
            "shape %s → %s, interp=%s",
            src_gsd, ref_gsd, gsd_ratio, src.shape, resampled.shape, interp_name,
        )
    else:
        # Reference is coarser — but we NEVER resample the reference here.
        # The caller (preprocess.py) should handle the ref-is-coarser case.
        # We log a warning and return src unchanged.
        gsd_ratio = ref_gsd / src_gsd
        meta["gsd_ratio"] = gsd_ratio
        meta["which_resampled"] = "ref"
        logger.warning(
            "reconcile_gsd: ref is coarser (ref=%.2f, src=%.2f). "
            "The reference image must be resampled by the caller — "
            "returning src unchanged.",
            ref_gsd, src_gsd,
        )
        resampled = src.copy()

    return resampled.astype(np.float32), meta

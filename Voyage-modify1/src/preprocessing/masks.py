"""
src/preprocessing/masks.py
==========================
F04 — Shadow and Invalid-Pixel Masking for lunar imagery.

Implements a three-stage shadow detection pipeline suitable for OHRC
(0.31 m/px) and TMC-2 (5 m/px) imagery of the lunar surface:

  1. Dark-pixel test      — absolute darkness relative to global statistics
  2. Flat-variance test   — textureless dark regions (cast shadows)
  3. Incidence-angle test — high solar incidence flag (> threshold_deg)

Parameters are drawn from configs/ohrc_nac.yaml and configs/tmc_wac.yaml
under the ``preprocessing.shadow_mask`` key.

References:
  - FEATURES.md F04 (Shadow Masking)
  - CONFIGURATION.md §3 (L1 Preprocessing parameters)
  - PROGRESS.md §2.1
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shadow_mask(
    image: np.ndarray,
    solar_incidence_deg: float,
    incidence_threshold_deg: float = 80.0,
    local_variance_window: int = 15,
    flat_variance_threshold: float = 10.0,
    dark_k: float = 2.0,
    dilate_boundary_px: int = 0,
) -> np.ndarray:
    """
    Compute a boolean invalid-pixel mask for a single-band lunar image.

    Pixels are flagged as invalid (True) when they satisfy ANY of:
      - Dark-pixel test: intensity < mean - dark_k * std
      - Flat-variance test: local variance < flat_variance_threshold
                            AND pixel is below global median (cast shadow)
      - Incidence-angle test: solar_incidence_deg > incidence_threshold_deg
                              AND pixel is in the dark half of the histogram

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image, any numeric dtype.  Will be converted to
        float32 internally.
    solar_incidence_deg : float
        Solar incidence angle of the acquisition in degrees.
    incidence_threshold_deg : float
        Threshold above which the incidence-angle test is applied (default 80°).
    local_variance_window : int
        Side length of the square neighbourhood for local variance (odd, default 15).
    flat_variance_threshold : float
        Local variance below this is considered "flat" (default 10.0).
    dark_k : float
        Number of standard deviations below mean to classify as dark (default 2.0).

    Returns
    -------
    np.ndarray
        Boolean mask of the same spatial shape as *image*.
        True  → pixel is invalid (shadowed / saturated flat region).
        False → pixel is valid.
    """
    if image.ndim != 2:
        raise ValueError(f"shadow_mask expects a 2-D image, got shape {image.shape}")

    img = image.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Stage 1 — Dark-pixel test
    # ------------------------------------------------------------------ #
    mu = float(np.mean(img))
    sigma = float(np.std(img))
    dark_threshold = mu - dark_k * sigma
    dark_mask = img < dark_threshold

    # ------------------------------------------------------------------ #
    # Stage 2 — Flat-variance test (cast shadow: textureless dark region)
    # ------------------------------------------------------------------ #
    # Ensure window size is odd
    win = local_variance_window if local_variance_window % 2 == 1 else local_variance_window + 1

    # Local mean via box filter
    img_u8 = np.clip(img, 0, 255).astype(np.float32)
    local_mean = cv2.boxFilter(img_u8, ddepth=-1, ksize=(win, win))
    # Local variance = E[X²] - E[X]²
    local_sq_mean = cv2.boxFilter(img_u8 ** 2, ddepth=-1, ksize=(win, win))
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)

    median_val = float(np.median(img))
    flat_mask = (local_var < flat_variance_threshold) & (img < median_val)

    # ------------------------------------------------------------------ #
    # Stage 3 — Incidence-angle test (high sun-angle scenes)
    # ------------------------------------------------------------------ #
    incidence_mask = np.zeros_like(dark_mask)
    if solar_incidence_deg > incidence_threshold_deg:
        # In high-incidence (grazing light) scenes, extra dark pixels are
        # classified invalid — use a tighter dark threshold (1 sigma)
        tight_threshold = mu - 1.0 * sigma
        incidence_mask = img < tight_threshold

    # ------------------------------------------------------------------ #
    # Combine: any test fires → pixel is invalid
    # ------------------------------------------------------------------ #
    combined = dark_mask | flat_mask | incidence_mask

    # ------------------------------------------------------------------ #
    # Stage 4 — Optional penumbra boundary dilation
    # ------------------------------------------------------------------ #
    if dilate_boundary_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_boundary_px + 1, 2 * dilate_boundary_px + 1),
        )
        combined = cv2.dilate(combined.astype(np.uint8), kernel).astype(bool)

    logger.debug(
        "shadow_mask: dark=%.1f%% flat=%.1f%% incidence=%.1f%% total=%.1f%%",
        100.0 * dark_mask.mean(),
        100.0 * flat_mask.mean(),
        100.0 * incidence_mask.mean(),
        100.0 * combined.mean(),
    )
    return combined


def check_mask_fraction(
    mask: np.ndarray,
    min_pct: float = 5.0,
    max_pct: float = 30.0,
) -> Tuple[float, bool]:
    """
    Check whether the fraction of masked pixels falls within the expected range.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask (True = invalid) from :func:`shadow_mask`.
    min_pct : float
        Minimum acceptable mask percentage (default 5%).
    max_pct : float
        Maximum acceptable mask percentage (default 30%).

    Returns
    -------
    (fraction_masked, in_range)
        fraction_masked : float in [0, 1]
        in_range : bool — True if (min_pct/100) <= fraction_masked <= (max_pct/100)
    """
    fraction = float(np.mean(mask.astype(np.float32)))
    lo = min_pct / 100.0
    hi = max_pct / 100.0
    in_range = lo <= fraction <= hi
    logger.debug(
        "check_mask_fraction: fraction=%.3f in_range=%s (%.1f%%–%.1f%%)",
        fraction, in_range, min_pct, max_pct,
    )
    return fraction, in_range


def save_mask_png(mask: np.ndarray, out_path: Path) -> Path:
    """
    Save boolean mask as an 8-bit PNG.

    White (255) = valid pixel.
    Black (0)   = masked / invalid pixel.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask (True = invalid).
    out_path : Path
        Destination file path (will be created with parents).

    Returns
    -------
    Path
        Absolute path to the written PNG file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Valid pixels → white; masked pixels → black
    png = np.where(mask, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), png)
    logger.info("Saved valid_mask.png → %s", out_path)
    return out_path

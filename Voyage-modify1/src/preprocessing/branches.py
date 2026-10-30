"""
src/preprocessing/branches.py
==============================
F06 — Sensor-Pair-Specific Preprocessing Branches.

Three branches are defined, following FEATURES.md F06 and ARCHITECTURE.md L1:

  ohrc_to_nac  (OHRC vs LRO NAC)
      CLAHE → optional inversion → morphological dilation → PCA whitening

  tmc_to_wac   (TMC-2 vs LRO WAC 643nm)  — EXPERIMENTAL
      histogram matching → CLAHE
      A/B test: also returns minimal branch output for comparison

  minimal
      percentile clip ONLY — mandatory for M2 (LightGlue) and M3 (Crater/CNSFM).
      Heavy preprocessing must NEVER be applied to learned matchers.

Branch selection logic (`select_branch`) reads the config ``sensor_branch``
key for classical matchers, and forces ``minimal`` for learned matchers
regardless of sensor pair.

References:
  - FEATURES.md F06 (Sensor-Pair Preprocessing Branches)
  - ARCHITECTURE.md §3 L1 layer
  - CONFIGURATION.md §3 (sensor_branch, ohrc_to_nac, tmc_to_wac blocks)
  - PROGRESS.md §2.3
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Matcher IDs that must ALWAYS receive the minimal branch (no heavy processing)
_LEARNED_MATCHERS = frozenset({"lightglue", "crater", "crater_hough"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize float32 [0,1] to uint8 [0,255] for OpenCV operations."""
    img = np.clip(image, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _to_float32(image: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] back to float32 [0,1]."""
    return image.astype(np.float32) / 255.0


def _is_inverted(image: np.ndarray) -> bool:
    """
    Heuristic: image is 'bright-on-dark' (needs inversion) when the mean
    of the upper quartile is much larger than the mean of the lower quartile.
    Return True when the histogram is right-skewed (most mass in bright half).
    """
    flat = image.flatten()
    med = float(np.median(flat))
    upper_mean = float(np.mean(flat[flat > med]))
    lower_mean = float(np.mean(flat[flat <= med]))
    # If upper half is > 3x the lower half, image is bright-dominant → invert
    if lower_mean < 1e-6:
        return False
    return (upper_mean / lower_mean) > 3.0


def _clahe(image_u8: np.ndarray, clip_limit: float, tile_grid: tuple) -> np.ndarray:
    """Apply CLAHE to a uint8 image. Returns uint8."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
    )
    return clahe.apply(image_u8)


def _pca_whiten(image: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    PCA whitening for a 2-D float32 image.

    Flattens the image to a 1-D vector, applies PCA (n_components components),
    whitens, then reshapes back to the original spatial dimensions.

    For n_components=1 (typical use) this is equivalent to projecting onto
    the principal axis and rescaling — effectively a global contrast normalizer.
    """
    h, w = image.shape
    flat = image.flatten().reshape(-1, 1).astype(np.float32)

    # PCA via SVD on centered data
    mu = np.mean(flat, axis=0)
    centered = flat - mu
    # For a 1-D input, SVD is trivially one component
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto first component
    projected = (centered @ Vt[:n_components].T)  # (N, n_components)
    # Normalize to [0,1]
    mn, mx = projected.min(), projected.max()
    if mx - mn < 1e-8:
        return image
    whitened = ((projected - mn) / (mx - mn)).reshape(h, w)
    return whitened.astype(np.float32)


# ---------------------------------------------------------------------------
# Public branch functions
# ---------------------------------------------------------------------------

def apply_ohrc_nac(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Apply the OHRC→NAC preprocessing branch.

    Pipeline:
      1. Convert to uint8
      2. CLAHE (clip_limit and tile_grid from config)
      3. Optional auto-inversion (if 'inversion' == 'auto' and image is bright-on-dark)
      4. Morphological dilation (3×3 kernel, 1 iteration) to fill small gaps
      5. PCA whitening (1 component) for global contrast normalization
      6. Return float32 [0, 1]

    Parameters
    ----------
    image : np.ndarray
        Float32 [0, 1] single-band image.
    config : dict
        Sub-config from ``preprocessing.ohrc_to_nac`` block, e.g.:
        {clahe_clip_limit: 2.0, clahe_tile_grid: [8,8], pca_components: 1,
         inversion: "auto"}

    Returns
    -------
    np.ndarray
        Processed float32 [0, 1] image.
    """
    clip_limit = float(config.get("clahe_clip_limit", 2.0))
    tile_grid = config.get("clahe_tile_grid", [8, 8])
    pca_components = int(config.get("pca_components", 1))
    inversion = config.get("inversion", "auto")

    img_u8 = _to_uint8(image)

    # 1. CLAHE
    img_u8 = _clahe(img_u8, clip_limit, tile_grid)

    # 2. Optional inversion
    img_f = _to_float32(img_u8)
    if inversion == "auto":
        if _is_inverted(img_f):
            logger.debug("apply_ohrc_nac: inverting bright-on-dark image")
            img_f = 1.0 - img_f
    elif inversion is True or inversion == "always":
        img_f = 1.0 - img_f
    img_u8 = _to_uint8(img_f)

    # 3. Morphological dilation (fill small shadow gaps)
    kernel = np.ones((3, 3), dtype=np.uint8)
    img_u8 = cv2.dilate(img_u8, kernel, iterations=1)

    # 4. PCA whitening
    img_f = _to_float32(img_u8)
    result = _pca_whiten(img_f, n_components=pca_components)

    logger.debug("apply_ohrc_nac: output range [%.4f, %.4f]", result.min(), result.max())
    return result.astype(np.float32)


def apply_tmc_wac(
    image: np.ndarray,
    ref: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Apply the TMC-2→WAC preprocessing branch (EXPERIMENTAL).

    Pipeline:
      1. Histogram matching of *image* to *ref* distribution (skimage)
      2. CLAHE (clip_limit from config)
      3. Return float32 [0, 1]

    This branch is experimental and not confirmed by peer review.
    The caller (preprocess.py) records ``branch_experimental=true`` in meta.json.

    Parameters
    ----------
    image : np.ndarray
        Float32 [0, 1] TMC-2 source image.
    ref : np.ndarray
        Float32 [0, 1] WAC reference image (used as histogram template).
    config : dict
        Sub-config from ``preprocessing.tmc_to_wac`` block.

    Returns
    -------
    np.ndarray
        Processed float32 [0, 1] image.
    """
    try:
        from skimage.exposure import match_histograms  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for apply_tmc_wac. "
            "Install with: pip install scikit-image"
        ) from exc

    clip_limit = float(config.get("clahe_clip_limit", 1.5))
    tile_grid = config.get("clahe_tile_grid", [8, 8])
    do_hist_match = bool(config.get("histogram_match", True))

    img = image.astype(np.float32)

    # 1. Histogram matching
    if do_hist_match:
        ref_f = ref.astype(np.float32)
        matched = match_histograms(img, ref_f)
        img = np.clip(matched, 0.0, 1.0).astype(np.float32)
        logger.debug("apply_tmc_wac: histogram matching applied")

    # 2. CLAHE
    img_u8 = _to_uint8(img)
    img_u8 = _clahe(img_u8, clip_limit, tile_grid)
    result = _to_float32(img_u8)

    logger.debug("apply_tmc_wac: output range [%.4f, %.4f]", result.min(), result.max())
    return result


def apply_minimal(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Apply the minimal preprocessing branch (percentile clip only).

    This is the MANDATORY branch for learned matchers (M2 LightGlue, M3 Crater).
    Heavy preprocessing (CLAHE, PCA, inversion) must NEVER be applied to
    images destined for learned matchers.

    Parameters
    ----------
    image : np.ndarray
        Float32 [0, 1] image (already percentile-clipped at L1 entry).
    config : dict
        Unused — kept for API consistency.

    Returns
    -------
    np.ndarray
        The input image unchanged (already clipped by preprocess.py).
    """
    # Percentile clipping was already applied in preprocess.py before branch selection.
    # This branch is a no-op — it simply enforces the "no heavy processing" contract.
    result = np.clip(image, 0.0, 1.0).astype(np.float32)
    logger.debug("apply_minimal: returning percentile-clipped image as-is")
    return result


def select_branch(
    sensor_pair: str,
    matcher_id: str,
    config: Dict[str, Any],
) -> str:
    """
    Determine which preprocessing branch to apply for a given sensor pair and matcher.

    Rules (from FEATURES.md F06):
      - Learned matchers (lightglue, crater, crater_hough) → always "minimal"
      - OHRC-NAC + classical matcher → "ohrc_to_nac"
      - TMC-2-WAC + classical matcher → "tmc_to_wac"
      - Unknown / default → "minimal" (safe fallback)

    Parameters
    ----------
    sensor_pair : str
        Sensor pair identifier, e.g. "OHRC-NAC", "TMC-2-WAC", "IIRS-WAC".
    matcher_id : str
        Matcher identifier, e.g. "sift", "rift2", "lnift", "lightglue", "crater".
    config : dict
        Full config dict (used to read ``preprocessing.sensor_branch`` override).

    Returns
    -------
    str
        Branch name: one of "ohrc_to_nac", "tmc_to_wac", "minimal".
    """
    # Learned matchers always get minimal branch — no override allowed
    if matcher_id.lower() in _LEARNED_MATCHERS:
        logger.debug(
            "select_branch: matcher_id=%s is a learned matcher → minimal", matcher_id
        )
        return "minimal"

    # Check for explicit config override
    preproc_cfg = config.get("preprocessing", {})
    branch_override = preproc_cfg.get("sensor_branch", "").lower()
    if branch_override in ("ohrc_to_nac", "tmc_to_wac", "minimal"):
        logger.debug("select_branch: config override → %s", branch_override)
        return branch_override

    # Auto-select based on sensor pair
    sp = sensor_pair.upper().replace(" ", "-")
    if "OHRC" in sp and "NAC" in sp:
        return "ohrc_to_nac"
    if "TMC" in sp and "WAC" in sp:
        return "tmc_to_wac"
    if "IIRS" in sp:
        return "minimal"  # IIRS uses its own separate pipeline

    logger.warning(
        "select_branch: unknown sensor_pair=%r, defaulting to 'minimal'", sensor_pair
    )
    return "minimal"

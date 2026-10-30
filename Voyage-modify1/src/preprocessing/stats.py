"""
src/preprocessing/stats.py
===========================
Image texture and gradient statistics computation for L1 preprocessing (S3)
and Matcher Selection Model (MSM) feature extraction (L1.5 / S4.5).

Provides:
  - compute_texture_contrast: Mean local standard deviation in NxN sliding windows.
  - compute_mean_gradient: Mean Sobel gradient magnitude across the image.
  - compute_image_stats: Combined convenience function for image metrics.

References:
  - FEATURES.md F26
  - ARCHITECTURE.md §3 (L1.5)
  - PROGRESS.md §5.5.1
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import scipy.ndimage as ndimage
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def compute_texture_contrast(
    image: np.ndarray,
    window_size: int = 8,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean local standard deviation in sliding windows of size (window_size x window_size).

    Formula:
        Var(I) = E[I^2] - (E[I])^2
        Std(I) = sqrt(max(0, Var(I)))
        Contrast = mean(Std(I))

    Parameters
    ----------
    image : np.ndarray
        2-D image array (float or integer).
    window_size : int, default=8
        Sliding window size in pixels.
    valid_mask : np.ndarray, optional
        Boolean mask where True indicates valid (unmasked) pixels.

    Returns
    -------
    float
        Mean local standard deviation in DN. Returns 0.0 for constant or empty image.
    """
    if image.size == 0:
        return 0.0

    img = np.squeeze(image).astype(np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {img.shape}")

    h, w = img.shape
    if h < window_size or w < window_size:
        return float(np.std(img))

    if _CV2_AVAILABLE:
        ksize = (window_size, window_size)
        mean = cv2.blur(img, ksize)
        mean_sq = cv2.blur(img * img, ksize)
        var = np.maximum(0.0, mean_sq - (mean * mean))
        local_std = np.sqrt(var)
    elif _SCIPY_AVAILABLE:
        mean = ndimage.uniform_filter(img, size=window_size)
        mean_sq = ndimage.uniform_filter(img * img, size=window_size)
        var = np.maximum(0.0, mean_sq - (mean * mean))
        local_std = np.sqrt(var)
    else:
        # Pure numpy block approximation fallback
        pad_h = (window_size - (h % window_size)) % window_size
        pad_w = (window_size - (w % window_size)) % window_size
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        bh, bw = padded.shape[0] // window_size, padded.shape[1] // window_size
        blocks = padded.reshape(bh, window_size, bw, window_size).swapaxes(1, 2)
        local_std = np.std(blocks, axis=(-2, -1))

    if valid_mask is not None:
        mask_2d = np.squeeze(valid_mask).astype(bool)
        if mask_2d.shape == local_std.shape and np.any(mask_2d):
            return float(np.nanmean(local_std[mask_2d]))

    return float(np.nanmean(local_std))


def compute_mean_gradient(
    image: np.ndarray,
    ksize: int = 3,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean Sobel gradient magnitude across the image.

    Formula:
        G = sqrt(G_x^2 + G_y^2)
        Mean Gradient = mean(G)

    Parameters
    ----------
    image : np.ndarray
        2-D image array (float or integer).
    ksize : int, default=3
        Sobel kernel aperture size (must be 1, 3, 5, or 7).
    valid_mask : np.ndarray, optional
        Boolean mask where True indicates valid (unmasked) pixels.

    Returns
    -------
    float
        Mean gradient magnitude in DN/px. Returns 0.0 for constant or empty image.
    """
    if image.size == 0:
        return 0.0

    img = np.squeeze(image).astype(np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {img.shape}")

    h, w = img.shape
    if h < 3 or w < 3:
        return 0.0

    if _CV2_AVAILABLE:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = np.sqrt(gx * gx + gy * gy)
    elif _SCIPY_AVAILABLE:
        gx = ndimage.sobel(img, axis=1)
        gy = ndimage.sobel(img, axis=0)
        mag = np.sqrt(gx * gx + gy * gy)
    else:
        # Basic discrete difference fallback
        gy, gx = np.gradient(img)
        mag = np.sqrt(gx * gx + gy * gy)

    if valid_mask is not None:
        mask_2d = np.squeeze(valid_mask).astype(bool)
        if mask_2d.shape == mag.shape and np.any(mask_2d):
            return float(np.nanmean(mag[mask_2d]))

    return float(np.nanmean(mag))


def compute_image_stats(
    image: np.ndarray,
    window_size: int = 8,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute image texture contrast and mean gradient statistics.

    Parameters
    ----------
    image : np.ndarray
        2-D image array.
    window_size : int, default=8
        Window size for texture contrast.
    valid_mask : np.ndarray, optional
        Validity mask.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'texture_contrast' and 'mean_gradient'.
    """
    return {
        "texture_contrast": compute_texture_contrast(image, window_size=window_size, valid_mask=valid_mask),
        "mean_gradient": compute_mean_gradient(image, valid_mask=valid_mask),
    }

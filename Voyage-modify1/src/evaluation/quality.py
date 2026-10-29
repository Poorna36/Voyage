"""
src/evaluation/quality.py
===========================
Photometric Quality Verification Module.

Computes Structural Similarity Index (SSIM), Normalized Cross-Correlation (NCC),
and residual spatial error metrics between warped source and reference images.

Functions:
  compute_ssim(img1, img2, valid_mask=None) -> float
  compute_ncc(img1, img2, valid_mask=None) -> float
  generate_checkerboard(img1, img2, tile_size=64) -> np.ndarray
  generate_residual_heatmap(img1, img2, valid_mask=None) -> np.ndarray
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two single-band float32 [0,1] images.
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    c1 = (k1 * 1.0) ** 2
    c2 = (k2 * 1.0) ** 2

    # Gaussian blur kernel
    kernel = cv2.getGaussianKernel(win_size, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1.astype(np.float64), -1, window)
    mu2 = cv2.filter2D(img2.astype(np.float64), -1, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D((img1.astype(np.float64)) ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D((img2.astype(np.float64)) ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D((img1.astype(np.float64) * img2.astype(np.float64)), -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if valid_mask is not None:
        if valid_mask.dtype != bool:
            valid_mask = valid_mask.astype(bool)
        if valid_mask.sum() == 0:
            return 0.0
        return float(ssim_map[valid_mask].mean())

    return float(ssim_map.mean())


def compute_ncc(
    img1: np.ndarray,
    img2: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between two images in range [-1.0, 1.0].
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    f1 = img1.astype(np.float64)
    f2 = img2.astype(np.float64)

    if valid_mask is not None:
        mask = valid_mask.astype(bool)
        if mask.sum() == 0:
            return 0.0
        v1 = f1[mask]
        v2 = f2[mask]
    else:
        v1 = f1.flatten()
        v2 = f2.flatten()

    m1, m2 = v1.mean(), v2.mean()
    zero1 = v1 - m1
    zero2 = v2 - m2

    norm1 = np.linalg.norm(zero1)
    norm2 = np.linalg.norm(zero2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    ncc = np.dot(zero1, zero2) / (norm1 * norm2)
    return float(np.clip(ncc, -1.0, 1.0))


def generate_checkerboard(
    img1: np.ndarray,
    img2: np.ndarray,
    tile_size: int = 64,
) -> np.ndarray:
    """
    Generate an interleaved checkerboard image for visual verification of registration.
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    h, w = img1.shape

    # Normalize images to uint8
    def to_u8(arr):
        if arr.dtype == np.uint8:
            return arr
        norm = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr / 255.0, 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    u1 = to_u8(img1)
    u2 = to_u8(img2)

    checkerboard = u1.copy()
    num_tiles_y = (h + tile_size - 1) // tile_size
    num_tiles_x = (w + tile_size - 1) // tile_size

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            if (ty + tx) % 2 == 1:
                y0, y1 = ty * tile_size, min((ty + 1) * tile_size, h)
                x0, x1 = tx * tile_size, min((tx + 1) * tile_size, w)
                checkerboard[y0:y1, x0:x1] = u2[y0:y1, x0:x1]

    return checkerboard


def generate_residual_heatmap(
    img1: np.ndarray,
    img2: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate an absolute pixel error heatmap (RGB uint8) between two registered images.
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    f1 = np.clip(img1.astype(np.float32), 0.0, 1.0)
    f2 = np.clip(img2.astype(np.float32), 0.0, 1.0)

    diff = np.abs(f1 - f2)
    if valid_mask is not None:
        diff[~valid_mask.astype(bool)] = 0.0

    diff_u8 = (np.clip(diff * 255.0, 0, 255)).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)

    if valid_mask is not None:
        heatmap[~valid_mask.astype(bool)] = 0

    return heatmap

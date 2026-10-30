"""
synthetic/transforms.py — Physical Transformation Engine

Applies exact geometric and radiometric shifts to a source lunar image to
produce a synthetic target image, and simultaneously computes the exact
floating-point ground truth coordinates of all anchor points in the target.

Transformations are strictly derived from the SIH26166 problem statement:
  "Multi-modal, Sun-angle & scale-invariant image correspondence using
   Chandrayaan-2 (OHRC / TMC-2 / IIRS) vs LRO (NAC / WAC)"

Coordinate convention: all pixel coordinates are (col, row) = (x, y), 0-indexed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INTERP_MAP = {
    "lanczos": cv2.INTER_LANCZOS4,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


@dataclass
class TransformParams:
    """Parameters describing one synthetic transformation applied to a source image."""
    pair_id: str
    random_seed: int

    # Geometric
    scale_factor: float = 1.0
    rotation_deg: float = 0.0
    translation_px: List[float] = field(default_factory=lambda: [0.0, 0.0])
    interpolation: str = "lanczos"

    # Photometric
    illumination_gamma: float = 1.0
    shadow_extension_factor: float = 1.0
    mtf_blur_sigma: float = 0.0
    pushbroom_stripe_amplitude: float = 0.0

    # Derived / output
    transform_matrix: Optional[List[List[float]]] = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        if self.transform_matrix is not None:
            d["transform_matrix"] = [list(row) for row in self.transform_matrix]
        return d


def build_transform_matrix(
    image_shape: Tuple[int, int],
    scale_factor: float = 1.0,
    rotation_deg: float = 0.0,
    translation_px: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build the 3x3 homogeneous transformation matrix M."""
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    dx, dy = translation_px

    rot_scale = cv2.getRotationMatrix2D(center=(cx, cy), angle=rotation_deg, scale=scale_factor)
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = rot_scale.astype(np.float64)

    M[0, 2] += dx
    M[1, 2] += dy

    return M


def apply_transform(
    image: np.ndarray,
    M: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "lanczos",
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float = 0.0,
) -> np.ndarray:
    """Apply a 3x3 homogeneous transform matrix to an image using cv2.warpAffine."""
    assert image.ndim in (2, 3), "apply_transform expects 2D or 3D image array."
    h, w = image.shape[:2]
    out_h, out_w = output_shape if output_shape is not None else (h, w)
    interp_flag = _INTERP_MAP.get(interpolation, cv2.INTER_LANCZOS4)

    M_2x3 = M[:2, :].astype(np.float64)
    transformed = cv2.warpAffine(
        src=image,
        M=M_2x3,
        dsize=(out_w, out_h),
        flags=interp_flag,
        borderMode=border_mode,
        borderValue=border_value,
    )
    return transformed


def transform_gt_points(
    src_pts: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Apply exact transformation matrix M to GT anchor coordinates."""
    assert src_pts.ndim == 2 and src_pts.shape[-1] == 2, "Expected (N, 2) array: (col, row)"
    if src_pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    ones = np.ones((src_pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([src_pts.astype(np.float64), ones])
    tgt_h = (M @ pts_h.T).T
    tgt_pts = tgt_h[:, :2] / tgt_h[:, 2:3]
    return tgt_pts


def apply_illumination_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Non-linear photometric distortion simulating different solar incidence angles."""
    assert gamma > 0, f"Gamma must be positive, got {gamma}"
    img_clipped = np.clip(image.astype(np.float32), 0.0, 1.0)
    return np.power(img_clipped, gamma).astype(np.float32)


def apply_mtf_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Sensor-specific MTF blurring."""
    if sigma <= 0:
        return image
    k = max(3, int(np.ceil(sigma * 3)) * 2 + 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_pushbroom_noise(
    image: np.ndarray,
    n_stripes: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pushbroom sensor row-noise as slight vertical striping."""
    out = image.astype(np.float32).copy()
    h, w = out.shape[:2]
    if n_stripes <= 0 or amplitude <= 0:
        return out
    n_stripes = min(n_stripes, w)
    stripe_cols = rng.choice(w, size=n_stripes, replace=False)
    img_mean = float(out.mean()) + 1e-8
    for col in stripe_cols:
        gain_offset = rng.uniform(-amplitude * img_mean, amplitude * img_mean)
        out[:, col] = np.clip(out[:, col] + gain_offset, 0.0, 1.0)
    return out


def apply_shadow_extension(
    image: np.ndarray,
    extension_factor: float,
    solar_azimuth_deg: float = 270.0,
) -> np.ndarray:
    """Mask and extend existing shadow regions in the direction of the sun."""
    if extension_factor <= 1.0:
        return image
    out = image.astype(np.float32).copy()
    shadow_threshold = float(np.percentile(out, 10))
    shadow_mask = out < shadow_threshold

    az_rad = np.deg2rad(solar_azimuth_deg)
    shadow_dx = -np.sin(az_rad)
    shadow_dy = -np.cos(az_rad)

    extension_px = max(1, int((extension_factor - 1.0) * 20))
    kernel_size = 2 * extension_px + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    cx_k, cy_k = extension_px, extension_px
    for step in range(1, extension_px + 1):
        dx_i = int(round(shadow_dx * step))
        dy_i = int(round(shadow_dy * step))
        r = cy_k + dy_i
        c = cx_k + dx_i
        if 0 <= r < kernel_size and 0 <= c < kernel_size:
            kernel[r, c] = 1
    kernel[cy_k, cx_k] = 1

    dilated_mask = cv2.dilate(shadow_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    out[dilated_mask & ~shadow_mask] = shadow_threshold * 0.8
    return np.clip(out, 0.0, 1.0)


def generate_synthetic_pair(
    source_image: np.ndarray,
    config: dict,
    pair_id: str,
    seed: int,
) -> Tuple[np.ndarray, TransformParams, np.ndarray]:
    """Generate a synthetic target image from a source image with known exact GT transform."""
    assert source_image.ndim == 2, "generate_synthetic_pair expects a 2D single-channel image."
    rng = np.random.default_rng(seed)
    tf_cfg = config.get("transforms", {})

    scale_factor = 1.0
    rotation_deg = 0.0
    translation_px = [0.0, 0.0]

    if tf_cfg.get("scale", {}).get("enabled", False):
        s_min = tf_cfg["scale"].get("min_factor", 0.25)
        s_max = tf_cfg["scale"].get("max_factor", 4.0)
        scale_factor = float(rng.uniform(s_min, s_max))

    if tf_cfg.get("rotation", {}).get("enabled", False):
        max_angle = tf_cfg["rotation"].get("max_angle_deg", 5.0)
        rotation_deg = float(rng.uniform(-max_angle, max_angle))

    if tf_cfg.get("translation", {}).get("enabled", True):
        max_shift = tf_cfg["translation"].get("max_shift_px", 1.0)
        dx = float(rng.uniform(-max_shift, max_shift))
        dy = float(rng.uniform(-max_shift, max_shift))
        if abs(dx - round(dx)) < 0.05:
            dx += 0.1 * rng.choice([-1, 1])
        if abs(dy - round(dy)) < 0.05:
            dy += 0.1 * rng.choice([-1, 1])
        translation_px = [dx, dy]

    M = build_transform_matrix(
        image_shape=source_image.shape,
        scale_factor=scale_factor,
        rotation_deg=rotation_deg,
        translation_px=tuple(translation_px),
    )
    interpolation = tf_cfg.get("scale", {}).get("interpolation", "lanczos")

    synthetic = apply_transform(source_image, M, interpolation=interpolation)
    synthetic = np.clip(synthetic, 0.0, 1.0).astype(np.float32)

    gamma = 1.0
    if tf_cfg.get("illumination", {}).get("enabled", False):
        g_range = tf_cfg["illumination"].get("gamma_range", [0.7, 1.4])
        gamma = float(rng.uniform(g_range[0], g_range[1]))
        synthetic = apply_illumination_gamma(synthetic, gamma)

    shadow_ext = 1.0
    if (tf_cfg.get("illumination", {}).get("enabled", False) and
            tf_cfg["illumination"].get("shadow_extension", {}).get("enabled", False)):
        max_ext = tf_cfg["illumination"]["shadow_extension"].get("max_extension_factor", 1.5)
        shadow_ext = float(rng.uniform(1.0, max_ext))
        synthetic = apply_shadow_extension(synthetic, shadow_ext, 270.0)

    mtf_sigma = 0.0
    if tf_cfg.get("sensor_simulation", {}).get("enabled", False):
        mtf_cfg = tf_cfg["sensor_simulation"].get("mtf_blur", {})
        if mtf_cfg.get("enabled", False):
            s_range = mtf_cfg.get("sigma_range", [0.5, 1.5])
            mtf_sigma = float(rng.uniform(s_range[0], s_range[1]))
            synthetic = apply_mtf_blur(synthetic, mtf_sigma)

    pushbroom_amp = 0.0
    if tf_cfg.get("sensor_simulation", {}).get("enabled", False):
        pb_cfg = tf_cfg["sensor_simulation"].get("pushbroom_noise", {})
        if pb_cfg.get("enabled", False):
            pushbroom_amp = pb_cfg.get("max_stripe_amplitude", 0.02)
            n_stripes = pb_cfg.get("n_stripes", 5)
            synthetic = apply_pushbroom_noise(synthetic, n_stripes, pushbroom_amp, rng)

    params = TransformParams(
        pair_id=pair_id,
        random_seed=seed,
        scale_factor=scale_factor,
        rotation_deg=rotation_deg,
        translation_px=translation_px,
        interpolation=interpolation,
        illumination_gamma=gamma,
        shadow_extension_factor=shadow_ext,
        mtf_blur_sigma=mtf_sigma,
        pushbroom_stripe_amplitude=pushbroom_amp,
        transform_matrix=M.tolist(),
    )

    return synthetic, params, M

"""
src/synthetic/transforms.py — Physical Transformation Engine

Applies exact geometric and radiometric shifts to a source lunar image to
produce a synthetic target image, and simultaneously computes the exact
floating-point ground truth coordinates of all anchor points in the target.

Transformations are strictly derived from the SIH26166 problem statement:
  "Multi-modal, Sun-angle & scale-invariant image correspondence using
   Chandrayaan-2 (OHRC / TMC-2 / IIRS) vs LRO (NAC / WAC)"

Transformation matrix convention:
  M = Trans(cx, cy) @ Rot(theta) @ Scale(s) @ Trans(-cx, -cy)  (for centered transforms)
  For pure translation + rotation (as used here): M = Trans(dx, dy) @ Rot(theta)

Coordinate convention: all pixel coordinates are (col, row) = (x, y), 0-indexed.

EXCLUDED (do NOT add back):
  - Perspective/affine warping (orbital imagery is near-nadir)
  - JPEG compression / salt-and-pepper noise (not representative of PDS4 data)
  - Color jitter (lunar is monochromatic)

References: SYNTHETIC_BENCHMARK_ARCHITECTURE.md §2 and §4.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Interpolation flag mapping
_INTERP_MAP = {
    "lanczos": cv2.INTER_LANCZOS4,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


@dataclass
class TransformParams:
    """Parameters describing one synthetic transformation applied to a source image.

    All values are exact floating-point; used to reproduce and audit the transform.
    """
    pair_id: str
    random_seed: int

    # Geometric
    scale_factor: float = 1.0      # uniform scale applied to source image
    rotation_deg: float = 0.0      # rotation in degrees (applied about image center)
    translation_px: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [dx, dy]
    interpolation: str = "lanczos"

    # Photometric
    illumination_gamma: float = 1.0     # I_out = I_in ^ gamma
    shadow_extension_factor: float = 1.0  # 1.0 = no extension; <1.5 (per config)
    mtf_blur_sigma: float = 0.0          # Gaussian sigma (0 = no blur)
    pushbroom_stripe_amplitude: float = 0.0  # relative to image mean

    # Derived / output
    transform_matrix: Optional[List[List[float]]] = None  # 3x3 homogeneous matrix M

    def to_dict(self) -> dict:
        """Return JSON-serializable dict."""
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
    """Build the 3x3 homogeneous transformation matrix M.

    Transformation order: Scale about image centre -> Rotate about image centre
    -> Translate. This matches cv2.warpAffine's coordinate conventions.

    M = T(cx, cy) @ Scale(s) @ Rot(theta) @ T(-cx, -cy) @ T(dx, dy)

    Args:
        image_shape: (height, width) of the source image.
        scale_factor: Uniform scale factor (e.g., 0.25 for OHRC→WAC GSD).
        rotation_deg: Rotation angle in degrees (positive = counter-clockwise).
        translation_px: (dx, dy) sub-pixel translation in pixels.

    Returns:
        3x3 numpy float64 homogeneous transformation matrix M.
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    dx, dy = translation_px

    # Rotation + Scale combined using cv2.getRotationMatrix2D (returns 2x3)
    # angle: positive = counter-clockwise in standard image coords (y down)
    rot_scale = cv2.getRotationMatrix2D(center=(cx, cy), angle=rotation_deg, scale=scale_factor)
    # Convert 2x3 affine to 3x3 homogeneous
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = rot_scale.astype(np.float64)

    # Apply translation
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
    """Apply a 3x3 homogeneous transform matrix to an image using cv2.warpAffine.

    Args:
        image: Source image array (H, W) or (H, W, C). float32 or uint8.
        M: 3x3 homogeneous transform matrix from build_transform_matrix().
        output_shape: (height, width) of output image. Defaults to image.shape[:2].
        interpolation: Interpolation method key (see _INTERP_MAP). Use "lanczos"
            for GSD resampling per ARCHITECTURE.md §L1.
        border_mode: cv2 border mode for out-of-bounds pixels.
        border_value: Constant fill value when border_mode=BORDER_CONSTANT.

    Returns:
        Transformed image array (same dtype as input).
    """
    assert image.ndim in (2, 3), "apply_transform expects 2D or 3D image array."
    h, w = image.shape[:2]
    out_h, out_w = output_shape if output_shape is not None else (h, w)
    interp_flag = _INTERP_MAP.get(interpolation, cv2.INTER_LANCZOS4)

    # cv2.warpAffine expects a 2x3 affine matrix; extract top two rows of M
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
    """Apply the exact transformation matrix M to GT anchor coordinates.

    Computes the exact floating-point target coordinates for each GT anchor point.
    This is the analytical ground truth — no interpolation error.

    Args:
        src_pts: (N, 2) float64 array of source coordinates [(col, row), ...].
        M: 3x3 homogeneous transformation matrix.

    Returns:
        (N, 2) float64 array of target coordinates [(tgt_col, tgt_row), ...].

    Raises:
        AssertionError: If src_pts shape is wrong.
    """
    assert src_pts.ndim == 2 and src_pts.shape[-1] == 2, (
        "Expected (N, 2) array: (col, row)"
    )
    if src_pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Convert to homogeneous coordinates (N, 3)
    ones = np.ones((src_pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([src_pts.astype(np.float64), ones])  # (N, 3)

    # Apply: tgt_h = M @ src_h.T -> (3, N); transpose back to (N, 3)
    tgt_h = (M @ pts_h.T).T  # (N, 3)

    # Perspective divide (w coordinate) — for affine transforms w=1 always
    tgt_pts = tgt_h[:, :2] / tgt_h[:, 2:3]

    return tgt_pts  # (N, 2) — (col, row)


# ---------------------------------------------------------------------------
# Photometric Simulation Functions
# ---------------------------------------------------------------------------

def apply_illumination_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Non-linear photometric distortion simulating different solar incidence angles.

    I_out = I_in ^ gamma   (applied on [0, 1] normalized float image)

    Per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §2.1:
      gamma in [0.7, 1.4] simulates phase-angle / sun-angle variation.

    Args:
        image: Single-channel float image, expected range [0, 1].
        gamma: Gamma exponent.

    Returns:
        Gamma-corrected float image in [0, 1].
    """
    assert gamma > 0, f"Gamma must be positive, got {gamma}"
    img_clipped = np.clip(image.astype(np.float32), 0.0, 1.0)
    return np.power(img_clipped, gamma).astype(np.float32)


def apply_mtf_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Sensor-specific MTF (Modulation Transfer Function) blurring.

    Simulates the optical response of a different sensor by applying a
    Gaussian blur with the sensor-specific MTF sigma.

    Per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §2.1:
      sigma in [0.5, 1.5] for cross-sensor simulation.

    Args:
        image: Source image array (H, W), float32 or uint8.
        sigma: Gaussian blur standard deviation in pixels.

    Returns:
        Blurred image (same dtype as input).
    """
    if sigma <= 0:
        return image
    # Kernel size: odd, at least 3, covering 3-sigma
    k = max(3, int(np.ceil(sigma * 3)) * 2 + 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_pushbroom_noise(
    image: np.ndarray,
    n_stripes: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pushbroom sensor row-noise as slight vertical striping.

    Adds random per-column gain offsets to n_stripes randomly selected columns,
    simulating the column-correlated noise pattern of pushbroom sensors.

    Per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §2.1:
      max_stripe_amplitude = 0.02 (relative to image mean).

    Args:
        image: Source image, float32 in [0, 1].
        n_stripes: Number of stripes (columns) to perturb.
        amplitude: Maximum gain perturbation amplitude (relative to image mean).
        rng: NumPy random generator.

    Returns:
        Image with pushbroom noise applied (float32, clipped to [0, 1]).
    """
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
    """Mask and extend existing shadow regions in the direction of the sun.

    Simulates different illumination conditions by extending existing dark regions
    in the solar azimuth direction.

    Per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §2.1:
      max_extension_factor = 1.5 (extend by up to 50%).

    Note: This is a synthetic simulation, NOT a physical BRDF model.
    Treat results as controlled stress tests, not real shadow physics.

    Args:
        image: Source image, float32 in [0, 1].
        extension_factor: How much to extend shadows (1.0 = none, 1.5 = 50%).
        solar_azimuth_deg: Solar azimuth angle in degrees (clockwise from north).

    Returns:
        Image with extended shadow regions (float32, range [0, 1]).
    """
    if extension_factor <= 1.0:
        return image
    out = image.astype(np.float32).copy()
    # Simple threshold-based shadow mask (dark pixels)
    shadow_threshold = float(np.percentile(out, 10))
    shadow_mask = out < shadow_threshold

    # Compute extension kernel direction from solar azimuth
    # Convert azimuth (CW from North) to image vector (x right, y down)
    az_rad = np.deg2rad(solar_azimuth_deg)
    # Shadow extends opposite to sun direction
    shadow_dx = -np.sin(az_rad)
    shadow_dy = -np.cos(az_rad)  # y down in image space

    # Dilate shadow mask in the computed direction
    extension_px = max(1, int((extension_factor - 1.0) * 20))  # rough px extension
    # Simple morphological dilation in the shadow direction
    kernel_size = 2 * extension_px + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    cx_k = extension_px
    cy_k = extension_px
    # Draw a line in the shadow direction
    for step in range(1, extension_px + 1):
        dx_i = int(round(shadow_dx * step))
        dy_i = int(round(shadow_dy * step))
        r = cy_k + dy_i
        c = cx_k + dx_i
        if 0 <= r < kernel_size and 0 <= c < kernel_size:
            kernel[r, c] = 1
    kernel[cy_k, cx_k] = 1  # center always on

    dilated_mask = cv2.dilate(shadow_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    # Apply darkness in extended shadow region (reduce to shadow_threshold level)
    out[dilated_mask & ~shadow_mask] = shadow_threshold * 0.8
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Composite Transform Generator
# ---------------------------------------------------------------------------

def generate_synthetic_pair(
    source_image: np.ndarray,
    config: dict,
    pair_id: str,
    seed: int,
) -> Tuple[np.ndarray, TransformParams, np.ndarray]:
    """Generate a synthetic target image from a source image with known exact GT transform.

    Applies all enabled transformation types from config in a fixed order:
      1. Scale (GSD resampling — Lanczos)
      2. Sub-pixel translation + rotation
      3. Illumination gamma correction
      4. Shadow extension
      5. MTF blur
      6. Pushbroom noise

    The transform matrix M for (1, 2) is returned as part of TransformParams.
    Photometric transforms do NOT affect GT coordinates — only geometric ones do.

    Args:
        source_image: Single-channel source image (H, W), float32 in [0, 1].
        config: Benchmark config dict (from synthetic_benchmark.yaml).
        pair_id: Pair identifier.
        seed: Random seed for reproducibility (N=50 seed independence).

    Returns:
        Tuple of:
          - synthetic_image: (H, W) float32 target image
          - params: TransformParams describing all applied transforms
          - M: 3x3 float64 transformation matrix for geometric GT mapping
    """
    assert source_image.ndim == 2, "generate_synthetic_pair expects a 2D single-channel image."
    rng = np.random.default_rng(seed)
    tf_cfg = config.get("transforms", {})

    # -------------------------------------------------------------------------
    # 1. Build geometric transform parameters
    # -------------------------------------------------------------------------
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
        # Force non-integer sub-pixel shifts (avoid trivially-easy integer shifts)
        dx = float(rng.uniform(-max_shift, max_shift))
        dy = float(rng.uniform(-max_shift, max_shift))
        # Ensure non-zero fractional part
        if abs(dx - round(dx)) < 0.05:
            dx += 0.1 * rng.choice([-1, 1])
        if abs(dy - round(dy)) < 0.05:
            dy += 0.1 * rng.choice([-1, 1])
        translation_px = [dx, dy]

    # Build combined geometric matrix M
    M = build_transform_matrix(
        image_shape=source_image.shape,
        scale_factor=scale_factor,
        rotation_deg=rotation_deg,
        translation_px=tuple(translation_px),
    )
    interpolation = tf_cfg.get("scale", {}).get("interpolation", "lanczos")

    # -------------------------------------------------------------------------
    # 2. Apply geometric transform to image
    # -------------------------------------------------------------------------
    synthetic = apply_transform(source_image, M, interpolation=interpolation)
    # Clip to [0, 1]: Lanczos interpolation can produce slight Gibbs ringing
    # (overshoot/undershoot). All subsequent photometric transforms require [0,1].
    synthetic = np.clip(synthetic, 0.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Photometric: illumination gamma correction
    # -------------------------------------------------------------------------
    gamma = 1.0
    if tf_cfg.get("illumination", {}).get("enabled", False):
        g_range = tf_cfg["illumination"].get("gamma_range", [0.7, 1.4])
        gamma = float(rng.uniform(g_range[0], g_range[1]))
        synthetic = apply_illumination_gamma(synthetic, gamma)

    # -------------------------------------------------------------------------
    # 4. Photometric: shadow extension
    # -------------------------------------------------------------------------
    shadow_ext = 1.0
    if (tf_cfg.get("illumination", {}).get("enabled", False) and
            tf_cfg["illumination"].get("shadow_extension", {}).get("enabled", False)):
        max_ext = tf_cfg["illumination"]["shadow_extension"].get("max_extension_factor", 1.5)
        shadow_ext = float(rng.uniform(1.0, max_ext))
        solar_azimuth = 270.0  # default if not provided
        synthetic = apply_shadow_extension(synthetic, shadow_ext, solar_azimuth)

    # -------------------------------------------------------------------------
    # 5. Photometric: MTF blur
    # -------------------------------------------------------------------------
    mtf_sigma = 0.0
    if tf_cfg.get("sensor_simulation", {}).get("enabled", False):
        mtf_cfg = tf_cfg["sensor_simulation"].get("mtf_blur", {})
        if mtf_cfg.get("enabled", False):
            s_range = mtf_cfg.get("sigma_range", [0.5, 1.5])
            mtf_sigma = float(rng.uniform(s_range[0], s_range[1]))
            synthetic = apply_mtf_blur(synthetic, mtf_sigma)

    # -------------------------------------------------------------------------
    # 6. Photometric: pushbroom noise
    # -------------------------------------------------------------------------
    pushbroom_amp = 0.0
    if tf_cfg.get("sensor_simulation", {}).get("enabled", False):
        pb_cfg = tf_cfg["sensor_simulation"].get("pushbroom_noise", {})
        if pb_cfg.get("enabled", False):
            pushbroom_amp = pb_cfg.get("max_stripe_amplitude", 0.02)
            n_stripes = pb_cfg.get("n_stripes", 5)
            synthetic = apply_pushbroom_noise(synthetic, n_stripes, pushbroom_amp, rng)

    # -------------------------------------------------------------------------
    # Build params record
    # -------------------------------------------------------------------------
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

    logger.debug(
        "Generated synthetic pair '%s' (seed=%d): scale=%.3f rot=%.2f deg "
        "shift=(%.3f, %.3f) gamma=%.2f mtf_sigma=%.2f",
        pair_id, seed, scale_factor, rotation_deg,
        translation_px[0], translation_px[1], gamma, mtf_sigma,
    )

    return synthetic, params, M

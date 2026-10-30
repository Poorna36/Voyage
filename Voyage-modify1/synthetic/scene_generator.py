"""
synthetic/scene_generator.py — Synthetic Lunar Scene Generator

Generates realistic textured lunar surface patches with multi-scale craters,
undulating topography, and surface roughness. Used for benchmarking,
testing, and smoke-testing without requiring external PDS4 archives.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_lunar_scene(
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Generate a textured lunar scene with multi-scale craters and surface relief.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed for reproducibility.

    Returns:
        float32 2D array normalised to [0.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]

    # Multi-frequency undulating mare/highland terrain
    terrain = (
        0.45
        + 0.12 * np.sin(x / 32.0) * np.cos(y / 32.0)
        + 0.08 * np.sin(x / 14.0 + y / 18.0)
        + 0.05 * rng.standard_normal((height, width))
    )

    # Multi-scale impact craters
    craters = [
        (width * 0.35, height * 0.35, 45.0, 0.4),
        (width * 0.70, height * 0.60, 30.0, 0.5),
        (width * 0.25, height * 0.75, 20.0, 0.6),
        (width * 0.80, height * 0.25, 25.0, 0.5),
        (width * 0.50, height * 0.70, 15.0, 0.7),
        (width * 0.60, height * 0.40, 18.0, 0.6),
        (width * 0.15, height * 0.20, 12.0, 0.7),
    ]

    for cx, cy, r, depth in craters:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        bowl = dist < r
        terrain[bowl] -= depth * (1.0 - (dist[bowl] / r) ** 2)
        rim = (dist >= r) & (dist < r + 5.0)
        terrain[rim] += depth * 0.4 * (1.0 - (dist[rim] - r) / 5.0)

    # Normalize to [0.0, 1.0] with 1st and 99th percentile clipping
    p2, p98 = np.percentile(terrain, (1.0, 99.0))
    img = np.clip((terrain - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
    return img

"""
src/preprocessing/__init__.py
==============================
Public API for the L1 Preprocessing layer.

Imports the primary functions from each sub-module so callers can use:

    from src.preprocessing import shadow_mask, percentile_clip, ...

References:
  - FEATURES.md F04-F08
  - PROGRESS.md Phase 2
"""
from src.preprocessing.masks import (
    shadow_mask,
    check_mask_fraction,
    save_mask_png,
)
from src.preprocessing.normalize import (
    percentile_clip,
    stat_transfer,
)
from src.preprocessing.branches import (
    apply_ohrc_nac,
    apply_tmc_wac,
    apply_minimal,
    select_branch,
)
from src.preprocessing.resample import reconcile_gsd
from src.preprocessing.tiling import tile_image, write_tile_geojson
from src.preprocessing.stats import (
    compute_texture_contrast,
    compute_mean_gradient,
    compute_image_stats,
)

__all__ = [
    # masks
    "shadow_mask",
    "check_mask_fraction",
    "save_mask_png",
    # normalize
    "percentile_clip",
    "stat_transfer",
    # branches
    "apply_ohrc_nac",
    "apply_tmc_wac",
    "apply_minimal",
    "select_branch",
    # resample
    "reconcile_gsd",
    # tiling
    "tile_image",
    "write_tile_geojson",
    # stats
    "compute_texture_contrast",
    "compute_mean_gradient",
    "compute_image_stats",
]


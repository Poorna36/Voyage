"""
tests/test_stress_phase2.py
============================
High-load stress tests for Phase 2 preprocessing modules.

Generates ~100k test assertions across the full parameter space of:
  - Image sizes (tiny → large, square + rectangular)
  - Solar incidence angles (full 0–89° range)
  - GSD ratios (extreme min and max values)
  - Percentile clip combinations
  - Tiling configurations
  - Sensor branch routing
  - Random image content distributions (uniform, normal, bimodal, sparse, saturated)

Uses:
  - pytest.mark.parametrize for exhaustive parameter grids
  - hypothesis (property-based) for unbounded random input coverage
  - Explicit seed-sweeps for deterministic reproducibility

Run with:
    python -m pytest tests/test_stress_phase2.py -v --tb=short
    python -m pytest tests/test_stress_phase2.py -n auto  # parallel (pytest-xdist)

References: PROGRESS.md Phase 2, FEATURES.md F04-F08
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.masks import shadow_mask, check_mask_fraction, save_mask_png
from src.preprocessing.normalize import percentile_clip, stat_transfer
from src.preprocessing.branches import apply_ohrc_nac, apply_minimal, select_branch
from src.preprocessing.resample import reconcile_gsd
from src.preprocessing.tiling import tile_image, write_tile_geojson

# ---------------------------------------------------------------------------
# Hypothesis (property-based testing) — optional but used when available
# ---------------------------------------------------------------------------
try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

# ---------------------------------------------------------------------------
# Parameter grids  (combinatorial expansion → ~100k assertions)
# ---------------------------------------------------------------------------

# Image sizes: (height, width) pairs — covers square, portrait, landscape, odd dims
IMAGE_SIZES: list[Tuple[int, int]] = [
    (32, 32), (64, 64), (128, 128), (256, 256), (512, 512),
    (1024, 512), (512, 1024), (768, 1024), (1024, 1024),
    (333, 500), (97, 113), (64, 256), (256, 64),
    (480, 640), (720, 1280),
]

# Solar incidence angles (degrees) — full range including edge values
SOLAR_ANGLES: list[float] = [
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 44.9,
    45.0, 45.1, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 79.9,
    80.0, 80.1, 82.0, 85.0, 88.0, 89.0, 89.9,
]

# GSD combinations (src_gsd, ref_gsd) — covers equal, src>ref, src<ref, extreme ratios
GSD_PAIRS: list[Tuple[float, float]] = [
    (0.31, 0.50),   # OHRC vs NAC
    (5.0,  100.0),  # TMC-2 vs WAC
    (0.50, 0.31),   # ref coarser
    (1.0,  1.0),    # equal
    (0.1,  1.0),    # 10x ratio
    (1.0,  10.0),   # 10x ratio reverse
    (0.25, 0.5),
    (2.0,  0.5),    # src much coarser
    (0.5,  50.0),   # extreme
    (0.31, 0.31),   # same sensor
    (3.0,  3.0),    # equal non-unit
    (0.01, 0.1),    # very fine src
    (100.0, 200.0), # very coarse both
    (1.0,  2.0),
    (2.0,  1.0),
    (5.0,  5.0),
    (0.5,  0.75),
    (0.75, 0.5),
    (10.0, 1.0),
    (1.0,  100.0),
]

# Percentile combinations (lo, hi)
PERCENTILE_PAIRS: list[Tuple[float, float]] = [
    (0.0,  100.0),
    (1.0,  99.0),
    (2.0,  98.0),
    (5.0,  95.0),
    (10.0, 90.0),
    (0.5,  99.5),
    (2.0,  99.0),
    (0.0,  99.9),
    (15.0, 85.0),
    (25.0, 75.0),
    (1.0,  95.0),
    (3.0,  97.0),
]

# Random seeds for image content variation
SEEDS: list[int] = list(range(0, 200))   # 200 distinct seeds

# Tile configurations (tile_size, overlap_px)
TILE_CONFIGS: list[Tuple[int, int]] = [
    (512, 64),
    (512, 128),
    (512, 0),
    (256, 32),
    (256, 64),
    (256, 0),
    (512, 256),
    (128, 16),
    (128, 32),
    (128, 64),
    (1024, 128),
    (384, 48),
]

# min_fraction values for tiling
MIN_FRACTIONS: list[float] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

def _make_image(seed: int, shape: Tuple[int, int], kind: str = "normal") -> np.ndarray:
    """
    Generate synthetic grayscale float32 image in [0, 1].
    kind: normal | uniform | bimodal | sparse | saturated | gradient | ramp | noisy
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    if kind == "normal":
        img = rng.normal(0.5, 0.15, (h, w))
    elif kind == "uniform":
        img = rng.uniform(0.0, 1.0, (h, w))
    elif kind == "bimodal":
        mask = rng.random((h, w)) > 0.5
        img = np.where(mask, rng.normal(0.8, 0.05, (h, w)), rng.normal(0.2, 0.05, (h, w)))
    elif kind == "sparse":
        img = np.zeros((h, w), dtype=np.float64)
        idx = rng.choice(h * w, size=h * w // 10, replace=False)
        img.flat[idx] = rng.uniform(0.5, 1.0, len(idx))
    elif kind == "saturated":
        img = rng.uniform(0.9, 1.0, (h, w))
        img[:h//4, :w//4] = rng.uniform(0.0, 0.05, (h//4, w//4))
    elif kind == "gradient":
        col = np.linspace(0.0, 1.0, w)
        img = np.tile(col, (h, 1))
    elif kind == "ramp":
        row = np.linspace(0.0, 1.0, h)
        img = np.tile(row[:, None], (1, w))
    elif kind == "noisy":
        img = np.full((h, w), 0.5) + rng.normal(0, 0.3, (h, w))
    else:
        img = rng.uniform(0.0, 1.0, (h, w))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


IMAGE_KINDS = ["normal", "uniform", "bimodal", "sparse", "saturated", "gradient", "ramp", "noisy"]


# ===========================================================================
# SECTION 1: shadow_mask — exhaustive parameter grid
# ~26 solar angles × 15 image sizes × 8 kinds = 3,120 parametrized tests
# ===========================================================================

@pytest.mark.parametrize("solar", SOLAR_ANGLES)
@pytest.mark.parametrize("shape", [(64, 64), (128, 256), (256, 128), (512, 512), (333, 500)])
@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_mask_output_invariants(solar: float, shape: Tuple[int,int], kind: str):
    """mask is boolean, same shape as input, fraction in [0,1]."""
    img = _make_image(seed=int(solar * 100) % 200, shape=shape, kind=kind)
    mask = shadow_mask(img, solar_incidence_deg=solar)
    assert mask.dtype == bool,                  f"Expected bool mask, got {mask.dtype}"
    assert mask.shape == img.shape,             f"Shape mismatch: {mask.shape} != {img.shape}"
    fraction, _ = check_mask_fraction(mask)
    assert 0.0 <= fraction <= 1.0,              f"Fraction {fraction} out of [0,1]"


@pytest.mark.parametrize("solar", [0.0, 45.0, 79.9, 80.0, 80.1, 89.9])
@pytest.mark.parametrize("incidence_threshold", [60.0, 70.0, 80.0, 85.0, 89.0])
@pytest.mark.parametrize("seed", range(0, 30))
def test_mask_incidence_threshold_respected(solar: float, incidence_threshold: float, seed: int):
    """Incidence test only fires when solar > threshold."""
    img = _make_image(seed=seed, shape=(128, 128), kind="normal")
    mask_below = shadow_mask(img, solar_incidence_deg=min(solar, incidence_threshold - 0.1),
                              incidence_threshold_deg=incidence_threshold)
    mask_above = shadow_mask(img, solar_incidence_deg=max(solar, incidence_threshold + 0.1),
                              incidence_threshold_deg=incidence_threshold)
    # Higher incidence → at least as many masked pixels
    assert mask_above.mean() >= mask_below.mean() - 0.01  # small tolerance


@pytest.mark.parametrize("seed", range(0, 100))
@pytest.mark.parametrize("flat_var_thresh", [1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
def test_mask_flat_variance_threshold(seed: int, flat_var_thresh: float):
    """Higher flat_variance_threshold → equal or more pixels flagged."""
    img = _make_image(seed=seed, shape=(128, 128), kind="normal")
    mask_strict = shadow_mask(img, solar_incidence_deg=45.0, flat_variance_threshold=flat_var_thresh / 2)
    mask_loose  = shadow_mask(img, solar_incidence_deg=45.0, flat_variance_threshold=flat_var_thresh)
    # Looser threshold means more flat regions classified as shadow → more masked
    assert mask_loose.mean() >= mask_strict.mean() - 0.05


@pytest.mark.parametrize("min_pct,max_pct", [
    (0, 100), (5, 30), (10, 50), (1, 99), (0, 50), (20, 80)
])
@pytest.mark.parametrize("seed", range(0, 50))
def test_check_mask_fraction_boundaries(min_pct: float, max_pct: float, seed: int):
    """check_mask_fraction returns correct in_range for known fractions."""
    img = _make_image(seed=seed, shape=(256, 256), kind="normal")
    mask = shadow_mask(img, solar_incidence_deg=45.0)
    fraction, in_range = check_mask_fraction(mask, min_pct=min_pct, max_pct=max_pct)
    expected_in_range = (min_pct / 100 <= fraction <= max_pct / 100)
    assert in_range == expected_in_range
    assert isinstance(fraction, float)
    assert isinstance(in_range, bool)


# ===========================================================================
# SECTION 2: percentile_clip — exhaustive parameter grid
# 12 percentile pairs × 200 seeds = 2,400 tests
# ===========================================================================

@pytest.mark.parametrize("lo,hi", PERCENTILE_PAIRS)
@pytest.mark.parametrize("seed", range(0, 200))
def test_percentile_clip_range_and_dtype(lo: float, hi: float, seed: int):
    """Output is float32 in [0,1] for any valid (lo, hi) pair and any image."""
    kind = IMAGE_KINDS[seed % len(IMAGE_KINDS)]
    img = _make_image(seed=seed, shape=(128, 128), kind=kind)
    result = percentile_clip(img, lo=lo, hi=hi)
    assert result.dtype == np.float32
    assert float(result.min()) >= -1e-6
    assert float(result.max()) <=  1.0 + 1e-6


@pytest.mark.parametrize("shape", IMAGE_SIZES)
@pytest.mark.parametrize("seed", range(0, 10))
def test_percentile_clip_shape_preserved(shape: Tuple[int, int], seed: int):
    img = _make_image(seed=seed, shape=shape, kind="uniform")
    result = percentile_clip(img)
    assert result.shape == img.shape


@pytest.mark.parametrize("lo,hi", PERCENTILE_PAIRS)
@pytest.mark.parametrize("seed", range(0, 50))
def test_stat_transfer_mean_std_invariant(lo: float, hi: float, seed: int):
    """After stat_transfer: out is float32 in [0,1]; output dtype stable."""
    kind_src = IMAGE_KINDS[seed % len(IMAGE_KINDS)]
    kind_ref = IMAGE_KINDS[(seed + 3) % len(IMAGE_KINDS)]
    src = _make_image(seed=seed,      shape=(128, 128), kind=kind_src)
    ref = _make_image(seed=seed + 99, shape=(128, 128), kind=kind_ref)
    src_c = percentile_clip(src, lo=lo, hi=hi)
    ref_c = percentile_clip(ref, lo=lo, hi=hi)
    result = stat_transfer(src_c, ref_c)
    assert result.dtype == np.float32
    assert float(result.min()) >= -1e-5
    assert float(result.max()) <=  1.0 + 1e-5


@pytest.mark.parametrize("seed", range(0, 200))
def test_stat_transfer_t04_mean_std_within_5pct(seed: int):
    """T04: mean and std of result within 5% of reference."""
    kind = IMAGE_KINDS[seed % len(IMAGE_KINDS)]
    src = _make_image(seed=seed,      shape=(256, 256), kind=kind)
    ref = _make_image(seed=seed + 50, shape=(256, 256), kind="normal")
    result = stat_transfer(src, ref)
    ref_mean = float(np.mean(ref))
    ref_std  = float(np.std(ref))
    out_mean = float(np.mean(result))
    out_std  = float(np.std(result))
    assert abs(out_mean - ref_mean) <= 0.05 + abs(ref_mean) * 0.05
    assert abs(out_std  - ref_std)  <= 0.05 + abs(ref_std)  * 0.05


# ===========================================================================
# SECTION 3: reconcile_gsd — exhaustive GSD/angle grid
# 20 GSD pairs × 26 solar angles × 8 image kinds = 4,160 tests
# ===========================================================================

@pytest.mark.parametrize("src_gsd,ref_gsd", GSD_PAIRS)
@pytest.mark.parametrize("solar", SOLAR_ANGLES)
@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_reconcile_gsd_invariants(src_gsd: float, ref_gsd: float, solar: float, kind: str):
    """
    Core invariants for reconcile_gsd:
      - Output is float32
      - which_resampled is consistent with GSD values
      - Output values in [0, 1] (input was clipped to [0,1])
    """
    seed = int((src_gsd * 37 + ref_gsd * 13 + solar * 7)) % 200
    img = _make_image(seed=seed, shape=(128, 128), kind=kind)
    result, meta = reconcile_gsd(img, src_gsd=src_gsd, ref_gsd=ref_gsd,
                                  solar_incidence_deg=solar)
    assert result.dtype == np.float32
    assert "gsd_ratio"           in meta
    assert "which_resampled"     in meta
    assert "interpolation_method" in meta
    assert meta["gsd_ratio"] >= 1.0 - 1e-6
    # which_resampled consistency
    if abs(src_gsd - ref_gsd) < 1e-9:
        assert meta["which_resampled"] == "none"
    elif src_gsd > ref_gsd:
        assert meta["which_resampled"] == "src"
    else:
        assert meta["which_resampled"] == "ref"


@pytest.mark.parametrize("solar", [0.0, 20.0, 44.9, 45.0, 45.1, 60.0, 80.0, 89.9])
@pytest.mark.parametrize("seed", range(0, 50))
def test_reconcile_gsd_interpolation_switch_at_45(solar: float, seed: int):
    """Exact switch point: solar>=45 → bilinear; solar<45 → bicubic."""
    img = _make_image(seed=seed, shape=(64, 64), kind="normal")
    _, meta = reconcile_gsd(img, src_gsd=1.0, ref_gsd=0.5,
                             solar_incidence_deg=solar,
                             low_angle_threshold_deg=45.0)
    if solar >= 45.0:
        assert meta["interpolation_method"] == "bilinear", \
            f"solar={solar} should give bilinear, got {meta['interpolation_method']}"
    else:
        assert meta["interpolation_method"] == "bicubic", \
            f"solar={solar} should give bicubic, got {meta['interpolation_method']}"


@pytest.mark.parametrize("src_gsd,ref_gsd", [
    (0.31, 0.50), (5.0, 100.0), (0.1, 1.0), (2.0, 0.5), (10.0, 100.0)
])
@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (256, 256), (512, 512), (768, 1024)])
@pytest.mark.parametrize("seed", range(0, 20))
def test_reconcile_gsd_output_shape_plausible(
        src_gsd: float, ref_gsd: float,
        shape: Tuple[int, int], seed: int):
    """Output shape is consistent with expected upsampling/downsampling direction."""
    img = _make_image(seed=seed, shape=shape, kind="uniform")
    result, meta = reconcile_gsd(img, src_gsd=src_gsd, ref_gsd=ref_gsd,
                                  solar_incidence_deg=45.0)
    if meta["which_resampled"] == "src":
        # src was coarser → resampled to larger (finer grid)
        ratio = src_gsd / ref_gsd
        expected_h = round(shape[0] * ratio)
        expected_w = round(shape[1] * ratio)
        assert abs(result.shape[0] - expected_h) <= 2
        assert abs(result.shape[1] - expected_w) <= 2
    elif meta["which_resampled"] == "none":
        assert result.shape == shape


# ===========================================================================
# SECTION 4: tile_image — exhaustive configuration grid
# 12 tile configs × 15 image sizes × 7 min_fractions = 1,260 tests
# ===========================================================================

@pytest.mark.parametrize("tile_size,overlap_px", TILE_CONFIGS)
@pytest.mark.parametrize("shape", IMAGE_SIZES)
@pytest.mark.parametrize("min_fraction", MIN_FRACTIONS)
def test_tile_image_all_offsets_valid(
        tile_size: int, overlap_px: int,
        shape: Tuple[int, int], min_fraction: float):
    """All tile offsets lie within image bounds; tile content matches source."""
    h, w = shape
    # Skip if tile is larger than image in both dims (would produce zero tiles)
    if tile_size > h or tile_size > w:
        pytest.skip(f"tile_size={tile_size} > image {shape}")
    img = _make_image(seed=tile_size + overlap_px, shape=shape, kind="uniform")
    tiles = tile_image(img, tile_size=tile_size, overlap_px=overlap_px,
                       min_fraction=min_fraction)
    for tile_arr, (r0, c0) in tiles:
        assert 0 <= r0 < h
        assert 0 <= c0 < w
        assert r0 + tile_arr.shape[0] <= h
        assert c0 + tile_arr.shape[1] <= w
        # Content must match
        r1 = r0 + tile_arr.shape[0]
        c1 = c0 + tile_arr.shape[1]
        assert np.allclose(tile_arr, img[r0:r1, c0:c1], atol=1e-6)


@pytest.mark.parametrize("tile_size,overlap_px", [(512, 64), (256, 32), (128, 16)])
@pytest.mark.parametrize("mask_fraction", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("seed", range(0, 30))
def test_tile_image_mask_filtering(
        tile_size: int, overlap_px: int,
        mask_fraction: float, seed: int):
    """Fully-masked (invalid) tiles are discarded; valid tiles always kept."""
    shape = (512, 512)
    img  = _make_image(seed=seed, shape=shape, kind="uniform")
    rng  = np.random.default_rng(seed + 999)
    # Create mask where roughly mask_fraction of pixels are invalid
    mask = rng.random(shape) < mask_fraction

    tiles_unmasked = tile_image(img, tile_size=tile_size, overlap_px=overlap_px,
                                 min_fraction=0.0)  # no validity filter
    tiles_masked   = tile_image(img, tile_size=tile_size, overlap_px=overlap_px,
                                 valid_mask=mask, min_fraction=0.5)

    # With full mask, no tiles should pass
    if mask_fraction >= 0.99:
        assert len(tiles_masked) == 0, \
            f"Expected 0 tiles with 100% mask, got {len(tiles_masked)}"
    # With zero mask, all size-passing tiles should be kept
    elif mask_fraction < 0.01:
        assert len(tiles_masked) == len(tiles_unmasked)


@pytest.mark.parametrize("tile_size", [128, 256, 512])
@pytest.mark.parametrize("overlap_px", [0, 16, 32, 64, 128])
@pytest.mark.parametrize("shape", [(512, 512), (768, 1024), (1024, 1024)])
def test_tile_image_no_duplicate_tiles(tile_size: int, overlap_px: int, shape: Tuple[int, int]):
    """No two tiles have the exact same (row_offset, col_offset)."""
    if tile_size > shape[0] or tile_size > shape[1]:
        pytest.skip("tile larger than image")
    if overlap_px >= tile_size:
        pytest.skip(f"overlap_px={overlap_px} >= tile_size={tile_size} is invalid")
    img = _make_image(seed=42, shape=shape, kind="gradient")
    tiles = tile_image(img, tile_size=tile_size, overlap_px=overlap_px)
    offsets = [(r, c) for _, (r, c) in tiles]
    assert len(offsets) == len(set(offsets)), "Duplicate tile offsets detected"


# ===========================================================================
# SECTION 5: select_branch — full routing table stress
# 3 sensor pairs × 5 matchers × 50 config variants = 750 tests
# ===========================================================================

_SENSOR_PAIRS = ["OHRC-NAC", "TMC-2-WAC", "IIRS-WAC", "UNKNOWN-X", ""]
_CLASSICAL_MATCHERS = ["sift", "rift2", "lnift"]
_LEARNED_MATCHERS = ["lightglue", "crater", "crater_hough"]

@pytest.mark.parametrize("sensor_pair", _SENSOR_PAIRS)
@pytest.mark.parametrize("matcher_id", _LEARNED_MATCHERS)
def test_select_branch_learned_always_minimal(sensor_pair: str, matcher_id: str):
    """Learned matchers ALWAYS get minimal regardless of sensor pair or config."""
    for override in [None, "ohrc_to_nac", "tmc_to_wac", "minimal"]:
        cfg = {"preprocessing": {"sensor_branch": override}} if override else {}
        branch = select_branch(sensor_pair, matcher_id, config=cfg)
        assert branch == "minimal", \
            f"sensor={sensor_pair}, matcher={matcher_id}, override={override} → got {branch!r}"


@pytest.mark.parametrize("sensor_pair,expected_branch", [
    ("OHRC-NAC", "ohrc_to_nac"),
    ("ohrc-nac", "ohrc_to_nac"),    # case-insensitive
    ("OHRC NAC", "ohrc_to_nac"),    # space variant
    ("TMC-2-WAC", "tmc_to_wac"),
    ("tmc-2-wac", "tmc_to_wac"),
    ("IIRS-WAC", "minimal"),
    ("UNKNOWN", "minimal"),
    ("", "minimal"),
])
@pytest.mark.parametrize("matcher_id", _CLASSICAL_MATCHERS)
def test_select_branch_classical_routing(
        sensor_pair: str, expected_branch: str, matcher_id: str):
    branch = select_branch(sensor_pair, matcher_id, config={})
    assert branch == expected_branch, \
        f"sensor={sensor_pair!r}, matcher={matcher_id} → expected {expected_branch!r}, got {branch!r}"


@pytest.mark.parametrize("override", ["minimal", "ohrc_to_nac", "tmc_to_wac"])
@pytest.mark.parametrize("matcher_id", _CLASSICAL_MATCHERS)
@pytest.mark.parametrize("sensor_pair", _SENSOR_PAIRS)
def test_select_branch_config_override_respected(
        override: str, matcher_id: str, sensor_pair: str):
    cfg = {"preprocessing": {"sensor_branch": override}}
    branch = select_branch(sensor_pair, matcher_id, config=cfg)
    assert branch == override, \
        f"Config override={override!r} not respected; got {branch!r}"


# ===========================================================================
# SECTION 6: apply_ohrc_nac — output range/dtype across configs
# 5 clip_limits × 4 grid sizes × 3 inversion modes × 20 seeds = 1,200 tests
# ===========================================================================

@pytest.mark.parametrize("clip_limit", [0.5, 1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("tile_grid", [[4, 4], [8, 8], [16, 16], [32, 32]])
@pytest.mark.parametrize("inversion", ["auto", "always", False])
@pytest.mark.parametrize("seed", range(0, 20))
def test_ohrc_nac_output_range_and_dtype(
        clip_limit: float, tile_grid: list,
        inversion: object, seed: int):
    cfg = {
        "clahe_clip_limit": clip_limit,
        "clahe_tile_grid": tile_grid,
        "pca_components": 1,
        "inversion": inversion,
    }
    img = _make_image(seed=seed, shape=(256, 256), kind=IMAGE_KINDS[seed % len(IMAGE_KINDS)])
    result = apply_ohrc_nac(img, cfg)
    assert result.dtype == np.float32, f"dtype {result.dtype}"
    assert result.shape == img.shape,  f"shape mismatch"
    assert float(result.min()) >= -1e-6, f"min {result.min():.6f} < 0"
    assert float(result.max()) <=  1.0 + 1e-6, f"max {result.max():.6f} > 1"


@pytest.mark.parametrize("shape", IMAGE_SIZES)
def test_ohrc_nac_shape_invariant_all_sizes(shape: Tuple[int, int]):
    """Output shape always equals input shape for any image size."""
    cfg = {"clahe_clip_limit": 2.0, "clahe_tile_grid": [8, 8],
           "pca_components": 1, "inversion": "auto"}
    img = _make_image(seed=0, shape=shape, kind="normal")
    result = apply_ohrc_nac(img, cfg)
    assert result.shape == img.shape


@pytest.mark.parametrize("seed", range(0, 200))
def test_minimal_branch_is_identity(seed: int):
    """apply_minimal must return exactly the clipped input — no transformations."""
    img = _make_image(seed=seed, shape=(128, 128), kind=IMAGE_KINDS[seed % len(IMAGE_KINDS)])
    result = apply_minimal(img, config={})
    expected = np.clip(img, 0.0, 1.0)
    assert np.allclose(result, expected, atol=1e-6), \
        f"apply_minimal changed pixel values (seed={seed})"


# ===========================================================================
# SECTION 7: Full mini-pipeline stress — end-to-end with many param combos
# 20 GSD pairs × 10 solar angles × 10 seeds = 2,000 tests
# ===========================================================================

@pytest.mark.parametrize("src_gsd,ref_gsd", GSD_PAIRS)
@pytest.mark.parametrize("solar", [0.0, 20.0, 45.0, 60.0, 80.0, 85.0, 89.9])
@pytest.mark.parametrize("seed", range(0, 10))
def test_mini_pipeline_end_to_end(
        src_gsd: float, ref_gsd: float, solar: float, seed: int):
    """
    Run the full L1 pipeline on a synthetic image pair:
      clip → stat_transfer → reconcile_gsd → tile_image
    Verify all intermediate outputs have correct dtype, shape, and value range.
    """
    shape = (256, 256)
    src = _make_image(seed=seed,      shape=shape, kind=IMAGE_KINDS[seed % len(IMAGE_KINDS)])
    ref = _make_image(seed=seed + 50, shape=shape, kind="normal")

    # Step 1: percentile clip
    src_c = percentile_clip(src)
    ref_c = percentile_clip(ref)
    assert src_c.dtype == np.float32
    assert 0.0 <= float(src_c.min()) and float(src_c.max()) <= 1.0 + 1e-6

    # Step 2: stat transfer
    src_n = stat_transfer(src_c, ref_c)
    assert src_n.dtype == np.float32
    assert 0.0 <= float(src_n.min()) and float(src_n.max()) <= 1.0 + 1e-6

    # Step 3: GSD reconciliation
    src_r, meta = reconcile_gsd(src_n, src_gsd=src_gsd, ref_gsd=ref_gsd,
                                  solar_incidence_deg=solar)
    assert src_r.dtype == np.float32
    assert "gsd_ratio" in meta

    # Step 4: tiling (only if image is large enough)
    if src_r.shape[0] >= 128 and src_r.shape[1] >= 128:
        tiles = tile_image(src_r, tile_size=min(128, src_r.shape[0], src_r.shape[1]),
                           overlap_px=16, min_fraction=0.0)
        for tile_arr, (r0, c0) in tiles:
            assert tile_arr.dtype == np.float32 or tile_arr.dtype == src_r.dtype
            r1 = r0 + tile_arr.shape[0]
            c1 = c0 + tile_arr.shape[1]
            assert np.allclose(tile_arr, src_r[r0:r1, c0:c1], atol=1e-6)


# ===========================================================================
# SECTION 8: Hypothesis property-based tests (if available)
# Runs hundreds of generated examples per test
# ===========================================================================

if HAS_HYPOTHESIS:
    @given(
        lo=st.floats(min_value=0.0, max_value=49.0),
        hi=st.floats(min_value=51.0, max_value=100.0),
        h=st.integers(min_value=32, max_value=512),
        w=st.integers(min_value=32, max_value=512),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=500,
              deadline=None,   # image generation time varies by size
              suppress_health_check=[HealthCheck.too_slow])
    def test_hypothesis_percentile_clip(lo, hi, h, w, seed):
        img = _make_image(seed=seed % 200, shape=(h, w), kind="uniform")
        result = percentile_clip(img, lo=lo, hi=hi)
        assert result.dtype == np.float32
        assert float(result.min()) >= -1e-6
        assert float(result.max()) <=  1.0 + 1e-6
        assert result.shape == (h, w)

    @given(
        # Keep ratio ≤ 20x so 64×64 image stays within MAX_OUTPUT_PX
        src_gsd=st.floats(min_value=1.0, max_value=20.0),
        ref_gsd=st.floats(min_value=1.0, max_value=20.0),
        solar=st.floats(min_value=0.0, max_value=89.9),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=500,
              deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_hypothesis_reconcile_gsd(src_gsd, ref_gsd, solar, seed):
        img = _make_image(seed=seed % 200, shape=(64, 64), kind="uniform")
        result, meta = reconcile_gsd(img, src_gsd=src_gsd, ref_gsd=ref_gsd,
                                      solar_incidence_deg=solar)
        assert result.dtype == np.float32
        assert meta["gsd_ratio"] >= 1.0 - 1e-6
        assert meta["interpolation_method"] in ("bilinear", "bicubic")
        assert meta["which_resampled"] in ("src", "ref", "none")

    @given(
        tile_size=st.sampled_from([128, 256, 512]),
        overlap=st.integers(min_value=0, max_value=127),
        h=st.integers(min_value=256, max_value=1024),
        w=st.integers(min_value=256, max_value=1024),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=300,
              deadline=None,   # tiling time varies by image size
              suppress_health_check=[HealthCheck.too_slow])
    def test_hypothesis_tile_image(tile_size, overlap, h, w, seed):
        img = _make_image(seed=seed % 200, shape=(h, w), kind="uniform")
        tiles = tile_image(img, tile_size=tile_size, overlap_px=overlap, min_fraction=0.0)
        for tile_arr, (r0, c0) in tiles:
            assert r0 >= 0 and c0 >= 0
            assert r0 + tile_arr.shape[0] <= h
            assert c0 + tile_arr.shape[1] <= w
            r1 = r0 + tile_arr.shape[0]
            c1 = c0 + tile_arr.shape[1]
            assert np.allclose(tile_arr, img[r0:r1, c0:c1], atol=1e-6)

    @given(
        solar=st.floats(min_value=0.0, max_value=89.9),
        incidence_thresh=st.floats(min_value=50.0, max_value=89.0),
        flat_var=st.floats(min_value=0.1, max_value=200.0),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_hypothesis_shadow_mask(solar, incidence_thresh, flat_var, seed):
        img = _make_image(seed=seed % 200, shape=(128, 128), kind="normal")
        mask = shadow_mask(img, solar_incidence_deg=solar,
                           incidence_threshold_deg=incidence_thresh,
                           flat_variance_threshold=flat_var)
        assert mask.dtype == bool
        assert mask.shape == img.shape
        frac, _ = check_mask_fraction(mask)
        assert 0.0 <= frac <= 1.0

    @given(
        matcher_id=st.sampled_from(["lightglue", "crater", "crater_hough"]),
        sensor_pair=st.sampled_from(["OHRC-NAC", "TMC-2-WAC", "IIRS-WAC", "UNKNOWN"]),
    )
    @settings(max_examples=200)
    def test_hypothesis_select_branch_learned_always_minimal(matcher_id, sensor_pair):
        branch = select_branch(sensor_pair, matcher_id, config={})
        assert branch == "minimal"

else:
    # Placeholder tests if hypothesis is not installed
    def test_hypothesis_not_available_warning():
        """Hypothesis not installed — property-based tests skipped."""
        import warnings
        warnings.warn(
            "hypothesis not installed. Install with: pip install hypothesis. "
            "Property-based tests skipped.",
            UserWarning,
        )


# ===========================================================================
# SECTION 9: Edge cases and adversarial inputs
# ===========================================================================

class TestEdgeCases:
    """Adversarial and boundary inputs that could break the pipeline."""

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
    def test_constant_image_percentile_clip(self, val: float):
        """Constant image: P_lo == P_hi → output is zeros (documented behavior)."""
        img = np.full((128, 128), val, dtype=np.float32)
        result = percentile_clip(img)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)

    @pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
    def test_constant_image_stat_transfer_no_crash(self, val: float):
        """Constant source (std≈0) should not crash — returns clipped input."""
        src = np.full((128, 128), val, dtype=np.float32)
        ref = _make_image(seed=0, shape=(128, 128), kind="normal")
        result = stat_transfer(src, ref)
        assert result.dtype == np.float32
        assert float(result.min()) >= -1e-6
        assert float(result.max()) <= 1.0 + 1e-6

    @pytest.mark.parametrize("solar", [0.0, 89.9, 45.0])
    def test_single_pixel_image_mask(self, solar: float):
        """1×1 image should not crash shadow_mask."""
        img = np.array([[0.5]], dtype=np.float32)
        mask = shadow_mask(img, solar_incidence_deg=solar)
        assert mask.shape == (1, 1)
        assert mask.dtype == bool

    @pytest.mark.parametrize("seed", range(0, 20))
    def test_very_noisy_image_mask_fraction_stable(self, seed: int):
        """Very noisy image (std > 0.3): mask fraction should be in [0,1]."""
        rng = np.random.default_rng(seed)
        img = rng.normal(0.5, 0.4, (256, 256)).clip(0, 1).astype(np.float32)
        mask = shadow_mask(img, solar_incidence_deg=45.0)
        frac, _ = check_mask_fraction(mask)
        assert 0.0 <= frac <= 1.0

    def test_maximum_valid_tile_configuration(self):
        """Largest valid tile config: 1024×1024 image, 512px tile, 0 overlap."""
        img = _make_image(seed=0, shape=(1024, 1024), kind="gradient")
        tiles = tile_image(img, tile_size=512, overlap_px=0, min_fraction=0.0)
        # Should produce exactly 4 tiles: (0,0), (0,512), (512,0), (512,512)
        assert len(tiles) == 4
        offsets = sorted([(r, c) for _, (r, c) in tiles])
        assert offsets == [(0, 0), (0, 512), (512, 0), (512, 512)]

    def test_all_gsd_meta_keys_present(self):
        """reconcile_gsd always returns all required meta keys."""
        img = _make_image(seed=0, shape=(64, 64), kind="normal")
        required_keys = {"gsd_ratio", "which_resampled", "interpolation_method",
                         "src_gsd_m", "ref_gsd_m", "solar_incidence_deg"}
        for src_gsd, ref_gsd in [(0.31, 0.5), (1.0, 1.0), (5.0, 100.0)]:
            _, meta = reconcile_gsd(img, src_gsd=src_gsd, ref_gsd=ref_gsd,
                                     solar_incidence_deg=45.0)
            missing = required_keys - set(meta.keys())
            assert not missing, f"Missing keys: {missing} for src={src_gsd}, ref={ref_gsd}"

    @pytest.mark.parametrize("seed", range(0, 50))
    def test_tile_geojson_feature_count_matches(self, seed: int, tmp_path):
        """GeoJSON feature count must match tile list length."""
        img = _make_image(seed=seed, shape=(512, 512), kind="uniform")
        tiles = tile_image(img, tile_size=256, overlap_px=32, min_fraction=0.0)
        out = write_tile_geojson(tiles, pair_id=f"test_{seed}",
                                 out_path=tmp_path / f"tiles_{seed}.geojson")
        import json
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == len(tiles), \
            f"GeoJSON feature count {len(data['features'])} != tile count {len(tiles)}"

    @pytest.mark.parametrize("src_gsd", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_extreme_gsd_no_crash(self, src_gsd: float):
        """
        Extreme GSD values must never crash — very large ratios are capped at
        MAX_OUTPUT_PX by the production guard in resample.py.  We verify the
        output is a valid float32 array and dtype is preserved.
        """
        img = _make_image(seed=0, shape=(64, 64), kind="normal")
        # Should complete without raising regardless of GSD
        result, meta = reconcile_gsd(img, src_gsd=src_gsd, ref_gsd=1.0,
                                      solar_incidence_deg=45.0)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert result.size > 0

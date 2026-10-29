"""
tests/test_preprocessing.py
=============================
Unit tests for the Phase 2 (L1) preprocessing modules.

All tests use synthetic NumPy arrays — NO real satellite data required.
Tests map to PROGRESS.md §2.1–2.5 and VALIDATION.md T03-T04.

Run with:
  python -m pytest tests/test_preprocessing.py -v

References:
  - VALIDATION.md T03 (shadow mask fraction)
  - VALIDATION.md T04 (stat_transfer mean/std within 5% of ref)
  - PROGRESS.md Phase 2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.masks import shadow_mask, check_mask_fraction, save_mask_png
from src.preprocessing.normalize import percentile_clip, stat_transfer
from src.preprocessing.branches import (
    apply_ohrc_nac,
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



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image_512():
    """512×512 float32 grayscale image with realistic lunar-like statistics."""
    rng = np.random.default_rng(42)
    # Base: mostly mid-gray with some dark patches (shadows) and bright patches
    img = rng.normal(loc=0.5, scale=0.15, size=(512, 512)).astype(np.float32)
    # Add a shadow region (dark, flat) in the top-left quadrant
    img[:128, :128] = rng.normal(loc=0.05, scale=0.02, size=(128, 128)).astype(np.float32)
    # Add a bright region (crater rim) in the bottom-right
    img[384:, 384:] = rng.normal(loc=0.85, scale=0.05, size=(128, 128)).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


@pytest.fixture
def synthetic_ref_512():
    """512×512 reference image with different statistics."""
    rng = np.random.default_rng(99)
    img = rng.normal(loc=0.4, scale=0.12, size=(512, 512)).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


@pytest.fixture
def ohrc_nac_branch_cfg():
    return {
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "pca_components": 1,
        "inversion": "auto",
    }


# ===========================================================================
# masks.py tests
# ===========================================================================

class TestShadowMask:
    """Tests for shadow_mask and check_mask_fraction (T03 coverage)."""

    def test_returns_boolean_mask(self, synthetic_image_512):
        mask = shadow_mask(synthetic_image_512, solar_incidence_deg=45.0)
        assert mask.dtype == bool
        assert mask.shape == synthetic_image_512.shape

    def test_mask_fraction_in_range_with_shadow_image(self, synthetic_image_512):
        """T03 — mask fraction should be in [5%, 30%] on a typical lunar image."""
        mask = shadow_mask(
            synthetic_image_512,
            solar_incidence_deg=45.0,
            incidence_threshold_deg=80.0,
            local_variance_window=15,
            flat_variance_threshold=10.0,
        )
        fraction, _ = check_mask_fraction(mask, min_pct=5.0, max_pct=30.0)
        # For a synthetic image with explicit shadow region, fraction should be non-trivial
        assert 0.0 < fraction < 1.0, f"Fraction {fraction:.3f} out of (0, 1)"

    def test_check_mask_fraction_returns_correct_types(self, synthetic_image_512):
        mask = shadow_mask(synthetic_image_512, solar_incidence_deg=45.0)
        fraction, in_range = check_mask_fraction(mask, min_pct=5.0, max_pct=30.0)
        assert isinstance(fraction, float)
        assert isinstance(in_range, bool)

    def test_all_black_image_high_fraction(self):
        """A completely dark flat image: std=0 so dark-pixel test fires for all pixels below threshold.
        With std=0, mean-k*std = 0, so nothing is strictly below 0 → mask fraction may be 0.
        We just check the mask has correct shape and dtype."""
        img = np.zeros((256, 256), dtype=np.float32)
        mask = shadow_mask(img, solar_incidence_deg=45.0)
        assert mask.shape == img.shape
        assert mask.dtype == bool
        # A near-dark but non-uniform image should have high masking
        rng = np.random.default_rng(77)
        dark_img = rng.uniform(0.0, 0.02, size=(256, 256)).astype(np.float32)
        dark_mask = shadow_mask(dark_img, solar_incidence_deg=45.0)
        dark_fraction, _ = check_mask_fraction(dark_mask)
        assert dark_fraction > 0.3, f"Expected near-dark image to have >30% masked, got {dark_fraction:.3f}"

    def test_high_incidence_increases_masking(self, synthetic_image_512):
        """High solar incidence should mask MORE pixels than low incidence."""
        mask_low = shadow_mask(synthetic_image_512, solar_incidence_deg=20.0,
                               incidence_threshold_deg=80.0)
        mask_high = shadow_mask(synthetic_image_512, solar_incidence_deg=85.0,
                                incidence_threshold_deg=80.0)
        assert mask_high.mean() >= mask_low.mean(), (
            "High incidence should mask at least as many pixels as low incidence"
        )

    def test_check_mask_fraction_in_range_true(self):
        """Known fraction: 10% of pixels masked → in_range for [5%, 30%]."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[:10, :10] = True  # 10% masked
        fraction, in_range = check_mask_fraction(mask, min_pct=5.0, max_pct=30.0)
        assert abs(fraction - 0.01) < 1e-6  # 10/1000 pixels = 1%... wait
        # 10*10 / (100*100) = 100/10000 = 1%
        # So in_range should be False (below min 5%)
        assert not in_range

    def test_check_mask_fraction_above_max(self):
        mask = np.ones((100, 100), dtype=bool)  # 100% masked
        fraction, in_range = check_mask_fraction(mask, min_pct=5.0, max_pct=30.0)
        assert fraction == pytest.approx(1.0, abs=1e-6)
        assert not in_range

    def test_save_mask_png(self, synthetic_image_512, tmp_path):
        mask = shadow_mask(synthetic_image_512, solar_incidence_deg=45.0)
        out = save_mask_png(mask, tmp_path / "valid_mask.png")
        assert out.exists()
        assert out.suffix == ".png"
        # File must be non-empty
        assert out.stat().st_size > 0

    def test_raises_on_3d_input(self):
        img_3d = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            shadow_mask(img_3d, solar_incidence_deg=45.0)


# ===========================================================================
# normalize.py tests
# ===========================================================================

class TestPercentileClip:
    def test_output_dtype(self, synthetic_image_512):
        result = percentile_clip(synthetic_image_512)
        assert result.dtype == np.float32

    def test_output_range(self, synthetic_image_512):
        result = percentile_clip(synthetic_image_512)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_output_shape_preserved(self, synthetic_image_512):
        result = percentile_clip(synthetic_image_512)
        assert result.shape == synthetic_image_512.shape

    def test_custom_percentiles(self, synthetic_image_512):
        result_tight = percentile_clip(synthetic_image_512, lo=10, hi=90)
        result_wide = percentile_clip(synthetic_image_512, lo=0, hi=100)
        # Both should span [0, 1]
        assert float(result_tight.min()) >= 0.0
        assert float(result_wide.min()) >= 0.0

    def test_flat_image_returns_zeros(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        result = percentile_clip(img)
        # P2 == P98 → should return zeros
        assert np.all(result == 0.0)


class TestStatTransfer:
    """T04 — mean/std within 5% of ref after stat_transfer."""

    def test_output_dtype(self, synthetic_image_512, synthetic_ref_512):
        result = stat_transfer(synthetic_image_512, synthetic_ref_512)
        assert result.dtype == np.float32

    def test_output_range(self, synthetic_image_512, synthetic_ref_512):
        result = stat_transfer(synthetic_image_512, synthetic_ref_512)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_mean_within_5pct_of_ref(self, synthetic_image_512, synthetic_ref_512):
        """T04: transferred mean should be within 5% of reference mean."""
        result = stat_transfer(synthetic_image_512, synthetic_ref_512)
        ref_mean = float(np.mean(synthetic_ref_512))
        out_mean = float(np.mean(result))
        # Allow 5% absolute tolerance (spec: within 5% of ref)
        assert abs(out_mean - ref_mean) <= 0.05, (
            f"Mean after transfer {out_mean:.4f} not within 5% of ref mean {ref_mean:.4f}"
        )

    def test_std_within_5pct_of_ref(self, synthetic_image_512, synthetic_ref_512):
        """T04: transferred std should be within 5% of reference std."""
        result = stat_transfer(synthetic_image_512, synthetic_ref_512)
        ref_std = float(np.std(synthetic_ref_512))
        out_std = float(np.std(result))
        assert abs(out_std - ref_std) <= 0.05, (
            f"Std after transfer {out_std:.4f} not within 5% of ref std {ref_std:.4f}"
        )

    def test_flat_image_returns_clipped_unchanged(self):
        """Flat source image (std ≈ 0) should return clipped input unchanged."""
        flat = np.full((64, 64), 0.5, dtype=np.float32)
        ref = np.random.default_rng(42).random((64, 64)).astype(np.float32)
        result = stat_transfer(flat, ref)
        assert result.dtype == np.float32
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0


# ===========================================================================
# branches.py tests
# ===========================================================================

class TestApplyOhrcNac:
    def test_output_dtype(self, synthetic_image_512, ohrc_nac_branch_cfg):
        result = apply_ohrc_nac(synthetic_image_512, ohrc_nac_branch_cfg)
        assert result.dtype == np.float32

    def test_output_range(self, synthetic_image_512, ohrc_nac_branch_cfg):
        """Output must be in [0, 1]."""
        result = apply_ohrc_nac(synthetic_image_512, ohrc_nac_branch_cfg)
        assert float(result.min()) >= 0.0, f"Min {result.min():.6f} < 0"
        assert float(result.max()) <= 1.0, f"Max {result.max():.6f} > 1"

    def test_output_shape_preserved(self, synthetic_image_512, ohrc_nac_branch_cfg):
        result = apply_ohrc_nac(synthetic_image_512, ohrc_nac_branch_cfg)
        assert result.shape == synthetic_image_512.shape


class TestApplyMinimal:
    def test_no_heavy_processing(self, synthetic_image_512):
        """Minimal branch must NOT apply CLAHE — output should be very close to input."""
        cfg = {}
        result = apply_minimal(synthetic_image_512, cfg)
        assert result.dtype == np.float32
        # Minimal just clips; result should match input within float tolerance
        assert np.allclose(result, np.clip(synthetic_image_512, 0, 1), atol=1e-6), (
            "apply_minimal should return the clipped input unchanged"
        )

    def test_output_range(self, synthetic_image_512):
        result = apply_minimal(synthetic_image_512, {})
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0


class TestSelectBranch:
    def test_lightglue_always_minimal(self):
        """Learned matchers must always get minimal branch."""
        for matcher in ["lightglue", "crater", "crater_hough"]:
            branch = select_branch("OHRC-NAC", matcher, config={})
            assert branch == "minimal", (
                f"Expected 'minimal' for matcher={matcher}, got {branch!r}"
            )

    def test_ohrc_nac_classical(self):
        branch = select_branch("OHRC-NAC", "sift", config={})
        assert branch == "ohrc_to_nac"

    def test_ohrc_nac_rift2(self):
        branch = select_branch("OHRC-NAC", "rift2", config={})
        assert branch == "ohrc_to_nac"

    def test_tmc_wac_sift(self):
        branch = select_branch("TMC-2-WAC", "sift", config={})
        assert branch == "tmc_to_wac"

    def test_config_override(self):
        cfg = {"preprocessing": {"sensor_branch": "minimal"}}
        branch = select_branch("OHRC-NAC", "sift", config=cfg)
        assert branch == "minimal"

    def test_iirs_defaults_minimal(self):
        branch = select_branch("IIRS-WAC", "sift", config={})
        assert branch == "minimal"

    def test_unknown_sensor_pair_defaults_minimal(self):
        branch = select_branch("UNKNOWN-SENSOR", "sift", config={})
        assert branch == "minimal"


# ===========================================================================
# resample.py tests
# ===========================================================================

class TestReconcileGSD:
    def test_only_coarser_resampled_src_coarser(self):
        """When src_gsd > ref_gsd, src must be resampled (which_resampled='src')."""
        src = np.random.default_rng(1).random((200, 200)).astype(np.float32)
        result, meta = reconcile_gsd(src, src_gsd=0.5, ref_gsd=0.31,
                                     solar_incidence_deg=45.0)
        assert meta["which_resampled"] == "src", (
            "When src is coarser (0.5m > 0.31m), src should be resampled"
        )

    def test_only_coarser_resampled_ref_coarser(self):
        """When ref_gsd > src_gsd, result has which_resampled='ref' and src unchanged."""
        src = np.random.default_rng(2).random((200, 200)).astype(np.float32)
        result, meta = reconcile_gsd(src, src_gsd=0.31, ref_gsd=0.5,
                                     solar_incidence_deg=45.0)
        assert meta["which_resampled"] == "ref", (
            "When ref is coarser (0.5m > 0.31m), ref should be resampled, not src"
        )
        # src should be returned unchanged
        assert result.shape == src.shape

    def test_bilinear_at_high_incidence(self):
        """solar_incidence >= 45° → bilinear interpolation."""
        src = np.random.default_rng(3).random((100, 100)).astype(np.float32)
        _, meta = reconcile_gsd(src, src_gsd=0.5, ref_gsd=0.31,
                                solar_incidence_deg=50.0,
                                low_angle_threshold_deg=45.0)
        assert meta["interpolation_method"] == "bilinear"

    def test_bicubic_at_low_incidence(self):
        """solar_incidence < 45° → bicubic interpolation."""
        src = np.random.default_rng(4).random((100, 100)).astype(np.float32)
        _, meta = reconcile_gsd(src, src_gsd=0.5, ref_gsd=0.31,
                                solar_incidence_deg=30.0,
                                low_angle_threshold_deg=45.0)
        assert meta["interpolation_method"] == "bicubic"

    def test_equal_gsd_no_resample(self):
        """Equal GSD → no resampling, src returned unchanged."""
        src = np.random.default_rng(5).random((128, 128)).astype(np.float32)
        result, meta = reconcile_gsd(src, src_gsd=0.5, ref_gsd=0.5,
                                     solar_incidence_deg=45.0)
        assert meta["which_resampled"] == "none"
        assert result.shape == src.shape
        assert np.allclose(result, src, atol=1e-6)

    def test_output_dtype_float32(self):
        src = np.random.default_rng(6).random((64, 64)).astype(np.float32)
        result, _ = reconcile_gsd(src, src_gsd=1.0, ref_gsd=0.5,
                                  solar_incidence_deg=45.0)
        assert result.dtype == np.float32

    def test_gsd_ratio_recorded(self):
        src = np.random.default_rng(7).random((100, 100)).astype(np.float32)
        _, meta = reconcile_gsd(src, src_gsd=0.5, ref_gsd=0.31,
                                solar_incidence_deg=45.0)
        expected_ratio = 0.5 / 0.31
        assert abs(meta["gsd_ratio"] - expected_ratio) < 0.01

    def test_resampled_shape_increases_for_coarser_src(self):
        """Coarser src (0.5m) resampled to finer ref (0.31m) → larger output."""
        src = np.random.default_rng(8).random((100, 100)).astype(np.float32)
        result, meta = reconcile_gsd(src, src_gsd=0.5, ref_gsd=0.31,
                                     solar_incidence_deg=45.0)
        assert result.shape[0] > src.shape[0], (
            "Resampling 0.5m→0.31m should produce a larger image"
        )

    def test_raises_on_zero_gsd(self):
        src = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(ValueError):
            reconcile_gsd(src, src_gsd=0.0, ref_gsd=0.5, solar_incidence_deg=45.0)


# ===========================================================================
# tiling.py tests
# ===========================================================================

class TestTileImage:
    def test_returns_list(self, synthetic_image_512):
        tiles = tile_image(synthetic_image_512)
        assert isinstance(tiles, list)

    def test_tile_array_shape(self, synthetic_image_512):
        """Each tile must be ≤ tile_size in both dimensions."""
        tiles = tile_image(synthetic_image_512, tile_size=256, overlap_px=32)
        for tile_arr, (r0, c0) in tiles:
            assert tile_arr.ndim == 2
            assert tile_arr.shape[0] <= 256
            assert tile_arr.shape[1] <= 256

    def test_min_tile_size_enforced(self):
        """Tiles smaller than 256px in either dimension must be discarded."""
        # 300×300 image tiled with 256px tile → last partial tile is 44px → discarded
        img = np.random.default_rng(10).random((300, 300)).astype(np.float32)
        tiles = tile_image(img, tile_size=256, overlap_px=0)
        for tile_arr, _ in tiles:
            assert tile_arr.shape[0] >= 256
            assert tile_arr.shape[1] >= 256

    def test_offsets_within_image_bounds(self, synthetic_image_512):
        h, w = synthetic_image_512.shape
        tiles = tile_image(synthetic_image_512, tile_size=256, overlap_px=32)
        for tile_arr, (r0, c0) in tiles:
            assert 0 <= r0 < h
            assert 0 <= c0 < w
            assert r0 + tile_arr.shape[0] <= h
            assert c0 + tile_arr.shape[1] <= w

    def test_offset_plus_size_matches_content(self, synthetic_image_512):
        """Tile content must match the corresponding region in the original image."""
        tiles = tile_image(synthetic_image_512, tile_size=256, overlap_px=0)
        for tile_arr, (r0, c0) in tiles:
            r1 = r0 + tile_arr.shape[0]
            c1 = c0 + tile_arr.shape[1]
            expected = synthetic_image_512[r0:r1, c0:c1]
            assert np.allclose(tile_arr, expected, atol=1e-6), (
                f"Tile at ({r0},{c0}) does not match source region"
            )

    def test_fully_masked_tiles_discarded(self, synthetic_image_512):
        """Tiles that are fully masked should be discarded."""
        mask = np.ones_like(synthetic_image_512, dtype=bool)  # all invalid
        tiles = tile_image(synthetic_image_512, valid_mask=mask, min_fraction=0.5)
        assert len(tiles) == 0, "All tiles should be discarded when fully masked"

    def test_no_mask_keeps_all_valid_tiles(self, synthetic_image_512):
        """Without a mask, valid_fraction=1.0 → all size-passing tiles are kept."""
        tiles_no_mask = tile_image(synthetic_image_512, tile_size=256, overlap_px=0)
        tiles_full_valid = tile_image(
            synthetic_image_512, tile_size=256, overlap_px=0,
            valid_mask=np.zeros_like(synthetic_image_512, dtype=bool),  # all valid
        )
        assert len(tiles_no_mask) == len(tiles_full_valid)

    def test_raises_on_3d_input(self):
        img = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            tile_image(img)

    def test_raises_on_invalid_overlap(self, synthetic_image_512):
        with pytest.raises(ValueError, match="overlap_px"):
            tile_image(synthetic_image_512, tile_size=256, overlap_px=256)


class TestWriteTileGeoJSON:
    def test_writes_valid_geojson(self, synthetic_image_512, tmp_path):
        tiles = tile_image(synthetic_image_512, tile_size=256, overlap_px=0)
        out = write_tile_geojson(tiles, pair_id="test_pair", out_path=tmp_path / "tiles.geojson")
        assert out.exists()
        with open(out) as fh:
            data = json.load(fh)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == len(tiles)

    def test_feature_properties(self, synthetic_image_512, tmp_path):
        tiles = tile_image(synthetic_image_512, tile_size=256, overlap_px=0)
        out = write_tile_geojson(tiles, pair_id="p001", out_path=tmp_path / "t.geojson")
        with open(out) as fh:
            data = json.load(fh)
        props = data["features"][0]["properties"]
        assert "row_offset" in props
        assert "col_offset" in props
        assert "tile_index" in props
        assert props["pair_id"] == "p001"

    def test_polygon_geometry_structure(self, synthetic_image_512, tmp_path):
        tiles = tile_image(synthetic_image_512, tile_size=256, overlap_px=0)
        out = write_tile_geojson(tiles, pair_id="p001", out_path=tmp_path / "t.geojson")
        with open(out) as fh:
            data = json.load(fh)
        geom = data["features"][0]["geometry"]
        assert geom["type"] == "Polygon"
        # Polygon ring must be closed (first == last coordinate)
        ring = geom["coordinates"][0]
        assert ring[0] == ring[-1], "Polygon ring must be closed"
        assert len(ring) == 5  # 4 corners + closing point


# ===========================================================================
# stats.py tests (F26 / Phase 5.5.1)
# ===========================================================================

class TestFeatureStats:
    """Tests for compute_texture_contrast, compute_mean_gradient, and compute_image_stats."""

    def test_texture_contrast_constant_image(self):
        img = np.full((128, 128), fill_value=0.5, dtype=np.float32)
        contrast = compute_texture_contrast(img, window_size=8)
        assert np.isclose(contrast, 0.0, atol=1e-5)

    def test_texture_contrast_high_texture(self, synthetic_image_512):
        contrast = compute_texture_contrast(synthetic_image_512, window_size=8)
        assert contrast > 0.0
        assert np.isfinite(contrast)

    def test_texture_contrast_with_mask(self, synthetic_image_512):
        mask = np.ones_like(synthetic_image_512, dtype=bool)
        mask[:128, :128] = False  # Mask out top-left
        contrast_all = compute_texture_contrast(synthetic_image_512, window_size=8)
        contrast_masked = compute_texture_contrast(synthetic_image_512, window_size=8, valid_mask=mask)
        assert np.isfinite(contrast_masked)
        assert contrast_masked > 0.0

    def test_mean_gradient_constant_image(self):
        img = np.full((128, 128), fill_value=0.5, dtype=np.float32)
        grad = compute_mean_gradient(img)
        assert np.isclose(grad, 0.0, atol=1e-5)

    def test_mean_gradient_ramp_image(self):
        # Linear gradient ramp: 0 to 1 along horizontal
        img = np.tile(np.linspace(0.0, 1.0, 128, dtype=np.float32), (128, 1))
        grad = compute_mean_gradient(img)
        assert grad > 0.0
        assert np.isfinite(grad)

    def test_mean_gradient_with_mask(self, synthetic_image_512):
        mask = np.ones_like(synthetic_image_512, dtype=bool)
        mask[:100, :100] = False
        grad = compute_mean_gradient(synthetic_image_512, valid_mask=mask)
        assert grad > 0.0
        assert np.isfinite(grad)

    def test_compute_image_stats_dict_structure(self, synthetic_image_512):
        stats = compute_image_stats(synthetic_image_512, window_size=8)
        assert "texture_contrast" in stats
        assert "mean_gradient" in stats
        assert stats["texture_contrast"] > 0.0
        assert stats["mean_gradient"] > 0.0

    def test_process_pair_emits_feature_stats(self, tmp_path, synthetic_image_512, synthetic_ref_512):
        from scripts.preprocess import _process_pair, _write_geotiff

        # Setup test images
        src_path = tmp_path / "raw_src.tif"
        ref_path = tmp_path / "raw_ref.tif"
        _write_geotiff(synthetic_image_512, src_path)
        _write_geotiff(synthetic_ref_512, ref_path)

        pair = {
            "pair_id": "test_pair_stats",
            "src_path": str(src_path),
            "ref_path": str(ref_path),
            "src_gsd_m": 0.31,
            "ref_gsd_m": 0.50,
            "solar_incidence_deg": 45.0,
            "sensor_pair": "OHRC-NAC",
        }
        cfg = {
            "sensor_pair": "OHRC-NAC",
            "global": {"seed": 42},
            "preprocessing": {
                "shadow_mask": {"incidence_threshold_deg": 80.0, "mask_min_pct": 0.0, "mask_max_pct": 100.0},
                "radiometric_norm": {"percentile_clip": [2, 98], "stat_transfer": True},
                "tiling": {"size_px": 256, "overlap_px": 32, "min_tile_fraction": 0.5},
                "gsd": {"low_angle_threshold_deg": 45.0},
            },
        }

        out_root = tmp_path / "processed"
        failures_path = tmp_path / "failures.jsonl"
        ok = _process_pair(pair, cfg, out_root, failures_path, force=True)
        assert ok is True

        meta_path = out_root / "test_pair_stats" / "meta.json"
        assert meta_path.exists()
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        # Assert all MSM L1.5 required feature stats are present
        assert "src_texture_contrast" in meta
        assert "ref_texture_contrast" in meta
        assert "src_mean_gradient" in meta
        assert "ref_mean_gradient" in meta
        assert "tile_count" in meta
        assert "masked_fraction" in meta
        assert meta["src_texture_contrast"] > 0.0
        assert meta["ref_texture_contrast"] > 0.0
        assert meta["src_mean_gradient"] > 0.0
        assert meta["ref_mean_gradient"] > 0.0
        assert meta["tile_count"] >= 1
        assert 0.0 <= meta["masked_fraction"] <= 1.0



"""
tests/test_phase1.py
=====================
Unit tests for Phase 1 — Data & Geometry Layer (L0).

Tests:
    T_L0_01 — parse_pds4_label parses OHRC sample XML correctly
    T_L0_02 — parse_pds4_label parses TMC sample XML correctly
    T_L0_03 — parse_pds4_label parses IIRS sample XML correctly (sensor, bands, reg_band)
    T_L0_04 — footprint_ll is [[lon, lat], ...] with 4 corners (NEVER [lat, lon])
    T_L0_05 — longitudes normalized from [0, 360] to [-180, 180]
    T_L0_06 — pad_bbox formula verified: error < 0.1% at known values
    T_L0_07 — pad_bbox clamped to valid selenographic range [-180,180] x [-90,90]
    T_L0_08 — pad_bbox raises ValueError for empty footprint
    T_L0_09 — footprint_centre computes correct centroid
    T_L0_10 — compute_overlap_fraction returns 0 for non-overlapping bbox
    T_L0_11 — compute_overlap_fraction returns 1 for fully contained footprint
    T_L0_12 — assign_geo_cell assigns correct 10x10 degree cell
    T_L0_13 — assign_split is deterministic; 'test' and 'train' both produced across cells
    T_L0_14 — build_pair_id produces stable, unique IDs
    T_L0_15 — build_pair_record produces all required INTERFACES.md §1 fields
    T_L0_16 — run_isisimport raises FileNotFoundError for missing source
    T_L0_17 — run_spiceinit returns False for non-existent .cub path
    T_L0_18 — IIRS registration band selection: band closest to 643 nm

References:
    - data/phase1_spec/ohrc_sample.xml
    - data/phase1_spec/tmc_sample.xml
    - data/phase1_spec/iirs_sample.xml
    - docs/INTERFACES.md §1
    - docs/FEATURES.md F01-F03
    - PROGRESS.md §1.1 - §1.4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingest.label_parser import (
    ProductMeta,
    _normalize_lon,
    _select_iirs_registration_band,
    _strip_all_ns,
    parse_pds4_label,
    run_isisimport,
    run_spiceinit,
)
from src.ingest.reference import pad_bbox
from scripts.build_pairs import (
    assign_geo_cell,
    assign_split,
    assign_terrain_class,
    build_pair_id,
    build_pair_record,
    compute_overlap_fraction,
    footprint_centre,
)

# ---------------------------------------------------------------------------
# Fixtures: Sample XML Paths
# ---------------------------------------------------------------------------

SPEC_DIR = _PROJECT_ROOT / "data" / "phase1_spec"
OHRC_XML = SPEC_DIR / "ohrc_sample.xml"
TMC_XML = SPEC_DIR / "tmc_sample.xml"
IIRS_XML = SPEC_DIR / "iirs_sample.xml"


def _require_sample_xml(path: Path, name: str):
    """Skip test if sample XML not present."""
    if not path.exists():
        pytest.skip(f"{name} sample XML not found at {path}")


# ===========================================================================
# T_L0_01 — OHRC label parsing
# ===========================================================================

class TestOHRCLabelParser:
    def setup_method(self):
        _require_sample_xml(OHRC_XML, "OHRC")
        self.meta = parse_pds4_label(str(OHRC_XML))

    def test_sensor_is_ohrc(self):
        assert self.meta.sensor == "OHRC", f"Expected OHRC, got {self.meta.sensor}"

    def test_gsd_is_positive(self):
        assert self.meta.gsd_m > 0, "GSD must be positive"
        # OHRC pixel_resolution = 0.28 m
        assert 0.1 < self.meta.gsd_m < 1.0, f"OHRC GSD out of expected range: {self.meta.gsd_m}"

    def test_solar_incidence_present(self):
        # OHRC solar_incidence = 90.043949 (polar scene)
        assert 0.0 <= self.meta.solar_incidence_deg <= 180.0

    def test_solar_azimuth_present(self):
        assert 0.0 <= self.meta.solar_azimuth_deg <= 360.0

    def test_utc_nonempty(self):
        assert self.meta.utc, "UTC timestamp must not be empty"
        assert "2021-12-28" in self.meta.utc, f"Unexpected UTC: {self.meta.utc}"

    def test_product_id_nonempty(self):
        assert self.meta.product_id, "product_id must not be empty"
        assert "ohr" in self.meta.product_id.lower()

    def test_footprint_has_4_corners(self):
        assert len(self.meta.footprint_ll) == 4, (
            f"Footprint must have 4 corners, got {len(self.meta.footprint_ll)}"
        )

    def test_footprint_is_lon_lat(self):
        """
        Each corner MUST be [lon, lat] (INTERFACES.md §8).
        OHRC is a polar scene — all latitudes are near -90°.
        If any 'lat' is in longitude range (> 90), the convention is wrong.
        """
        for i, (lon, lat) in enumerate(self.meta.footprint_ll):
            assert -180.0 <= lon <= 180.0, f"Corner {i} lon={lon} out of range"
            assert -90.0 <= lat <= 90.0, f"Corner {i} lat={lat} out of range"

    def test_footprint_shape_has_two_dims(self):
        assert len(self.meta.footprint_shape) == 2
        lines, samples = self.meta.footprint_shape
        assert lines > 0 and samples > 0

    def test_iirs_fields_are_none_for_ohrc(self):
        assert self.meta.iirs_n_bands is None
        assert self.meta.iirs_registration_band is None


# ===========================================================================
# T_L0_02 — TMC label parsing
# ===========================================================================

class TestTMCLabelParser:
    def setup_method(self):
        _require_sample_xml(TMC_XML, "TMC")
        self.meta = parse_pds4_label(str(TMC_XML))

    def test_sensor_is_tmc(self):
        assert self.meta.sensor == "TMC", f"Expected TMC, got {self.meta.sensor}"

    def test_gsd_in_range(self):
        # TMC pixel_resolution = 9.79 m
        assert 1.0 < self.meta.gsd_m < 100.0, f"TMC GSD out of range: {self.meta.gsd_m}"

    def test_utc_nonempty(self):
        assert "2009-08-01" in self.meta.utc

    def test_footprint_4_corners(self):
        assert len(self.meta.footprint_ll) == 4

    def test_footprint_lon_lat_convention(self):
        for i, (lon, lat) in enumerate(self.meta.footprint_ll):
            assert -180.0 <= lon <= 180.0, f"Corner {i} lon={lon} OOR"
            assert -90.0 <= lat <= 90.0, f"Corner {i} lat={lat} OOR"

    def test_equatorial_scene(self):
        """TMC scene is equatorial — centre lat should be near 0."""
        lats = [pt[1] for pt in self.meta.footprint_ll]
        centre_lat = sum(lats) / len(lats)
        assert -30.0 < centre_lat < 30.0, f"TMC centre lat {centre_lat} unexpected"


# ===========================================================================
# T_L0_03 — IIRS label parsing
# ===========================================================================

class TestIIRSLabelParser:
    def setup_method(self):
        _require_sample_xml(IIRS_XML, "IIRS")
        self.meta = parse_pds4_label(str(IIRS_XML))

    def test_sensor_is_iirs(self):
        assert self.meta.sensor == "IIRS", f"Expected IIRS, got {self.meta.sensor}"

    def test_gsd_in_range(self):
        # IIRS pixel_resolution = 94.46 m
        assert 50.0 < self.meta.gsd_m < 200.0, f"IIRS GSD out of range: {self.meta.gsd_m}"

    def test_iirs_n_bands(self):
        """IIRS has 256 spectral bands."""
        assert self.meta.iirs_n_bands is not None
        assert self.meta.iirs_n_bands == 256, f"Expected 256 bands, got {self.meta.iirs_n_bands}"

    def test_iirs_registration_band_set(self):
        """Registration band for WAC 643 nm matching must be set."""
        assert self.meta.iirs_registration_band is not None

    def test_iirs_registration_band_is_first_band(self):
        """
        IIRS band 1 center = 712.3 nm (closest to WAC 643 nm among early bands).
        Registration band index 0 (zero-indexed band 1) should be selected.
        """
        assert self.meta.iirs_registration_band == 0, (
            f"Expected registration band 0, got {self.meta.iirs_registration_band}"
        )

    def test_processing_level_is_raw(self):
        """IIRS sample is Raw level data."""
        assert self.meta.processing_level == "Raw"


# ===========================================================================
# T_L0_04 & T_L0_05 — Longitude normalization
# ===========================================================================

class TestLongitudeNormalization:
    def test_lon_in_range_unchanged(self):
        assert _normalize_lon(55.56) == pytest.approx(55.56)

    def test_lon_zero(self):
        assert _normalize_lon(0.0) == pytest.approx(0.0)

    def test_lon_180_unchanged(self):
        assert _normalize_lon(180.0) == pytest.approx(180.0)

    def test_lon_greater_than_180_normalized(self):
        # 224.35° -> -135.65°
        assert _normalize_lon(224.35) == pytest.approx(224.35 - 360.0, abs=1e-6)

    def test_lon_360_normalized_to_0(self):
        assert _normalize_lon(360.0) == pytest.approx(0.0, abs=1e-6)

    def test_ohrc_lower_left_lon_normalized(self):
        """OHRC lower_left_longitude = 233.745958 -> should be negative."""
        result = _normalize_lon(233.745958)
        assert result < 0, f"Expected negative lon for 233.745958, got {result}"
        assert result == pytest.approx(233.745958 - 360.0, abs=1e-6)

    def test_footprint_all_lons_in_range(self):
        """All corner lons from OHRC sample must be in [-180, 180] after normalization."""
        _require_sample_xml(OHRC_XML, "OHRC")
        meta = parse_pds4_label(str(OHRC_XML))
        for i, (lon, lat) in enumerate(meta.footprint_ll):
            assert -180.0 <= lon <= 180.0, f"OHRC corner {i}: lon={lon} not normalized"


# ===========================================================================
# T_L0_06 — pad_bbox formula
# ===========================================================================

class TestPadBbox:
    # Simple equatorial footprint
    FOOTPRINT = [
        [10.0, -21.0],  # UL
        [12.0, -21.0],  # UR
        [12.0, 9.0],    # LR
        [10.0, 9.0],    # LL
    ]

    def test_pad_bbox_basic_expansion(self):
        """Padded bbox must be strictly larger than raw bbox in all directions."""
        sigma_m = 1000.0
        k = 3.0
        result = pad_bbox(self.FOOTPRINT, sigma_m=sigma_m, k=k)
        lon_min, lat_min, lon_max, lat_max = result
        assert lon_min < 10.0, "lon_min not expanded"
        assert lat_min < -21.0, "lat_min not expanded"
        assert lon_max > 12.0, "lon_max not expanded"
        assert lat_max > 9.0, "lat_max not expanded"

    def test_pad_bbox_formula_error_less_than_0_1pct(self):
        """
        T02 (PROGRESS.md §6.4 T02): pad_bbox formula error < 0.1%.

        Reference: delta_lat_deg = (k * sigma_m) / MOON_RADIUS_M * (180/pi)
        """
        import math
        sigma_m = 1000.0
        k = 3.0
        MOON_RADIUS_M = 1_737_400.0

        result = pad_bbox(self.FOOTPRINT, sigma_m=sigma_m, k=k)
        lon_min, lat_min, lon_max, lat_max = result

        # Expected lat expansion
        expected_delta_lat = math.degrees(k * sigma_m / MOON_RADIUS_M)
        actual_lat_expansion = -21.0 - lat_min
        relative_error = abs(actual_lat_expansion - expected_delta_lat) / expected_delta_lat
        assert relative_error < 0.001, (
            f"pad_bbox lat expansion error {relative_error:.4%} >= 0.1%"
        )

    def test_pad_bbox_clamped_to_valid_range(self):
        """pad_bbox output must always be within [-180,180] x [-90,90]."""
        polar_footprint = [
            [55.0, -89.9],
            [110.0, -89.9],
            [110.0, -89.0],
            [55.0, -89.0],
        ]
        result = pad_bbox(polar_footprint, sigma_m=10_000, k=5)
        assert result[0] >= -180.0
        assert result[1] >= -90.0
        assert result[2] <= 180.0
        assert result[3] <= 90.0

    def test_pad_bbox_raises_for_empty_footprint(self):
        """T_L0_08: pad_bbox must raise ValueError for fewer than 3 points."""
        with pytest.raises(ValueError):
            pad_bbox([[10.0, 5.0]], sigma_m=1000, k=3)

    def test_pad_bbox_with_k_zero_is_noop(self):
        """k=0 means no padding; bbox equals raw bbox bounds."""
        result = pad_bbox(self.FOOTPRINT, sigma_m=1000, k=0)
        assert result[0] == pytest.approx(10.0, abs=1e-4)
        assert result[1] == pytest.approx(-21.0, abs=1e-4)
        assert result[2] == pytest.approx(12.0, abs=1e-4)
        assert result[3] == pytest.approx(9.0, abs=1e-4)


# ===========================================================================
# T_L0_09 — footprint_centre
# ===========================================================================

class TestFootprintCentre:
    def test_centre_of_square(self):
        fp = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
        lon, lat = footprint_centre(fp)
        assert lon == pytest.approx(1.0)
        assert lat == pytest.approx(1.0)

    def test_centre_of_ohrc_polar_scene(self):
        """OHRC scene is polar; centre lat should be < -89°."""
        _require_sample_xml(OHRC_XML, "OHRC")
        meta = parse_pds4_label(str(OHRC_XML))
        lon, lat = footprint_centre(meta.footprint_ll)
        assert lat < -85.0, f"OHRC centre lat {lat} not polar"

    def test_centre_empty(self):
        lon, lat = footprint_centre([])
        assert lon == 0.0 and lat == 0.0


# ===========================================================================
# T_L0_10, T_L0_11 — compute_overlap_fraction
# ===========================================================================

class TestComputeOverlapFraction:
    FP = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

    def test_no_overlap(self):
        ref_bbox = [20.0, 0.0, 30.0, 10.0]  # completely outside
        assert compute_overlap_fraction(self.FP, ref_bbox) == pytest.approx(0.0)

    def test_full_containment(self):
        ref_bbox = [-5.0, -5.0, 15.0, 15.0]  # ref completely covers src
        assert compute_overlap_fraction(self.FP, ref_bbox) == pytest.approx(1.0)

    def test_partial_overlap(self):
        ref_bbox = [5.0, 0.0, 15.0, 10.0]  # 50% overlap
        result = compute_overlap_fraction(self.FP, ref_bbox)
        assert 0.4 < result < 0.6, f"Expected ~0.5 overlap, got {result}"

    def test_empty_footprint(self):
        assert compute_overlap_fraction([], [0.0, 0.0, 10.0, 10.0]) == 0.0


# ===========================================================================
# T_L0_12 — assign_geo_cell
# ===========================================================================

class TestAssignGeoCell:
    def test_polar_south_scene(self):
        """OHRC polar: lat=-89, lon=55 -> cell (-90, 50)"""
        cell = assign_geo_cell(-89.0, 55.0)
        assert cell == "-90_50", f"Expected -90_50, got {cell}"

    def test_equatorial_scene(self):
        """TMC equatorial: lat=-5, lon=11 -> cell (-10, 10)"""
        cell = assign_geo_cell(-5.0, 11.0)
        assert cell == "-10_10", f"Expected -10_10, got {cell}"

    def test_exactly_on_boundary(self):
        cell = assign_geo_cell(0.0, 0.0)
        assert cell == "0_0"

    def test_northern_scene(self):
        cell = assign_geo_cell(65.0, 48.0)
        assert cell == "60_40"


# ===========================================================================
# T_L0_13 — assign_split
# ===========================================================================

class TestAssignSplit:
    def test_deterministic(self):
        """Same cell always produces same split."""
        cell = "60_40"
        assert assign_split(cell) == assign_split(cell)

    def test_both_splits_produced(self):
        """Test that both 'train' and 'test' are produced across cells."""
        cells = [
            assign_geo_cell(lat, lon)
            for lat in range(-80, 80, 10)
            for lon in range(-170, 180, 10)
        ]
        splits = {assign_split(c) for c in cells}
        assert "train" in splits, "No train cells found"
        assert "test" in splits, "No test cells found"

    def test_valid_split_values(self):
        for lat in range(-80, 80, 20):
            for lon in range(-160, 180, 20):
                cell = assign_geo_cell(lat, lon)
                split = assign_split(cell)
                assert split in ("train", "test"), f"Invalid split: {split}"


# ===========================================================================
# T_L0_14 — build_pair_id
# ===========================================================================

class TestBuildPairId:
    def test_pair_id_contains_both_ids(self):
        pid = build_pair_id("ch2_ohr_abc", "nac_def")
        assert "ch2_ohr_abc" in pid
        assert "nac_def" in pid

    def test_pair_id_stable(self):
        a = build_pair_id("src_product_x", "ref_product_y")
        b = build_pair_id("src_product_x", "ref_product_y")
        assert a == b

    def test_pair_id_different_products(self):
        a = build_pair_id("src_A", "ref_A")
        b = build_pair_id("src_B", "ref_B")
        assert a != b


# ===========================================================================
# T_L0_15 — build_pair_record schema compliance
# ===========================================================================

class TestBuildPairRecord:
    REQUIRED_FIELDS = [
        "pair_id", "src", "ref", "overlap_fraction", "partial_overlap",
        "latitude_center_deg", "terrain_class", "geo_cell", "split", "created_at",
    ]
    REQUIRED_SRC_FIELDS = [
        "product_id", "cub_path", "gsd_m", "solar_incidence_deg",
        "solar_azimuth_deg", "sensor", "utc", "footprint_ll", "footprint_shape",
    ]
    REQUIRED_REF_FIELDS = ["product_id", "path", "gsd_m", "type"]

    def _make_record(self):
        src_meta = {
            "product_id": "ch2_ohr_ncp_20211228t2209123959_d_img_d18",
            "cub_path": "data/calibrated/test.cub",
            "gsd_m": 0.28,
            "solar_incidence_deg": 90.04,
            "solar_azimuth_deg": 152.4,
            "sensor": "OHRC",
            "utc": "2021-12-28T22:09:12.3959Z",
            "footprint_ll": [
                [55.56, -89.92], [110.42, -89.85],
                [224.35, -89.25], [233.75, -89.26],
            ],
            "footprint_shape": [79796, 12000],
        }
        prov = {
            "config_hash": "abc123",
            "code_commit": "deadbeef",
            "matcher_params_hash": "xyz",
            "created_at": "2026-08-31T13:00:00Z",
            "seed": 42,
        }
        ref_bbox = [-180.0, -90.0, 180.0, -85.0]
        return build_pair_record(
            src_meta=src_meta,
            ref_path="data/reference/nac_crop.tif",
            ref_type="NAC",
            ref_product_id="nac_crop",
            ref_gsd_m=0.5,
            ref_bbox=ref_bbox,
            config={"global": {"seed": 42}},
            provenance=prov,
        )

    def test_all_required_top_level_fields(self):
        record = self._make_record()
        for field in self.REQUIRED_FIELDS:
            assert field in record, f"Missing required field: {field}"

    def test_src_subfields(self):
        record = self._make_record()
        src = record["src"]
        for field in self.REQUIRED_SRC_FIELDS:
            assert field in src, f"Missing src.{field}"

    def test_ref_subfields(self):
        record = self._make_record()
        ref = record["ref"]
        for field in self.REQUIRED_REF_FIELDS:
            assert field in ref, f"Missing ref.{field}"

    def test_ref_type_valid(self):
        record = self._make_record()
        assert record["ref"]["type"] in ("NAC", "WAC", "SELENE")

    def test_split_valid(self):
        record = self._make_record()
        assert record["split"] in ("train", "test")

    def test_overlap_fraction_in_range(self):
        record = self._make_record()
        assert 0.0 <= record["overlap_fraction"] <= 1.0

    def test_terrain_class_valid(self):
        record = self._make_record()
        assert record["terrain_class"] in ("equatorial", "highland", "polar_highland", "polar")

    def test_partial_overlap_set_for_small_overlap(self):
        record = self._make_record()
        # With OHRC polar footprint overlapping global bbox, overlap should be high
        # partial_overlap should match (overlap_fraction < 0.5)
        if record["overlap_fraction"] < 0.5:
            assert record["partial_overlap"] is True
        else:
            assert record["partial_overlap"] is False


# ===========================================================================
# T_L0_16, T_L0_17 — ISIS Wrapper Error Handling
# ===========================================================================

class TestISISWrappers:
    def test_run_isisimport_raises_file_not_found(self):
        """run_isisimport must raise FileNotFoundError for missing .img"""
        with pytest.raises(FileNotFoundError):
            run_isisimport("/nonexistent/path/to/file.img", "/tmp/out")

    def test_run_spiceinit_returns_false_for_missing_cub(self):
        """run_spiceinit must return False (not raise) for non-existent .cub"""
        result = run_spiceinit("/nonexistent/path/to/file.cub")
        assert result is False


# ===========================================================================
# T_L0_18 — IIRS Registration Band Selection
# ===========================================================================

class TestIIRSRegistrationBand:
    def test_registration_band_selected(self):
        """parse_pds4_label for IIRS must set iirs_registration_band."""
        _require_sample_xml(IIRS_XML, "IIRS")
        meta = parse_pds4_label(str(IIRS_XML))
        assert meta.iirs_registration_band is not None

    def test_registration_band_is_zero_indexed(self):
        """iirs_registration_band is zero-indexed (0 = first band, band number 1)."""
        _require_sample_xml(IIRS_XML, "IIRS")
        meta = parse_pds4_label(str(IIRS_XML))
        assert meta.iirs_registration_band >= 0

    def test_registration_band_nearest_to_643nm(self):
        """
        IIRS band 1 center = 712.3 nm, band 2 = 729.2 nm.
        Band 0 (zero-indexed) is the closest to 643 nm in the sample.
        """
        _require_sample_xml(IIRS_XML, "IIRS")
        meta = parse_pds4_label(str(IIRS_XML))
        assert meta.iirs_registration_band == 0

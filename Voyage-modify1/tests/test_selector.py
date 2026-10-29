"""
tests/test_selector.py
======================
Unit and regression tests for the Matcher Selection Model (MSM) (L1.5 / S4.5).

Tests:
  - T13: Feature extraction invariance & determinism
  - T14: Hard-rule gating override (Crater density & GPU gate)
  - T15: Dual-threshold confidence routing logic
  - SelectorResult serialization to selector.json

References:
  - FEATURES.md F26
  - VALIDATION.md T13-T15, §9
  - PROGRESS.md §5.5.2 - §5.5.4
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.selector import (
    MSMFeatureVector,
    extract_features,
    vectorize_features,
    hash_features,
    SelectorResult,
    MatcherSelector,
    FEATURE_NAMES,
    MATCHER_NAMES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pair_record():
    return {
        "pair_id": "ohr_20200827T003010__nac_M123456789",
        "sensor_pair": "OHRC-NAC",
        "src_gsd_m": 0.31,
        "ref_gsd_m": 0.50,
        "latitude_center_deg": -85.2,
        "delta_azimuth_deg": 12.5,
        "terrain_class": "polar_highland",
        "crater_density_per_km2": 6.8,
        "overlap_fraction": 0.88,
    }


@pytest.fixture
def sample_meta_json():
    return {
        "pair_id": "ohr_20200827T003010__nac_M123456789",
        "masked_fraction": 0.12,
        "tile_count": 8,
        "src_texture_contrast": 24.5,
        "ref_texture_contrast": 22.1,
        "src_mean_gradient": 15.3,
        "ref_mean_gradient": 14.8,
    }


# ===========================================================================
# 1. Feature Extraction Tests (T13)
# ===========================================================================

class TestFeatureExtraction:
    def test_extract_all_13_features(self, sample_pair_record, sample_meta_json):
        feat = extract_features(sample_pair_record, sample_meta_json)
        assert isinstance(feat, MSMFeatureVector)
        assert feat.pair_id == "ohr_20200827T003010__nac_M123456789"
        assert feat.sensor_pair_enc == 0  # OHRC-NAC
        assert 0.0 < feat.gsd_ratio <= 1.0
        assert feat.latitude_abs == 85.2
        assert feat.delta_solar_azimuth == 12.5
        assert feat.terrain_class_enc == 2  # polar_highland
        assert feat.crater_density > 0.0
        assert feat.masked_fraction == 0.12
        assert feat.overlap_fraction == 0.88
        assert feat.src_texture_contrast == 24.5
        assert feat.ref_texture_contrast == 22.1
        assert feat.src_mean_gradient == 15.3
        assert feat.ref_mean_gradient == 14.8
        assert feat.tile_count == 8
        assert len(feat.feature_vector_hash) == 32  # MD5 hex length

    def test_vectorize_features_shape_and_order(self, sample_pair_record, sample_meta_json):
        feat = extract_features(sample_pair_record, sample_meta_json)
        arr = vectorize_features(feat)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (13,)
        assert arr.dtype == np.float32
        assert len(FEATURE_NAMES) == 13

    def test_feature_extraction_determinism(self, sample_pair_record, sample_meta_json):
        """T13 — Extracting features on identical records yields identical hash."""
        feat1 = extract_features(sample_pair_record, sample_meta_json)
        feat2 = extract_features(sample_pair_record, sample_meta_json)
        assert feat1.feature_vector_hash == feat2.feature_vector_hash
        assert np.array_equal(vectorize_features(feat1), vectorize_features(feat2))

    def test_feature_bounds_clamping(self):
        pair = {
            "pair_id": "p_edge",
            "src_gsd_m": 5.0,
            "ref_gsd_m": 0.5,
            "latitude_center_deg": -105.0,  # invalid lat, should clamp to 90
            "delta_azimuth_deg": 250.0,     # > 180, should wrap/clamp to 110
            "terrain_class": "unknown_terrain",
            "crater_density_per_km2": -5.0,
            "overlap_fraction": 1.5,        # > 1.0
        }
        meta = {
            "masked_fraction": 1.2,         # > 1.0
            "tile_count": -3,               # < 1
            "src_texture_contrast": -2.0,
            "ref_texture_contrast": -1.0,
            "src_mean_gradient": -4.0,
            "ref_mean_gradient": -3.0,
        }
        feat = extract_features(pair, meta)
        assert 0.0 < feat.gsd_ratio <= 1.0
        assert feat.latitude_abs <= 90.0
        assert 0.0 <= feat.delta_solar_azimuth <= 180.0
        assert feat.crater_density >= 0.0
        assert feat.masked_fraction <= 1.0
        assert feat.overlap_fraction <= 1.0
        assert feat.tile_count >= 1
        assert feat.src_texture_contrast >= 0.0
        assert feat.src_mean_gradient >= 0.0


# ===========================================================================
# 2. Hard-Rule Gating & Routing Tests (T14, T15)
# ===========================================================================

class DummyModel:
    """Mock model providing custom predict_proba for testing routing logic."""
    def __init__(self, prob_dist):
        self.prob_dist = np.array(prob_dist, dtype=np.float32)

    def predict_proba(self, X):
        return np.tile(self.prob_dist, (len(X), 1))


class TestMatcherSelectorRouting:
    def test_hard_rule_crater_density_gated_off(self, sample_pair_record, sample_meta_json):
        """T14 — Crater branch suppressed when crater density < tau_c."""
        pair = dict(sample_pair_record)
        pair["crater_density_per_km2"] = 1.0  # < tau_c (5.0)
        feat = extract_features(pair, sample_meta_json)

        # Model strongly predicts Crater (class 3)
        selector = MatcherSelector()
        selector.model = DummyModel([0.05, 0.05, 0.10, 0.80])
        selector.is_loaded = True

        result = selector.predict(feat)
        # Crater must be suppressed by hard gate
        assert result.all_probs["crater"] == 0.0
        assert result.selected_matcher != "crater"
        assert any("crater_density_below_tau_c" in r for r in result.hard_rules_applied)

    def test_dual_threshold_high_confidence_single_matcher(self, sample_pair_record, sample_meta_json):
        """T15 — P_max >= 0.65 dispatches single matcher only."""
        feat = extract_features(sample_pair_record, sample_meta_json)
        selector = MatcherSelector()
        selector.model = DummyModel([0.05, 0.10, 0.80, 0.05])  # LightGlue (class 2) = 0.80 >= 0.65
        selector.is_loaded = True

        result = selector.predict(feat)
        assert result.selected_matcher == "lightglue"
        assert result.fallback_matcher == "rift2"
        assert result.confidence == 0.80
        assert result.matchers_to_run == ["lightglue"]
        assert result.routing_reason == "high_confidence_single_matcher"

    def test_dual_threshold_medium_confidence_dual_matcher(self, sample_pair_record, sample_meta_json):
        """T15 — 0.40 <= P_max < 0.65 dispatches primary + fallback matchers."""
        feat = extract_features(sample_pair_record, sample_meta_json)
        selector = MatcherSelector()
        selector.model = DummyModel([0.15, 0.30, 0.50, 0.05])  # LightGlue = 0.50, RIFT2 = 0.30
        selector.is_loaded = True

        result = selector.predict(feat)
        assert result.selected_matcher == "lightglue"
        assert result.fallback_matcher == "rift2"
        assert result.matchers_to_run == ["lightglue", "rift2"]
        assert result.routing_reason == "medium_confidence_dual_matcher"

    def test_dual_threshold_low_confidence_safe_mode(self, sample_pair_record, sample_meta_json):
        """T15 — P_max < 0.40 triggers safe mode running all matchers."""
        feat = extract_features(sample_pair_record, sample_meta_json)
        selector = MatcherSelector()
        selector.model = DummyModel([0.28, 0.26, 0.26, 0.20])  # All below 0.40
        selector.is_loaded = True

        result = selector.predict(feat)
        assert len(result.matchers_to_run) == 4
        assert result.routing_reason == "low_confidence_safe_mode"

    def test_save_and_load_selector_json(self, tmp_path, sample_pair_record, sample_meta_json):
        feat = extract_features(sample_pair_record, sample_meta_json)
        selector = MatcherSelector()
        result = selector.predict(feat)

        out_path = tmp_path / "selector.json"
        saved = selector.save_result(result, out_path)
        assert saved.exists()

        with open(saved, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["pair_id"] == feat.pair_id
        assert "selected_matcher" in data
        assert "confidence" in data
        assert "fallback_matcher" in data
        assert "all_probs" in data
        assert "matchers_to_run" in data
        assert "routing_reason" in data
        assert "hard_rules_applied" in data
        assert "selector_version" in data
        assert "feature_vector_hash" in data

    def test_load_trained_model_and_predict(self, sample_pair_record, sample_meta_json):
        """Test loading actual trained model pickle from models/msm_v1.pkl."""
        model_path = Path("models/msm_v1.pkl")
        if not model_path.exists():
            pytest.skip("models/msm_v1.pkl not found")

        selector = MatcherSelector({"msm": {"model_path": str(model_path), "enabled": True}})
        assert selector.is_loaded is True

        feat = extract_features(sample_pair_record, sample_meta_json)
        result = selector.predict(feat)
        assert result.selected_matcher in MATCHER_NAMES
        assert result.fallback_matcher in MATCHER_NAMES
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.matchers_to_run) >= 1
        assert result.feature_vector_hash == feat.feature_vector_hash


# ===========================================================================
# 3. Training Pipeline Tests (F15 / F27)
# ===========================================================================

class TestMSMTrainingPipeline:
    def test_geo_cell_disjoint_cv_zero_leakage(self):
        """F15 / F27 — Verify GroupKFold ensures zero geo-cell leakage across folds."""
        from scripts.train_msm import train_model

        # Synthetic dataset with repeating geo-cells
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 13)).astype(np.float32)
        y = rng.integers(0, 4, size=20).astype(np.int32)
        groups = [f"cell_{i % 5}" for i in range(20)]  # 5 distinct cells

        model, stats = train_model(X, y, groups, n_splits=3, seed=42)
        assert stats["model_version"] == "msm_v1"
        assert stats["geo_cell_leakage_audit"] == "PASSED (0 overlap between CV folds)"
        assert stats["cv_mean_accuracy"] >= 0.0
        assert len(stats["feature_importance_split"]) == 13


# ===========================================================================
# 4. Benchmark MSM Pipeline Integration Tests
# ===========================================================================

class TestBenchmarkMSMIntegration:
    def test_benchmark_argparser_msm_mode(self):
        from scripts.benchmark import _parse_args
        args = _parse_args(["--pair", "data/pairs/manifest.jsonl", "--mode", "msm", "--msm-config", "configs/msm.yaml"])
        assert args.mode == "msm"
        assert args.msm_config == "configs/msm.yaml"


# ===========================================================================
# 5. Full Validation Protocol Tests (T13–T16, AC1–AC8)
# ===========================================================================

class TestMSMValidationProtocol:
    def test_t13_feature_extraction_invariance(self, sample_pair_record, sample_meta_json):
        """T13 — MSM feature extraction invariance under identical inputs."""
        feat1 = extract_features(sample_pair_record, sample_meta_json)
        feat2 = extract_features(sample_pair_record, sample_meta_json)
        assert feat1.feature_vector_hash == feat2.feature_vector_hash
        assert np.allclose(vectorize_features(feat1), vectorize_features(feat2))

    def test_t14_hard_rule_gating_override(self, sample_pair_record, sample_meta_json):
        """T14 — Hard rule overrides model output when conditions violated."""
        pair = dict(sample_pair_record)
        pair["crater_density_per_km2"] = 0.5  # Below tau_c
        feat = extract_features(pair, sample_meta_json)

        selector = MatcherSelector()
        selector.model = DummyModel([0.0, 0.0, 0.0, 1.0])  # Predicts crater
        selector.is_loaded = True

        res = selector.predict(feat)
        assert res.all_probs["crater"] == 0.0
        assert res.selected_matcher != "crater"

    def test_t15_dual_threshold_routing(self, sample_pair_record, sample_meta_json):
        """T15 — Dual-threshold confidence routing and safe mode logic."""
        feat = extract_features(sample_pair_record, sample_meta_json)
        selector = MatcherSelector()

        # High confidence
        selector.model = DummyModel([0.70, 0.10, 0.10, 0.10])
        selector.is_loaded = True
        res_high = selector.predict(feat)
        assert res_high.matchers_to_run == ["sift"]

        # Medium confidence
        selector.model = DummyModel([0.45, 0.35, 0.10, 0.10])
        res_med = selector.predict(feat)
        assert res_med.matchers_to_run == ["sift", "rift2"]

        # Safe mode (low confidence)
        selector.model = DummyModel([0.25, 0.25, 0.25, 0.25])
        res_low = selector.predict(feat)
        assert len(res_low.matchers_to_run) == 4

    def test_t16_disjoint_geocell_cv_leakage_audit(self, tmp_path):
        """T16 — Audit manifest and MSM dataset for geo-cell disjointness."""
        from src.evaluation.leakage_audit import run_audit
        manifest_path = Path("data/pairs/manifest.jsonl")
        if not manifest_path.exists():
            pytest.skip("manifest.jsonl not found")

        passed = run_audit(manifest_path, check_msm=True, msm_stats_path="models/msm_v1_stats.json")
        assert passed is True

    def test_msm_eval_suite_runs_and_passes(self, tmp_path):
        """Verify src/evaluation/msm_eval.py runs and meets AC1–AC8."""
        from src.evaluation.msm_eval import evaluate_msm_suite
        manifest_path = Path("data/pairs/manifest_phase7.jsonl") if Path("data/pairs/manifest_phase7.jsonl").exists() else Path("data/pairs/manifest.jsonl")
        model_path = Path("models/msm_v1.pkl")
        model_stats_path = Path("models/msm_v1_stats.json")

        if not manifest_path.exists() or not model_path.exists():
            pytest.skip("manifest or model not found")

        report = evaluate_msm_suite(
            manifest_path=manifest_path,
            results_dir=Path("results"),
            processed_dir=Path("data/processed"),
            model_path=model_path,
            model_stats_path=model_stats_path,
        )
        assert report["status"] == "PASSED"
        assert report["criteria"]["AC1_selector_accuracy"]["passed"] is True
        assert report["criteria"]["AC2_top2_accuracy"]["passed"] is True
        assert report["criteria"]["AC3_mean_rmse_degradation"]["passed"] is True
        assert report["criteria"]["AC4_max_rmse_degradation"]["passed"] is True
        assert report["criteria"]["AC5_runtime_reduction"]["passed"] is True
        assert report["criteria"]["AC6_fallback_rate"]["passed"] is True
        assert report["criteria"]["AC7_feature_importance"]["passed"] is True
        assert report["criteria"]["AC8_leakage_audit"]["passed"] is True



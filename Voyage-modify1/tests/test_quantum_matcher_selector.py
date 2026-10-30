"""
tests/test_quantum_matcher_selector.py
======================================
Unit and guardrail tests for Build Spec v2 §3A Intelligent Matcher Selector.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.matcher_selector import (
    RuleBasedMatcherSelector,
    LearnedMatcherEvaluator,
    log_matcher_run,
    get_available_matchers,
    DEFAULT_FALLBACK_MATCHER,
)
from src.selector.model import MatcherSelector, SelectorResult
from src.selector.features import MSMFeatureVector


@pytest.fixture
def base_features():
    return {
        "pair_id": "test_pair_001",
        "sensor_pair_enc": 0,          # OHRC-NAC (optical)
        "gsd_ratio": 1.0,
        "delta_solar_azimuth": 10.0,
        "crater_density": 0.5,
        "src_texture_contrast": 25.0,
        "ref_texture_contrast": 22.0,
        "terrain_class_enc": 0,
    }


def test_available_matchers_detection():
    available = get_available_matchers()
    assert isinstance(available, list)
    assert len(available) > 0
    # Must contain at least sift or real matchers
    assert any(m in available for m in ["sift", "rift2", "crater", "lnift"])


def test_rule_optical_standard_selects_sift(base_features):
    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "crater"])
    result = selector.select(base_features)

    assert result.selected_matcher == "sift"
    assert result.confidence >= 0.70
    assert "Selected sift" in result.explanation
    assert result.manual_override is False
    assert result.fallback_matcher in ["rift2", "crater"]


def test_rule_high_crater_density_selects_crater(base_features):
    features = dict(base_features)
    # crater_density is log(1 + craters/km2); log(1 + 6.0) ~ 1.945 > log(1 + 5.0) ~ 1.79
    features["crater_density"] = math.log1p(6.0)

    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "crater"])
    result = selector.select(features)

    assert result.selected_matcher == "crater"
    assert "high crater landmark density" in result.explanation
    assert any("high_crater_density" in step for step in result.decision_path)


def test_rule_multispectral_selects_rift2_or_lnift(base_features):
    features = dict(base_features)
    features["sensor_pair_enc"] = 2  # IIRS multispectral

    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "crater"])
    result = selector.select(features)

    assert result.selected_matcher == "rift2"
    assert "cross-sensor/multispectral" in result.explanation


def test_rule_large_illumination_diff_selects_rift2(base_features):
    features = dict(base_features)
    features["delta_solar_azimuth"] = 65.0  # > 45.0 degrees

    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "crater"])
    result = selector.select(features)

    assert result.selected_matcher == "rift2"
    assert "illumination difference" in result.explanation


def test_rule_large_scale_disparity_selects_rift2(base_features):
    features = dict(base_features)
    features["gsd_ratio"] = 0.25  # GSD ratio < 0.5

    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "crater"])
    result = selector.select(features)

    assert result.selected_matcher == "rift2"
    assert "scale disparity" in result.explanation


def test_rule_low_texture_selects_lightglue_or_rift2(base_features):
    features = dict(base_features)
    features["src_texture_contrast"] = 2.0
    features["ref_texture_contrast"] = 3.0

    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "lightglue"])
    result = selector.select(features)

    assert result.selected_matcher == "lightglue"
    assert "low texture contrast" in result.explanation


def test_manual_override_valid():
    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2", "crater"])
    result = selector.select({"pair_id": "p1"}, override_matcher="rift2")

    assert result.selected_matcher == "rift2"
    assert result.manual_override is True
    assert result.confidence == 1.0
    assert "Manual override" in result.explanation


def test_manual_override_unknown_matcher_rejected():
    selector = RuleBasedMatcherSelector(available_matchers=["sift", "rift2"])
    # "non_existent_matcher_xyz" is not in available_matchers
    result = selector.select({"pair_id": "p1"}, override_matcher="non_existent_matcher_xyz")

    # Guardrail rejects unknown matcher and applies normal rule/fallback
    assert result.selected_matcher in ["sift", "rift2"]
    assert result.manual_override is False


def test_guardrail_only_available_matchers_selected():
    # Only "rift2" is available
    selector = RuleBasedMatcherSelector(available_matchers=["rift2"])
    result = selector.select({"pair_id": "p1"})

    assert result.selected_matcher == "rift2"
    assert result.selected_matcher in result.available_matchers


def test_learned_matcher_evaluator_metrics():
    predictions = ["sift", "rift2", "crater", "sift"]
    oracle_best = ["sift", "rift2", "sift", "sift"]
    metrics = [
        {"sift": 0.9, "rift2": 0.5, "crater": 0.2},
        {"sift": 0.6, "rift2": 0.8, "crater": 0.1},
        {"sift": 0.7, "rift2": 0.4, "crater": 0.3},
        {"sift": 0.95, "rift2": 0.5, "crater": 0.2},
    ]

    res = LearnedMatcherEvaluator.evaluate(predictions, oracle_best, metrics)
    assert res["total_samples"] == 4
    # 3 out of 4 correct
    assert res["selection_accuracy"] == 0.75
    # Sample 3 had crater (0.3) vs oracle sift (0.7) -> regret = 0.4. Mean regret = 0.4/4 = 0.10
    assert abs(res["mean_regret"] - 0.10) < 1e-4


def test_benchmark_logging(tmp_path):
    log_file = tmp_path / "matcher_benchmark.jsonl"
    entry = log_matcher_run(
        pair_id="pair_abc",
        selected_matcher="sift",
        success=True,
        keypoint_count=1200,
        inlier_count=850,
        runtime_ms=145.2,
        metadata={"gsd_ratio": 0.8},
        log_path=log_file,
    )

    assert entry["pair_id"] == "pair_abc"
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as fh:
        loaded = json.loads(fh.readline())
    assert loaded["selected_matcher"] == "sift"
    assert loaded["inlier_count"] == 850


def test_integration_with_matcher_selector_class():
    ms = MatcherSelector(config={"msm": {"strategy": "rules"}})
    dummy_feat = MSMFeatureVector(
        pair_id="pair_int_01",
        sensor_pair_enc=0,
        gsd_ratio=1.0,
        latitude_abs=15.0,
        delta_solar_azimuth=5.0,
        terrain_class_enc=0,
        crater_density=0.2,
        masked_fraction=0.0,
        overlap_fraction=0.9,
        src_texture_contrast=20.0,
        ref_texture_contrast=18.0,
        src_mean_gradient=10.0,
        ref_mean_gradient=9.0,
        tile_count=4,
    )

    res = ms.predict(dummy_feat)
    assert isinstance(res, SelectorResult)
    assert res.selected_matcher in ["sift", "rift2", "lightglue", "crater"]
    assert len(res.explanation) > 0
    assert res.routing_reason in ["rule_based_selection", "high_confidence_single_matcher"]

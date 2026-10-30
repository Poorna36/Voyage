"""
tests/test_quantum_q0_filter.py
===============================
Unit tests for Q0b QUBO Consistency Filter (matcher_qubo.py).
"""
import numpy as np
import pytest

from quantum.matcher_qubo import QUBOMatcher
from src.matching.base import MatchResult


def test_q0b_filter_preserves_consistent_matches():
    # Synthetic consistent shift: ref = src + [10, -5]
    src_xy = np.array([
        [50.0, 50.0],
        [150.0, 60.0],
        [200.0, 180.0],
        [75.0, 220.0],
    ], dtype=np.float32)
    ref_xy = src_xy + np.array([10.0, -5.0], dtype=np.float32)
    conf = np.array([0.9, 0.85, 0.88, 0.92], dtype=np.float32)

    raw = MatchResult(src_xy=src_xy, ref_xy=ref_xy, confidence=conf)

    matcher = QUBOMatcher({"solver": "sqa", "lambda_penalty": 5.0})
    filtered, info = matcher.filter_matches(raw, gsd_ratio=1.0)

    assert filtered.count >= 2
    assert info["energy"] <= 0.0  # Favorable negative energy for consistent set
    assert "qubo_sqa_filter" in filtered.matcher_params["matcher_id"]


def test_q0b_filter_rejects_duplicate_keypoints():
    # Two matches sharing same source point (one-to-one violation)
    src_xy = np.array([
        [100.0, 100.0],
        [100.0, 100.0],  # Duplicate src
        [200.0, 200.0],
    ], dtype=np.float32)
    ref_xy = np.array([
        [110.0, 110.0],
        [180.0, 180.0],  # Different ref
        [210.0, 210.0],
    ], dtype=np.float32)
    conf = np.array([0.95, 0.50, 0.90], dtype=np.float32)

    raw = MatchResult(src_xy=src_xy, ref_xy=ref_xy, confidence=conf)

    matcher = QUBOMatcher({"solver": "sqa", "lambda_penalty": 10.0})
    filtered, info = matcher.filter_matches(raw, gsd_ratio=1.0)

    # Filter must select at most 1 of the duplicate points
    kept_src = filtered.src_xy
    if len(kept_src) > 1:
        dists = np.linalg.norm(kept_src[0] - kept_src[1:])
        assert np.all(dists > 1.0)


def test_q0_service_magsac_twin_side_by_side():
    from quantum.service import QuantumService

    svc = QuantumService()
    rec = svc.run_module(pair_id="test_q0_twin", module="Q0_filter", seed=42)

    assert rec.module == "Q0_filter"
    assert rec.problem["n_candidates"] == 24

    # Quantum metrics
    assert rec.metrics["n_kept"] > 0
    assert "qubo_energy" in rec.metrics

    # Classical twin metrics - MAGSAC++/RANSAC verified
    twin = rec.classical_twin
    assert twin["name"] in ("magsac_plus_plus", "ransac")
    assert twin["n_kept"] == 18  # Exactly 18 true inliers out of 24
    assert twin["kept_fraction"] == 0.75
    assert twin["median_cut_n_kept"] == 12  # Median cut reported side-by-side
    assert twin["runtime_s"] >= 0.0


def test_q0_service_custom_matches_magsac():
    from quantum.service import QuantumService

    svc = QuantumService()
    # 10 points: 7 true inliers shifted by (5, -3), 3 outliers
    rng = np.random.default_rng(123)
    s_xy = rng.uniform(50, 200, size=(10, 2)).astype(np.float32)
    r_xy = s_xy + np.array([5.0, -3.0], dtype=np.float32)
    r_xy[-3:] += np.array([[80.0, 90.0], [-70.0, 60.0], [50.0, -90.0]], dtype=np.float32)
    conf = np.ones(10, dtype=np.float32) * 0.9

    rec = svc.run_module(
        pair_id="custom_q0",
        module="Q0_filter",
        inputs={"src_xy": s_xy, "ref_xy": r_xy, "confidence": conf},
        seed=123,
    )

    twin = rec.classical_twin
    assert twin["name"] in ("magsac_plus_plus", "ransac")
    assert twin["n_kept"] == 7  # 7 true inliers detected
    assert twin["kept_fraction"] == 0.7
    assert twin["median_cut_n_kept"] == 10  # Median cut kept all 10 because conf all equal


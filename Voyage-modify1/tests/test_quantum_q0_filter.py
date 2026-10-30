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

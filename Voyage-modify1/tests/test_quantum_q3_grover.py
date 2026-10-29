"""
tests/test_quantum_q3_grover.py
===============================
Unit tests for Q3 Grover Consensus Search (grover_consensus.py).
"""
import numpy as np
import pytest

from quantum.grover_consensus import GroverConsensusSearch, compute_homography_4pts, count_inliers


def test_dlt_homography_reconstruction():
    # Affine transform
    src = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=float)
    ref = src + np.array([5.0, -3.0])

    H = compute_homography_4pts(src, ref)
    assert H is not None

    inliers, mask, rmse = count_inliers(H, src, ref, threshold_px=1.0)
    assert inliers == 4
    assert rmse < 1e-4


def test_grover_search_synthetic_matches():
    # 8 points with affine transform
    src = np.array([
        [20, 20], [60, 25], [100, 30], [30, 80],
        [70, 90], [120, 85], [40, 140], [90, 150]
    ], dtype=float)
    ref = src + np.array([12.0, -8.0])

    searcher = GroverConsensusSearch(tau_inlier_ratio=0.50, inlier_threshold_px=2.0, seed=42)
    res = searcher.search(src, ref, max_candidates=8)

    assert res["success"] is True
    assert res["oracle_queries_grover"] >= 1
    assert res["oracle_queries_grover"] <= res["expected_queries_classical"] + 1
    assert res["measured_inliers"] >= 4

"""
tests/test_quantum_qaoa_correspondence.py
=========================================
Tests for Build Spec v2 §3E: QAOA-Based Correspondence Optimisation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.classical_baseline import (
    ExactQUBOSolver,
    GreedyCompatibilitySolver,
    compute_constraint_diagnostics,
)
from quantum.qaoa_correspondence import (
    QAOACorrespondenceOptimizer,
    QAOACorrespondenceResult,
)


@pytest.fixture
def sample_correspondence_problem():
    # 6 points arranged in a rigid 2D grid
    src_pts = np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [10.0, 50.0],
        [50.0, 50.0],
        [90.0, 10.0],
        [90.0, 50.0],
    ], dtype=np.float32)

    # Reference points: shifted by (20, 15) plus one spurious point
    shift = np.array([20.0, 15.0], dtype=np.float32)
    ref_pts = np.vstack([
        src_pts + shift,
        [[200.0, 200.0]],  # outlier
    ]).astype(np.float32)

    # Candidate matches: 6 true matches, 2 conflicting/outlier matches
    # Pairs: (src_idx, ref_idx)
    candidate_pairs = np.array([
        [0, 0],  # true
        [1, 1],  # true
        [2, 2],  # true
        [3, 3],  # true
        [4, 4],  # true
        [5, 5],  # true
        [0, 6],  # conflict on src 0 with outlier
        [1, 0],  # conflict on ref 0 with src 1
    ], dtype=np.int32)

    affinities = np.array([0.9, 0.85, 0.88, 0.92, 0.80, 0.86, 0.4, 0.35], dtype=np.float32)

    return src_pts, ref_pts, candidate_pairs, affinities


def test_exact_qubo_solver():
    solver = ExactQUBOSolver(max_vars=8)
    # Simple 3-variable QUBO: min at x=[1, 1, 0]
    Q = np.array([
        [-2.0, 0.5, 3.0],
        [0.5, -3.0, 2.0],
        [3.0, 2.0, 1.0],
    ])
    res = solver.solve(Q)
    assert res["solver"] == "exact_bruteforce"
    assert res["n_states_evaluated"] == 8
    # Global minimum is negative
    assert res["energy"] < 0
    assert len(res["solution"]) == 3


def test_greedy_compatibility_solver(sample_correspondence_problem):
    src_pts, ref_pts, pairs, affinities = sample_correspondence_problem
    opt = QAOACorrespondenceOptimizer()
    Q, _ = opt.build_correspondence_qubo(src_pts, ref_pts, pairs, affinities)

    solver = GreedyCompatibilitySolver()
    res = solver.solve(Q, pairs=pairs, unary_affinities=affinities)

    assert res["solver"] == "greedy_compatibility"
    assert len(res["solution"]) == len(pairs)
    assert np.sum(res["solution"]) >= 4

    # Diagnostics: ensure greedy avoided conflicts
    diag = compute_constraint_diagnostics(res["solution"], pairs=pairs, Q=Q)
    assert diag["total_violations"] == 0
    assert diag["is_valid"] is True


def test_qaoa_correspondence_optimizer_with_exact_gap(sample_correspondence_problem):
    src_pts, ref_pts, pairs, affinities = sample_correspondence_problem
    opt = QAOACorrespondenceOptimizer(p_layers=1, max_vars=8, use_exact_benchmark=True)

    result = opt.optimize(src_pts, ref_pts, pairs, affinities, use_qaoa=True)

    assert isinstance(result, QAOACorrespondenceResult)
    assert result.n_candidates == 8
    assert result.n_selected >= 4
    assert "exact" in result.baselines
    assert "greedy_compatibility" in result.baselines

    # Check constraint diagnostics
    diag = result.constraint_diagnostics
    assert "total_violations" in diag
    assert "energy" in diag
    assert diag["gap_to_exact"] is not None

    # Check homography estimation output
    assert result.homography_result is not None
    assert result.homography_result["success"] is True
    assert result.homography_result["n_inliers"] >= 4


def test_switch_qaoa_vs_classical(sample_correspondence_problem):
    src_pts, ref_pts, pairs, affinities = sample_correspondence_problem
    opt = QAOACorrespondenceOptimizer(p_layers=1, max_vars=8)

    # QAOA enabled
    res_q = opt.optimize(src_pts, ref_pts, pairs, affinities, use_qaoa=True)
    assert res_q.use_qaoa is True
    assert len(res_q.src_xy_selected) > 0

    # Classical path
    res_c = opt.optimize(src_pts, ref_pts, pairs, affinities, use_qaoa=False)
    assert res_c.use_qaoa is False
    assert len(res_c.src_xy_selected) > 0

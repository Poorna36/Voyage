"""
tests/test_quantum_q1_selection.py
==================================
Unit tests for Q1 QAOA Spatial Subset Selection (selection_qaoa.py).
"""
import numpy as np
import pytest

from quantum.selection_qaoa import build_cell_qubo, select_cell_qaoa, select_qaoa, solve_cell_exact


def test_cell_qubo_energy_consistency():
    # Random small cell candidate set
    coords = np.array([[10.0, 10.0], [12.0, 11.0], [80.0, 85.0]], dtype=float)
    conf = np.array([0.9, 0.7, 0.85], dtype=float)

    Q, lam = build_cell_qubo(coords, conf, k=2, cell_size=100.0)

    # Diagonal check: Q_ii = -c_i + lambda * (1 - 2k)
    for i in range(len(coords)):
        expected_diag = -conf[i] + lam * (1.0 - 2.0 * 2)
        assert abs(Q[i, i] - expected_diag) < 1e-9

    # Pairwise check: points close together (0 and 1) have high redundancy penalty s_ij
    assert Q[0, 1] > Q[0, 2]


def test_qaoa_energy_vs_exact():
    # QAOA energy should be >= exact min energy
    coords = np.array([[10.0, 10.0], [50.0, 50.0], [20.0, 90.0], [80.0, 10.0]], dtype=float)
    conf = np.array([0.8, 0.9, 0.7, 0.85], dtype=float)

    res = select_cell_qaoa(coords, conf, k=2, cell_size=100.0, p_layers=2, restarts=3, seed=42)

    assert res["energy"] >= res["exact_energy"] - 1e-6
    assert 0.0 <= res["approx_ratio"] <= 1.0


def test_full_grid_qaoa_selection():
    rng = np.random.default_rng(42)
    N = 30
    src_xy = rng.uniform(0, 500, size=(N, 2)).astype(np.float32)
    ref_xy = src_xy + np.array([4.0, -2.0], dtype=np.float32)
    conf = rng.uniform(0.5, 0.99, size=N).astype(np.float32)

    s_out, r_out, c_out, metrics = select_qaoa(src_xy, ref_xy, conf, n=4, cap=3, seed=42)

    assert len(s_out) <= N
    assert len(s_out) == len(r_out) == len(c_out)
    assert metrics["approx_ratio_mean"] >= 0.80

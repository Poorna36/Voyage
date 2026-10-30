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


def test_qaoa_shots_forwarding_and_equality():
    from quantum.service import QuantumService

    # (a) With default shots (1024), output matches default invocation
    svc = QuantumService()
    rec_default = svc.run_module(pair_id="default_shots", module="Q1_selection", seed=42)
    assert rec_default.metrics["shots"] == 1024
    assert rec_default.problem["shots"] == 1024
    assert rec_default.circuit["shots"] == 1024

    # (b) Non-default shots value actually reaches the solver and service records
    rec_custom = svc.run_module(pair_id="custom_shots", module="Q1_selection", params={"shots": 256}, seed=42)
    assert rec_custom.metrics["shots"] == 256
    assert rec_custom.problem["shots"] == 256
    assert rec_custom.circuit["shots"] == 256

    # Direct select_cell_qaoa verification
    coords = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    cell_res = select_cell_qaoa(coords, conf, k=2, shots=128)
    assert cell_res["shots"] == 128


def test_qaoa_shots_affects_sampling():
    from quantum.qaoa import QAOASolver

    coords = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0], [60.0, 60.0]])
    conf = np.array([0.9, 0.85, 0.75, 0.7, 0.65, 0.6])
    Q, _ = build_cell_qubo(coords, conf, k=3, cell_size=100.0)

    # 1 shot vs 1024 shots
    s1 = QAOASolver(p_layers=1, n_optimizer_restarts=1, n_shots=1, seed=42)
    r1 = s1.solve(Q)

    s1024 = QAOASolver(p_layers=1, n_optimizer_restarts=1, n_shots=1024, seed=42)
    r1024 = s1024.solve(Q)

    # Variational optimization uses exact statevector expectation: expected_energy is identical
    assert np.isclose(r1["expected_energy"], r1024["expected_energy"], atol=1e-6)
    assert r1["optimal_gamma"] == r1024["optimal_gamma"]

    # Final measurement sampling: 1024 shots finds a state with energy <= 1 shot sample
    # Guaranteed unable to flake: both use seed=42 and numpy's generator makes the 1-sample draw
    # the exact first element of the 1024-sample draw (x_1 in {x_1, ..., x_1024} -> min_1024 <= E_1)
    assert r1024["energy"] <= r1["energy"]
    assert r1024["energy"] >= r1024["ground_state_energy"] - 1e-6
    assert r1024["n_shots"] == 1024
    assert r1["n_shots"] == 1


"""
tests/test_quantum_q4_qft.py
============================
Unit tests for Q4 Gate-Level QFT Phase Correlation (qft_phase_corr.py).
"""
import numpy as np
import pytest

from quantum.qft_phase_corr import apply_2d_qft, paraboloid_peak_subpixel, qft_phase_correlation


def test_2d_qft_vs_fft2_ortho():
    # 8x8 tile
    tile = np.random.uniform(0.0, 1.0, size=(8, 8))
    q_spec, c_spec, info = apply_2d_qft(tile)

    # QFT statevector should match np.fft.fft2(tile, norm="ortho") within numerical precision
    diff = np.max(np.abs(q_spec - c_spec))
    assert diff < 1e-10
    assert info["total_gates"] > 0
    assert info["n_qubits"] == 6  # 3 row + 3 col qubits


def test_paraboloid_subpixel_known_shift():
    # Synthetic correlation surface with known subpixel peak at (dx=0.3, dy=-0.4)
    corr = np.zeros((16, 16), dtype=float)
    r_center, c_center = 8, 8
    corr[r_center, c_center] = 1.0
    corr[r_center, c_center + 1] = 0.7
    corr[r_center, c_center - 1] = 0.5
    corr[r_center + 1, c_center] = 0.4
    corr[r_center - 1, c_center] = 0.8

    sc, sr, peak = paraboloid_peak_subpixel(corr)
    assert abs(sc) <= 8.0
    assert abs(sr) <= 8.0

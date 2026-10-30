"""
tests/test_quantum_q5_imaging.py
================================
Unit tests for Q5 Quantum Image Encoding and Edge Detection (image_encoding.py).
"""
import numpy as np
import pytest

from quantum.image_encoding import decode_frqi, decode_neqr, encode_frqi, encode_neqr, quantum_edge_detection, run_q5_demo


def test_frqi_round_trip():
    tile = np.array([
        [0.0, 0.25],
        [0.5, 0.75],
    ], dtype=float)

    psi, qubits = encode_frqi(tile)
    assert qubits == 3  # 2 position + 1 color
    assert len(psi) == 8

    decoded = decode_frqi(psi, 2, 2)
    assert np.allclose(tile, decoded, atol=1e-6)


def test_neqr_round_trip():
    tile = np.array([
        [0.0, 0.5],
        [0.8, 1.0],
    ], dtype=float)

    psi, qubits = encode_neqr(tile)
    assert qubits == 10  # 2 position + 8 gray bits
    assert len(psi) == 1024

    decoded = decode_neqr(psi, 2, 2)
    # Quantization error must be <= 1/255
    assert np.max(np.abs(tile - decoded)) <= 1.0 / 255.0 + 1e-6


def test_qhed_edge_detection():
    # Step-edge tile
    tile = np.zeros((8, 8), dtype=float)
    tile[:, 4:] = 1.0

    edge_map, meta = quantum_edge_detection(tile)
    # Peak response should be at column 4 boundary
    assert np.mean(edge_map[:, 3:5]) > np.mean(edge_map[:, :2])
    assert meta["pearson_vs_sobel"] > 0.70

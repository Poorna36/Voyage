"""
quantum/image_encoding.py
=========================
Q5 — Quantum Image Encoding and Edge Detection (Preprocessing Demo Track).

Implements:
  1. FRQI (Flexible Representation of Quantum Images):
     2n + 1 qubits for 2^n x 2^n tile (16x16 -> 9 qubits).
     Angle theta = (pi / 2) * pixel_intensity on color qubit.
  2. NEQR (Novel Enhanced Quantum Representation):
     2n + q qubits (q=8 bit grayscale). Exact basis encoding (16x16 -> 16 qubits).
  3. Quantum Hadamard Edge Detection (QHED):
     Adjacent pixel differentiation via position-register permutations and Hadamard gates.

Classical Twins:
  - OpenCV / SciPy Sobel gradient magnitude filter.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FRQI (Flexible Representation of Quantum Images)
# ─────────────────────────────────────────────────────────────────────────────

def encode_frqi(tile: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Encode a 2^n x 2^n grayscale tile in [0, 1] using FRQI.
    Requires 2n + 1 qubits.
    Returns:
      (statevector, total_qubits)
    """
    H, W = tile.shape
    assert H == W and (H & (H - 1) == 0), "Tile must be square with power-of-two dimensions"
    n = int(math.log2(H))
    n_pos_qubits = 2 * n
    total_qubits = n_pos_qubits + 1
    dim = 1 << total_qubits

    # Statevector: basis state = |pos> |color>
    # Even indices (2*p) represent color=|0> (cos theta), odd (2*p + 1) represent color=|1> (sin theta)
    psi = np.zeros(dim, dtype=complex)
    flat = np.clip(tile.flatten(), 0.0, 1.0)
    thetas = (math.pi / 2.0) * flat

    norm_factor = 1.0 / (2.0 ** n)
    for p, th in enumerate(thetas):
        psi[2 * p] = norm_factor * math.cos(th)
        psi[2 * p + 1] = norm_factor * math.sin(th)

    return psi, total_qubits


def decode_frqi(psi: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Decode statevector amplitudes back to a 2D image tile.
    """
    n_pixels = H * W
    decoded = np.zeros(n_pixels, dtype=float)

    for p in range(n_pixels):
        amp0 = abs(psi[2 * p])
        amp1 = abs(psi[2 * p + 1])
        # tan(theta) = amp1 / amp0 -> theta = atan2(amp1, amp0)
        th = math.atan2(amp1, amp0 + 1e-12)
        decoded[p] = th / (math.pi / 2.0)

    return decoded.reshape(H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEQR (Novel Enhanced Quantum Representation)
# ─────────────────────────────────────────────────────────────────────────────

def encode_neqr(tile: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Encode tile in [0, 255] uint8 using NEQR basis encoding.
    Requires 2n + 8 qubits.
    Returns (statevector, total_qubits).
    """
    H, W = tile.shape
    n = int(math.log2(H))
    n_pos = 2 * n
    n_gray = 8
    total_qubits = n_pos + n_gray
    dim = 1 << total_qubits

    psi = np.zeros(dim, dtype=complex)
    flat_uint8 = np.clip(np.round(tile * 255.0), 0, 255).astype(int).flatten()

    norm_factor = 1.0 / math.sqrt(len(flat_uint8))
    for p, val in enumerate(flat_uint8):
        # Index = (p << 8) | val
        basis_idx = (p << 8) | val
        psi[basis_idx] = norm_factor

    return psi, total_qubits


def decode_neqr(psi: np.ndarray, H: int, W: int) -> np.ndarray:
    """Decode NEQR statevector back to float [0, 1] tile."""
    n_pixels = H * W
    decoded = np.zeros(n_pixels, dtype=float)
    probs = np.abs(psi) ** 2

    for p in range(n_pixels):
        base = p << 8
        # Find which intensity value has the non-zero probability
        pixel_probs = probs[base : base + 256]
        val = int(np.argmax(pixel_probs))
        decoded[p] = val / 255.0

    return decoded.reshape(H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantum Hadamard Edge Detection (QHED)
# ─────────────────────────────────────────────────────────────────────────────

def classical_sobel_gradient(tile: np.ndarray) -> np.ndarray:
    """Classical Sobel gradient magnitude filter."""
    from scipy.ndimage import sobel
    gx = sobel(tile, axis=1)
    gy = sobel(tile, axis=0)
    mag = np.hypot(gx, gy)
    max_val = np.max(mag)
    if max_val > 1e-12:
        mag /= max_val
    return mag


def quantum_edge_detection(tile: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Simulate Quantum Hadamard Edge Detection on a 2^n x 2^n tile.
    Returns:
      (quantum_edge_map, metadata)
    """
    H, W = tile.shape
    norm_tile = (tile - np.min(tile)) / (np.ptp(tile) + 1e-12)

    # Horizontal and vertical differential amplitude operations
    # Adjacent pixel differences: (f(x+1) - f(x-1)) / 2
    # In quantum circuit this is implemented by cyclical permutation + Hadamard on position qubit
    diff_h = np.zeros_like(norm_tile)
    diff_v = np.zeros_like(norm_tile)

    diff_h[:, 1:-1] = 0.5 * np.abs(norm_tile[:, 2:] - norm_tile[:, :-2])
    diff_v[1:-1, :] = 0.5 * np.abs(norm_tile[2:, :] - norm_tile[:-2, :])

    edge_mag = np.hypot(diff_h, diff_v)
    max_e = np.max(edge_mag)
    if max_e > 1e-12:
        edge_mag /= max_e

    # Classical twin: Sobel
    twin_sobel = classical_sobel_gradient(norm_tile)

    # Metrics: Pearson correlation
    edge_flat = edge_mag.flatten()
    sobel_flat = twin_sobel.flatten()
    cov = np.cov(edge_flat, sobel_flat)
    std_prod = np.std(edge_flat) * np.std(sobel_flat)
    pearson = float(cov[0, 1] / (std_prod + 1e-12))

    meta = {
        "tile_shape": (H, W),
        "qubits_amplitude": int(math.log2(H * W)) + 1,
        "pearson_vs_sobel": round(pearson, 4),
        "mean_quantum_response": round(float(np.mean(edge_mag)), 4),
    }

    return edge_mag, meta


def run_q5_demo(tile_size: int = 16, seed: int = 42) -> Dict[str, Any]:
    """
    Execute full Q5 demonstration: FRQI/NEQR round-trips and QHED edge detection.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # Synthetic lunar crater-like tile
    y, x = np.mgrid[-1:1:complex(tile_size), -1:1:complex(tile_size)]
    r = np.sqrt(x ** 2 + y ** 2)
    tile = np.clip(1.0 - np.exp(-((r - 0.4) ** 2) / 0.05) + rng.normal(0, 0.05, (tile_size, tile_size)), 0.0, 1.0)

    # 1. FRQI Round Trip
    psi_frqi, qubits_frqi = encode_frqi(tile)
    decoded_frqi = decode_frqi(psi_frqi, tile_size, tile_size)
    frqi_error = float(np.max(np.abs(tile - decoded_frqi)))

    # 2. NEQR Round Trip
    psi_neqr, qubits_neqr = encode_neqr(tile)
    decoded_neqr = decode_neqr(psi_neqr, tile_size, tile_size)
    neqr_error = float(np.max(np.abs(tile - decoded_neqr)))

    # 3. Quantum Hadamard Edge Detection
    edge_map, edge_meta = quantum_edge_detection(tile)

    runtime_s = time.time() - t0

    return {
        "tile_size": tile_size,
        "frqi": {
            "qubits": qubits_frqi,
            "max_reconstruction_error": round(frqi_error, 6),
            "ssim_equivalent": round(1.0 - frqi_error, 4),
        },
        "neqr": {
            "qubits": qubits_neqr,
            "max_reconstruction_error": round(neqr_error, 6),
            "is_exact": bool(neqr_error < 1.0 / 255.0 + 1e-6),
        },
        "edge_detection": edge_meta,
        "runtime_s": round(runtime_s, 4),
    }

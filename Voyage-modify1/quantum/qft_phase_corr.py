"""
quantum/qft_phase_corr.py
=========================
Q4 — Gate-Level QFT Phase Correlation (Layer L5 Sub-Pixel Refinement).

Implements explicit gate-level 2D Quantum Fourier Transform on statevectors
for sub-pixel translation estimation between image tiles.

Tile Encoding:
  For a 2^k x 2^k tile (e.g. 16x16 -> 8 qubits, 32x32 -> 10 qubits):
    - 2k qubits (k row qubits, k col qubits)
    - Amplitude encoding: |tile> = sum_{r, c} tile[r, c] |r>|c>

Circuit Construction:
  2D QFT = QFT_row (on first k qubits) \\otimes QFT_col (on second k qubits)
  Each 1D n-qubit QFT:
    For j in 0..n-1:
      Hadamard(j)
      For m in 2..n-j:
        Controlled-Phase(control=j+m-1, target=j, phi = 2*pi / 2^m)
    Swaps: swap(j, n - 1 - j) for j < n//2

Classical Twin:
  numpy.fft.fft2(..., norm="ortho") phase correlation with 2D paraboloid fit.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def apply_1d_qft_statevector(
    psi: np.ndarray,
    n_qubits: int,
    qubit_indices: List[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Apply 1D Quantum Fourier Transform to specified qubit register using explicit gates.
    qubit_indices: list of qubit indices forming the register (most to least significant).
    Returns (transformed_psi, gate_list).
    """
    gates: List[Dict[str, Any]] = []
    m = len(qubit_indices)

    # Tensor representation
    shape = [2] * n_qubits
    tensor = psi.reshape(shape)

    H_mat = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)

    for i in range(m):
        target_q = qubit_indices[i]

        # 1. Hadamard on target qubit
        tensor = np.tensordot(H_mat, tensor, axes=([1], [target_q]))
        tensor = np.moveaxis(tensor, 0, target_q)
        gates.append({"gate": "H", "target": target_q})

        # 2. Controlled Phase rotations
        for j in range(i + 1, m):
            control_q = qubit_indices[j]
            k_power = j - i + 1
            phi = -2.0 * math.pi / (2.0 ** k_power)

            # Apply phase when both control and target are 1
            idx = [slice(None)] * n_qubits
            idx[control_q] = 1
            idx[target_q] = 1
            tensor[tuple(idx)] *= np.exp(1j * phi)

            gates.append({
                "gate": "CPHASE",
                "control": control_q,
                "target": target_q,
                "phi": round(phi, 4),
            })

    # 3. Swap qubits (reverse register order)
    for i in range(m // 2):
        q1 = qubit_indices[i]
        q2 = qubit_indices[m - 1 - i]
        # Swap axes q1 and q2
        tensor = np.swapaxes(tensor, q1, q2)
        gates.append({"gate": "SWAP", "targets": [q1, q2]})

    return tensor.reshape(-1), gates


def apply_2d_qft(
    tile: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Execute full gate-level 2D QFT on an amplitude-encoded tile.
    Tile must have shape (2^k, 2^k).
    Returns:
      (quantum_spectrum, classical_fft2_spectrum, circuit_info)
    """
    H, W = tile.shape
    assert H == W and (H & (H - 1) == 0), "Tile dimensions must be equal and a power of two"

    k = int(math.log2(H))
    n_qubits = 2 * k  # k row qubits + k col qubits
    dim = 1 << n_qubits

    # 1. Amplitude encoding: |tile> = sum_{r, c} tile[r, c] |r>|c>
    # Flatten tile row-major: index = r * W + c
    norm = np.linalg.norm(tile)
    if norm < 1e-12:
        norm = 1.0
    psi_init = (tile.flatten() / norm).astype(complex)

    # 2. Row register QFT on first k qubits
    row_qubits = list(range(0, k))
    psi_row, gates_row = apply_1d_qft_statevector(psi_init, n_qubits, row_qubits)

    # 3. Column register QFT on remaining k qubits
    col_qubits = list(range(k, 2 * k))
    psi_2d, gates_col = apply_1d_qft_statevector(psi_row, n_qubits, col_qubits)

    all_gates = gates_row + gates_col
    quantum_spectrum = psi_2d.reshape(H, W)

    # Classical reference twin: numpy.fft.fft2 with ortho normalization
    classical_spectrum = np.fft.fft2(tile / norm, norm="ortho")

    # Difference metric
    max_diff = float(np.max(np.abs(quantum_spectrum - classical_spectrum)))

    circuit_info = {
        "n_qubits": n_qubits,
        "tile_shape": (H, W),
        "total_gates": len(all_gates),
        "hadamards": sum(1 for g in all_gates if g["gate"] == "H"),
        "controlled_phase": sum(1 for g in all_gates if g["gate"] == "CPHASE"),
        "swaps": sum(1 for g in all_gates if g["gate"] == "SWAP"),
        "max_diff_vs_classical_fft2": max_diff,
        "gates_sequence": all_gates[:30],  # sample for inspection
    }

    return quantum_spectrum, classical_spectrum, circuit_info


def paraboloid_peak_subpixel(corr: np.ndarray) -> Tuple[float, float, float]:
    """
    Sub-pixel translation peak interpolation via 2D paraboloid fit.
    Returns (sub_col, sub_row, peak_value).
    """
    H, W = corr.shape
    r_peak, c_peak = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[r_peak, c_peak])

    # If peak is at boundary, return integer shift
    if r_peak == 0 or r_peak == H - 1 or c_peak == 0 or c_peak == W - 1:
        # Wrap shifts to [-H//2, H//2)
        shift_r = r_peak if r_peak < H // 2 else r_peak - H
        shift_c = c_peak if c_peak < W // 2 else c_peak - W
        return float(shift_c), float(shift_r), peak_val

    # 3x3 patch centered at peak
    patch = corr[r_peak - 1 : r_peak + 2, c_peak - 1 : c_peak + 2]

    # Quadratic fit in col (x) and row (y)
    dx = (patch[1, 2] - patch[1, 0]) / (2.0 * (2.0 * patch[1, 1] - patch[1, 0] - patch[1, 2]) + 1e-12)
    dy = (patch[2, 1] - patch[0, 1]) / (2.0 * (2.0 * patch[1, 1] - patch[0, 1] - patch[2, 1]) + 1e-12)

    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))

    # Center-relative shift
    shift_r = float(np.clip((r_peak if r_peak < H // 2 else r_peak - H) + dy, -H / 2.0, H / 2.0))
    shift_c = float(np.clip((c_peak if c_peak < W // 2 else c_peak - W) + dx, -W / 2.0, W / 2.0))

    return shift_c, shift_r, peak_val


def qft_phase_correlation(
    tile_src: np.ndarray,
    tile_ref: np.ndarray,
    hann_window: bool = True,
) -> Dict[str, Any]:
    """
    Q4: Full phase correlation pipeline between two tiles using gate-level QFT.
    """
    t0 = time.time()
    assert tile_src.shape == tile_ref.shape, "Tile shapes must match"
    H, W = tile_src.shape

    # Preprocessing: Hann window
    ts = tile_src.astype(float) - np.mean(tile_src)
    tr = tile_ref.astype(float) - np.mean(tile_ref)

    if hann_window:
        w_row = np.hanning(H)
        w_col = np.hanning(W)
        window = np.outer(w_row, w_col)
        ts *= window
        tr *= window

    # 1. Gate-level QFT on both tiles
    Q_src, _, c_info_src = apply_2d_qft(ts)
    Q_ref, _, c_info_ref = apply_2d_qft(tr)

    # 2. Normalized cross-power spectrum
    cross_power = Q_src * np.conj(Q_ref)
    eps = 1e-12
    cross_power /= (np.abs(cross_power) + eps)

    # 3. Inverse transform (via inverse FFT or QFT^dagger)
    corr = np.real(np.fft.ifft2(cross_power))

    # 4. Sub-pixel paraboloid estimation
    shift_col, shift_row, peak = paraboloid_peak_subpixel(corr)

    # Classical twin: standard np.fft.fft2
    t_twin_0 = time.time()
    F_s = np.fft.fft2(ts, norm="ortho")
    F_r = np.fft.fft2(tr, norm="ortho")
    R_twin = F_s * np.conj(F_r)
    R_twin /= (np.abs(R_twin) + eps)
    corr_twin = np.real(np.fft.ifft2(R_twin))
    twin_c, twin_r, twin_peak = paraboloid_peak_subpixel(corr_twin)
    twin_runtime = time.time() - t_twin_0

    runtime_s = time.time() - t0

    return {
        "shift_col_px": round(shift_col, 4),
        "shift_row_px": round(shift_row, 4),
        "peak_correlation": round(peak, 4),
        "classical_shift_col_px": round(twin_c, 4),
        "classical_shift_row_px": round(twin_r, 4),
        "qubits": c_info_src["n_qubits"],
        "gate_counts": {
            "H": c_info_src["hadamards"] * 2,
            "CPHASE": c_info_src["controlled_phase"] * 2,
            "SWAP": c_info_src["swaps"] * 2,
            "total": (c_info_src["total_gates"] + c_info_ref["total_gates"]),
        },
        "max_circuit_vs_fft2_error": c_info_src["max_diff_vs_classical_fft2"],
        "runtime_s": round(runtime_s, 4),
        "classical_twin_runtime_s": round(twin_runtime, 6),
    }

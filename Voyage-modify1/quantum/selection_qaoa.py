"""
quantum/selection_qaoa.py
=========================
Q1 — QAOA Keypoint-Subset Selection (Layer L3 Spatial Selection).

Inside each cell of an NxN grid (default 8x8), selects at most k matches (default 5)
balancing match confidence and spatial dispersion via QAOA statevector optimization.

QUBO Formulation per Cell (K candidates, K <= 12):
    H(x) = - sum_i c_i x_i
           + mu * sum_{i < j} s_ij x_i x_j
           + lambda * (sum_i x_i - k)^2

QUBO Matrix entries (upper triangular):
    Q_ii = -c_i + lambda * (1 - 2k)
    Q_ij = mu * s_ij + 2 * lambda   (for i < j)
where:
    s_ij = exp(- d_ij^2 / (2 * sigma^2))
    sigma = cell_size / 3.0
    lambda = max(c_i) + mu + 1.0 (cardinality dominating factor)

Classical Twins:
  - Twin A: grid_cap (top-k by confidence)
  - Twin B: exact brute-force search over 2^K bitstrings to measure approximation ratio.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .qaoa import QAOASolver

log = logging.getLogger(__name__)


def build_cell_qubo(
    coords: np.ndarray,
    confidence: np.ndarray,
    k: int = 5,
    cell_size: float = 128.0,
    mu: float = 1.5,
    lambda_factor: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Construct the QUBO matrix Q for a single cell candidate set.
    coords: (K, 2) array of (col, row)
    confidence: (K,) array of confidence scores in [0, 1]
    """
    K = len(coords)
    assert len(confidence) == K, "Coords and confidence must have identical length"

    Q = np.zeros((K, K), dtype=np.float64)
    sigma = max(cell_size / 3.0, 1e-3)
    c_max = float(np.max(confidence)) if K > 0 else 1.0
    lam = lambda_factor if lambda_factor is not None else (c_max + mu + 1.0)

    # Diagonal terms: -c_i + lambda * (1 - 2k)
    for i in range(K):
        Q[i, i] = -float(confidence[i]) + lam * (1.0 - 2.0 * k)

    # Off-diagonal pairwise terms: mu * s_ij + 2 * lambda
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = float(np.linalg.norm(coords[i] - coords[j]))
            s_ij = float(np.exp(- (d_ij ** 2) / (2.0 * (sigma ** 2))))
            Q[i, j] = mu * s_ij + 2.0 * lam

    return Q, lam


def solve_cell_exact(Q: np.ndarray, k: int) -> Tuple[np.ndarray, float, float, float]:
    """
    Brute-force evaluate all 2^K bitstrings for K <= 16.
    Returns (best_bitstring, min_energy, max_energy, mean_energy).
    """
    K = Q.shape[0]
    best_x = np.zeros(K, dtype=int)
    min_energy = float("inf")
    max_energy = float("-inf")
    total_energy = 0.0

    n_states = 1 << K
    for idx in range(n_states):
        # Convert integer to bit array
        x = np.array([(idx >> bit) & 1 for bit in range(K)], dtype=float)
        # Energy = x^T Q x
        E = float(x @ Q @ x)
        if E < min_energy:
            min_energy = E
            best_x = x.astype(int)
        if E > max_energy:
            max_energy = E
        total_energy += E

    return best_x, min_energy, max_energy, total_energy / n_states


def select_cell_qaoa(
    coords: np.ndarray,
    confidence: np.ndarray,
    k: int = 5,
    cell_size: float = 128.0,
    mu: float = 1.5,
    p_layers: int = 2,
    restarts: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Solve subset selection for one cell using QAOA and compare with exact solver.
    """
    K = len(coords)
    if K <= k:
        # All candidates kept trivially
        ones = np.ones(K, dtype=int)
        Q, _ = build_cell_qubo(coords, confidence, k, cell_size, mu)
        E = float(ones @ Q @ ones) if K > 0 else 0.0
        return {
            "solution": ones,
            "energy": E,
            "exact_energy": E,
            "approx_ratio": 1.0,
            "n_candidates": K,
            "n_selected": K,
            "trivially_satisfied": True,
        }

    # Cap problem to 12 qubits if K > 12 to guarantee statevector performance
    if K > 12:
        top_k_idx = np.argsort(-confidence)[:12]
        coords = coords[top_k_idx]
        confidence = confidence[top_k_idx]
        K = 12

    Q, lam = build_cell_qubo(coords, confidence, k, cell_size, mu)

    # Solve exact baseline (Twin B)
    exact_x, min_E, max_E, mean_E = solve_cell_exact(Q, k)

    # Solve via QAOA
    solver = QAOASolver(p_layers=p_layers, n_optimizer_restarts=restarts, seed=seed)
    qaoa_res = solver.solve(Q)

    qaoa_sol = np.asarray(qaoa_res.get("solution", exact_x), dtype=int)
    qaoa_E = float(qaoa_sol @ Q @ qaoa_sol)

    # Standard approximation ratio for minimization problem
    denom = (max_E - min_E) if abs(max_E - min_E) > 1e-9 else 1.0
    approx_ratio = float(np.clip((max_E - qaoa_E) / denom, 0.0, 1.0))

    return {
        "solution": qaoa_sol,
        "energy": qaoa_E,
        "exact_energy": min_E,
        "max_energy": max_E,
        "approx_ratio": approx_ratio,
        "n_candidates": K,
        "n_selected": int(np.sum(qaoa_sol)),
        "trivially_satisfied": False,
        "p_layers": p_layers,
    }


def select_qaoa(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    n: int = 8,
    cap: int = 5,
    p_layers: int = 2,
    restarts: int = 3,
    mu: float = 1.5,
    backend: Optional[Any] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Q1: Full NxN grid QAOA selection across all cells.

    Returns:
      (selected_src_xy, selected_ref_xy, selected_confidence, metrics_summary)
    """
    t0 = time.time()
    assert src_xy.shape[-1] == 2, "src_xy must have shape (N, 2)"
    assert ref_xy.shape[-1] == 2, "ref_xy must have shape (N, 2)"
    assert len(src_xy) == len(ref_xy) == len(confidence), "Length mismatch in match inputs"

    N = len(src_xy)
    if N == 0:
        return src_xy, ref_xy, confidence, {
            "n_cells_processed": 0,
            "approx_ratio_mean": 1.0,
            "approx_ratio_min": 1.0,
            "runtime_s": 0.0,
        }

    if image_shape is not None:
        H, W = image_shape
    else:
        H = float(np.max(src_xy[:, 1])) + 1.0
        W = float(np.max(src_xy[:, 0])) + 1.0

    cell_w = W / n
    cell_h = H / n
    cell_size = (cell_w + cell_h) / 2.0

    # Assign cell IDs
    cell_x = np.clip((src_xy[:, 0] / W * n).astype(int), 0, n - 1)
    cell_y = np.clip((src_xy[:, 1] / H * n).astype(int), 0, n - 1)
    cell_ids = cell_y * n + cell_x

    keep_mask = np.zeros(N, dtype=bool)
    cell_results: List[Dict[str, Any]] = []
    approx_ratios: List[float] = []

    unique_cells = np.unique(cell_ids)
    for cid in unique_cells:
        cell_match_indices = np.where(cell_ids == cid)[0]
        K_cell = len(cell_match_indices)
        if K_cell == 0:
            continue

        c_coords = src_xy[cell_match_indices]
        c_conf = confidence[cell_match_indices]

        # Deterministic per-cell seed
        cell_seed = int((seed + cid * 997) % 2147483647)

        res = select_cell_qaoa(
            coords=c_coords,
            confidence=c_conf,
            k=cap,
            cell_size=cell_size,
            mu=mu,
            p_layers=p_layers,
            restarts=restarts,
            seed=cell_seed,
        )
        cell_results.append(res)
        approx_ratios.append(res["approx_ratio"])

        sol = res["solution"]
        # If sol length matches K_cell, assign mask
        if len(sol) == K_cell:
            for idx_local, is_kept in enumerate(sol):
                if is_kept == 1:
                    keep_mask[cell_match_indices[idx_local]] = True
        else:
            # Fallback if truncated to 12
            for idx_local in range(min(len(sol), K_cell)):
                if sol[idx_local] == 1:
                    keep_mask[cell_match_indices[idx_local]] = True

    runtime_s = time.time() - t0

    # Classical twin A computation: grid_cap
    from src.selection.spatial import grid_cap
    t_twin_0 = time.time()
    t_src, t_ref, t_conf = grid_cap(src_xy, ref_xy, confidence, n=n, cap=cap, image_shape=image_shape)
    runtime_twin_s = time.time() - t_twin_0

    mean_ar = float(np.mean(approx_ratios)) if approx_ratios else 1.0
    min_ar = float(np.min(approx_ratios)) if approx_ratios else 1.0
    ratio_above_95 = float(np.mean([1.0 if r >= 0.95 else 0.0 for r in approx_ratios])) if approx_ratios else 1.0

    metrics = {
        "n_candidates_total": N,
        "n_selected_qaoa": int(np.sum(keep_mask)),
        "n_selected_grid_cap": len(t_src),
        "n_cells_occupied": len(unique_cells),
        "approx_ratio_mean": round(mean_ar, 4),
        "approx_ratio_min": round(min_ar, 4),
        "approx_ratio_above_095": round(ratio_above_95, 4),
        "p_layers": p_layers,
        "runtime_s": round(runtime_s, 4),
        "classical_twin_runtime_s": round(runtime_twin_s, 6),
    }

    return src_xy[keep_mask], ref_xy[keep_mask], confidence[keep_mask], metrics

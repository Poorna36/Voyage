"""
quantum/classical_baseline.py
==============================
Classical Hungarian Algorithm baseline for QUBO correspondence problems.

Provides the provably-optimal assignment under a linear cost model for
direct benchmarking against Quantum Annealing and QAOA solutions.

Uses the Jonker-Volgenant or a NumPy-native Hungarian implementation
(no scipy.optimize.linear_sum_assignment dependency required, though
that is used as a fast C-backed fallback when available).

References
----------
- Kuhn (1955), "The Hungarian Method for the Assignment Problem",
  Naval Research Logistics Quarterly.
- quantum/QUANTUM_COMPUTING_RESEARCH.txt §3 (classical pipeline)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)


class HungarianSolver:
    """
    Solves the linear assignment problem using the Hungarian algorithm.

    Given a cost matrix C (N_s × N_r), finds the minimum-cost matching.
    For the QUBO correspondence problem, C = -U (negative unary affinity)
    so that maximum affinity = minimum cost.

    Parameters
    ----------
    confidence_threshold : float
        Minimum unary affinity for a match to be accepted post-assignment.
    """

    def __init__(self, confidence_threshold: float = 0.20) -> None:
        self.conf_thresh = float(confidence_threshold)

    def solve(
        self,
        unary: np.ndarray,
        pairs: np.ndarray,
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
        Q: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal correspondence using Hungarian assignment.

        Parameters
        ----------
        unary  : (N_s, N_r) float64 — unary affinity matrix
        pairs  : (M, 2) int — candidate (i,j) pairs from QUBOBuilder
        src_xy : (N_s, 2) float32 — source keypoints (col, row)
        ref_xy : (N_r, 2) float32 — reference keypoints (col, row)
        Q      : (M, M) float64 — optional QUBO matrix for fair energy evaluation

        Returns
        -------
        dict with:
            'solution'          : (M,) int binary vector (1 = match accepted)
            'src_xy'            : (K, 2) accepted source coords
            'ref_xy'            : (K, 2) accepted reference coords
            'confidence'        : (K,) affinity scores of accepted matches
            'energy'            : float — full QUBO energy x^T Q x (if Q provided) or negative affinity
            'linear_cost_energy': float — negative total affinity
            'runtime_s'         : float
            'solver'            : 'hungarian'
        """
        t0 = time.time()
        N_s, N_r = unary.shape

        if N_s == 0 or N_r == 0:
            return self._empty_result(len(pairs), t0)

        # Cost matrix = negative unary (minimising cost = maximising affinity)
        cost = -unary.copy()

        # Run Hungarian algorithm
        row_ind, col_ind = self._hungarian(cost)

        # Accept only matches above threshold
        matched_src_xy = []
        matched_ref_xy = []
        matched_conf = []
        accepted_pairs = set()

        for r, c in zip(row_ind, col_ind):
            aff = unary[r, c]
            if aff >= self.conf_thresh:
                matched_src_xy.append(src_xy[r])
                matched_ref_xy.append(ref_xy[c])
                matched_conf.append(float(aff))
                accepted_pairs.add((int(r), int(c)))

        # Build binary solution vector aligned to the QUBO pairs list
        solution = np.zeros(len(pairs), dtype=np.int32)
        for k, (i, j) in enumerate(pairs):
            if (int(i), int(j)) in accepted_pairs:
                solution[k] = 1

        linear_energy = -float(sum(matched_conf))  # Linear assignment cost
        if Q is not None and len(solution) > 0:
            qubo_energy = float(solution @ Q @ solution)
        else:
            qubo_energy = linear_energy

        runtime = time.time() - t0
        log.debug(
            "Hungarian: %d/%d matches accepted, linear_cost=%.4f, qubo_energy=%.4f, t=%.3fs",
            len(matched_conf), len(row_ind), linear_energy, qubo_energy, runtime,
        )

        out_src = np.array(matched_src_xy, dtype=np.float32) if matched_src_xy else np.empty((0, 2), dtype=np.float32)
        out_ref = np.array(matched_ref_xy, dtype=np.float32) if matched_ref_xy else np.empty((0, 2), dtype=np.float32)
        out_conf = np.array(matched_conf, dtype=np.float32)

        return {
            "solution": solution,
            "src_xy": out_src,
            "ref_xy": out_ref,
            "confidence": out_conf,
            "energy": qubo_energy,
            "linear_cost_energy": linear_energy,
            "runtime_s": runtime,
            "solver": "hungarian",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Hungarian implementation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _hungarian(cost: np.ndarray):
        """
        Solve the linear assignment problem.
        Falls back to scipy if available (faster C backend), otherwise
        uses a pure-NumPy O(n³) implementation.
        """
        try:
            from scipy.optimize import linear_sum_assignment
            return linear_sum_assignment(cost)
        except ImportError:
            return HungarianSolver._hungarian_numpy(cost)

    @staticmethod
    def _hungarian_numpy(cost: np.ndarray):
        """
        Pure-NumPy Hungarian algorithm (O(n³)).
        Handles rectangular matrices (N_s × N_r, N_s ≤ N_r).
        """
        C = np.array(cost, dtype=np.float64)
        n, m = C.shape

        # Pad to square if needed
        if n > m:
            C = np.hstack([C, np.full((n, n - m), C.max() + 1e9)])
            m = n
        elif m > n:
            C = np.vstack([C, np.full((m - n, m), C.max() + 1e9)])
            n = m

        N = n
        # Step 1: subtract row minimums
        C -= C.min(axis=1, keepdims=True)
        # Step 2: subtract col minimums
        C -= C.min(axis=0, keepdims=True)

        # Step 3–7: cover zeros and augment
        assignment = np.full(N, -1, dtype=np.int32)
        for _ in range(N * 2):
            # Find zero positions
            zero_rows, zero_cols = np.where(C == 0)
            if len(zero_rows) == 0:
                break
            # Greedy assignment of uncovered zeros
            assigned_rows, assigned_cols = set(), set()
            for r, c in zip(zero_rows, zero_cols):
                if r not in assigned_rows and c not in assigned_cols:
                    assignment[r] = c
                    assigned_rows.add(r)
                    assigned_cols.add(c)

            if len(assigned_rows) == N:
                break

            # Cover rows/cols without assignments and with zeros
            covered_rows = set(assigned_rows)
            covered_cols = set()
            for r, c in zip(zero_rows, zero_cols):
                if r not in covered_rows:
                    covered_cols.add(c)
            for r, c in zip(zero_rows, zero_cols):
                if c in covered_cols and r not in covered_rows:
                    covered_rows.add(r)

            uncovered_mask = np.ones((N, N), dtype=bool)
            for r in covered_rows:
                uncovered_mask[r, :] = False
            for c in covered_cols:
                uncovered_mask[:, c] = False

            uncovered_vals = C[uncovered_mask]
            if len(uncovered_vals) == 0:
                break
            min_val = uncovered_vals.min()
            if min_val <= 0:
                break

            C[uncovered_mask] -= min_val
            for r in covered_rows:
                for c_ in covered_cols:
                    C[r, c_] += min_val

        # Extract result for original (pre-pad) dimensions
        orig_n, orig_m = cost.shape
        row_ind = np.where(assignment >= 0)[0]
        col_ind = assignment[row_ind]

        # Filter to original dimensions
        mask = (row_ind < orig_n) & (col_ind < orig_m)
        return row_ind[mask], col_ind[mask]

    @staticmethod
    def _empty_result(M: int, t0: float) -> Dict[str, Any]:
        return {
            "solution": np.zeros(M, dtype=np.int32),
            "src_xy": np.empty((0, 2), dtype=np.float32),
            "ref_xy": np.empty((0, 2), dtype=np.float32),
            "confidence": np.empty(0, dtype=np.float32),
            "energy": 0.0,
            "runtime_s": time.time() - t0,
            "solver": "hungarian",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Classical QUBO Solver: Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalSimulatedAnnealer:
    """
    Classical Simulated Annealing baseline for QUBO problems.

    Minimises H(x) = x^T Q x over x ∈ {0,1}^M using single-bit flip Metropolis updates.
    Provides a fair classical benchmark against Simulated Quantum Annealing (SQA) on
    the exact same objective function.
    """

    def __init__(
        self,
        n_sweeps: int = 1000,
        n_reads: int = 5,
        T_start: float = 5.0,
        T_end: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_sweeps = int(n_sweeps)
        self.n_reads = int(n_reads)
        self.T_start = float(T_start)
        self.T_end = float(T_end)
        self.seed = int(seed)

    def solve(self, Q: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        Q = np.asarray(Q, dtype=np.float64)
        M = Q.shape[0]

        if M == 0:
            return {
                "solution": np.zeros(0, dtype=np.int32),
                "energy": 0.0,
                "all_energies": [],
                "runtime_s": time.time() - t0,
                "n_vars": 0,
                "solver": "simulated_annealing_classical",
            }

        Q_sym = (Q + Q.T) / 2.0
        h = np.diag(Q_sym).copy()
        rng = np.random.default_rng(self.seed)

        best_x: Optional[np.ndarray] = None
        best_e = np.inf
        all_energies = []
        T_schedule = np.geomspace(self.T_start, self.T_end, self.n_sweeps)

        for read in range(self.n_reads):
            x = rng.integers(0, 2, size=M, dtype=np.int32)
            cur_e = float(x @ Q_sym @ x)
            read_best_x = x.copy()
            read_best_e = cur_e

            for T in T_schedule:
                for i in rng.permutation(M):
                    delta = 1 - 2 * x[i]
                    dE = (h[i] + 2.0 * (Q_sym[i] @ x - h[i] * x[i])) * delta
                    if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                        x[i] = 1 - x[i]
                        cur_e += dE
                        if cur_e < read_best_e:
                            read_best_e = cur_e
                            read_best_x = x.copy()

            all_energies.append(float(read_best_e))
            if read_best_e < best_e:
                best_e = read_best_e
                best_x = read_best_x.copy()

        runtime = time.time() - t0
        return {
            "solution": best_x.astype(np.int32) if best_x is not None else np.zeros(M, dtype=np.int32),
            "energy": float(best_e),
            "all_energies": all_energies,
            "runtime_s": runtime,
            "n_vars": M,
            "solver": "simulated_annealing_classical",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Plain Feature Baseline: SIFT + Lowe (0.75) + RANSAC Homography
# ─────────────────────────────────────────────────────────────────────────────

def sift_ransac_baseline(
    src: np.ndarray,
    ref: np.ndarray,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 3.0,
) -> Dict[str, Any]:
    """
    Standard classical baseline: SIFT feature detection & description,
    Lowe ratio test filtering, and RANSAC homography estimation.
    """
    t0 = time.time()
    try:
        import cv2
    except ImportError:
        return {
            "src_xy": np.empty((0, 2), dtype=np.float32),
            "ref_xy": np.empty((0, 2), dtype=np.float32),
            "runtime_s": time.time() - t0,
            "solver": "sift_ransac",
            "n_matches": 0,
        }

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    kp2, des2 = sift.detectAndCompute(ref_gray, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return {
            "src_xy": np.empty((0, 2), dtype=np.float32),
            "ref_xy": np.empty((0, 2), dtype=np.float32),
            "runtime_s": time.time() - t0,
            "solver": "sift_ransac",
            "n_matches": 0,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_tuple in matches:
        if len(m_tuple) >= 1:
            m = m_tuple[0]
            if ratio_thresh is None or ratio_thresh >= 1.0:
                good.append(m)
            elif len(m_tuple) >= 2 and m.distance < ratio_thresh * m_tuple[1].distance:
                good.append(m)

    if len(good) < 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]) if good else np.empty((0, 2), dtype=np.float32)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]) if good else np.empty((0, 2), dtype=np.float32)
        return {
            "src_xy": src_pts,
            "ref_xy": dst_pts,
            "runtime_s": time.time() - t0,
            "solver": "sift_ransac",
            "n_matches": len(good),
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    H_est, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if inlier_mask is not None:
        mask = inlier_mask.ravel() == 1
        src_pts = src_pts[mask]
        dst_pts = dst_pts[mask]

    runtime = time.time() - t0
    return {
        "src_xy": src_pts,
        "ref_xy": dst_pts,
        "runtime_s": runtime,
        "solver": "sift_ransac",
        "n_matches": len(src_pts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spec v2 §3E: Exact Brute-Force QUBO Solver & Greedy Compatibility Baseline
# ─────────────────────────────────────────────────────────────────────────────

class ExactQUBOSolver:
    """
    Exact brute-force QUBO solver for small instances (M <= 16) (Spec v2 §3E).
    Evaluates all 2^M bitstrings to compute the true global optimum and verify QAOA solution quality.
    """

    def __init__(self, max_vars: int = 16) -> None:
        self.max_vars = int(max_vars)

    def solve(self, Q: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()
        Q = np.asarray(Q, dtype=np.float64)
        M = Q.shape[0]
        if M == 0:
            return {
                "solution": np.zeros(0, dtype=np.int32),
                "energy": 0.0,
                "runtime_s": 0.0,
                "solver": "exact_bruteforce",
                "n_states_evaluated": 0,
            }
        if M > self.max_vars:
            raise ValueError(f"Exact solver limited to M <= {self.max_vars}; got {M}")

        n_states = 1 << M
        # Generate all bitstrings
        idxs = ((np.arange(n_states)[:, None] & (1 << np.arange(M))) > 0).astype(np.float64)
        energies = np.sum((idxs @ Q) * idxs, axis=1)
        best_idx = int(np.argmin(energies))
        min_e = float(energies[best_idx])
        best_sol = idxs[best_idx].astype(np.int32)

        return {
            "solution": best_sol,
            "energy": min_e,
            "best_bitstring": "".join(map(str, best_sol)),
            "all_energies": energies,
            "runtime_s": time.time() - t0,
            "n_states_evaluated": n_states,
            "solver": "exact_bruteforce",
        }


class GreedyCompatibilitySolver:
    """
    Greedy compatibility selection baseline for correspondence optimization (Spec v2 §3E).
    Iteratively picks candidate matches with highest average consistency with already-selected set.
    """

    def __init__(self, max_matches: Optional[int] = None, min_consistency: float = 0.0) -> None:
        self.max_matches = max_matches
        self.min_consistency = min_consistency

    def solve(
        self,
        Q: np.ndarray,
        pairs: Optional[np.ndarray] = None,
        unary_affinities: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        Q = np.asarray(Q, dtype=np.float64)
        M = Q.shape[0]
        if M == 0:
            return {
                "solution": np.zeros(0, dtype=np.int32),
                "energy": 0.0,
                "selected_indices": [],
                "runtime_s": 0.0,
                "solver": "greedy_compatibility",
            }

        if unary_affinities is not None and len(unary_affinities) == M:
            unary = np.asarray(unary_affinities, dtype=np.float64)
        else:
            unary = -np.diag(Q).copy()

        Q_sym = (Q + Q.T) / 2.0
        pairwise_compat = -Q_sym.copy()
        np.fill_diagonal(pairwise_compat, 0.0)

        selected: List[int] = []
        unselected = set(range(M))
        has_pairs = pairs is not None and len(pairs) == M

        # Step 1: Pick match with highest unary affinity
        best_first = int(np.argmax(unary))
        if unary[best_first] > 0 or len(unselected) == 1:
            selected.append(best_first)
            unselected.remove(best_first)

        # Step 2: Iteratively add match with highest marginal gain
        max_k = self.max_matches or M
        while len(selected) < max_k and unselected:
            best_candidate = None
            best_marginal_gain = -np.inf

            for cand in unselected:
                if has_pairs:
                    src_k, ref_k = pairs[cand]
                    conflict = False
                    for s in selected:
                        if pairs[s][0] == src_k or pairs[s][1] == ref_k:
                            conflict = True
                            break
                    if conflict:
                        continue

                avg_compat = float(np.mean([pairwise_compat[cand, s] for s in selected])) if selected else 0.0
                marginal_gain = unary[cand] + avg_compat

                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_candidate = cand

            if best_candidate is None or best_marginal_gain < self.min_consistency:
                break

            selected.append(best_candidate)
            unselected.remove(best_candidate)

        solution = np.zeros(M, dtype=np.int32)
        solution[selected] = 1
        energy = float(solution @ Q @ solution)

        return {
            "solution": solution,
            "energy": energy,
            "selected_indices": selected,
            "runtime_s": time.time() - t0,
            "solver": "greedy_compatibility",
        }


def compute_constraint_diagnostics(
    solution: np.ndarray,
    pairs: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    exact_energy: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute constraint violations and quality metrics for a correspondence solution.
    """
    sol = np.asarray(solution, dtype=np.int32)
    selected_idx = np.where(sol == 1)[0]
    n_selected = len(selected_idx)

    src_conflicts = 0
    ref_conflicts = 0
    if pairs is not None and len(pairs) == len(sol):
        sel_pairs = pairs[selected_idx]
        if len(sel_pairs) > 0:
            src_counts: Dict[int, int] = {}
            ref_counts: Dict[int, int] = {}
            for s, r in sel_pairs:
                src_counts[int(s)] = src_counts.get(int(s), 0) + 1
                ref_counts[int(r)] = ref_counts.get(int(r), 0) + 1
            src_conflicts = sum(v - 1 for v in src_counts.values() if v > 1)
            ref_conflicts = sum(v - 1 for v in ref_counts.values() if v > 1)

    energy = float(sol @ Q @ sol) if Q is not None and len(sol) > 0 else 0.0
    gap = (energy - exact_energy) if exact_energy is not None else None

    return {
        "n_selected": n_selected,
        "src_conflicts": src_conflicts,
        "ref_conflicts": ref_conflicts,
        "total_violations": src_conflicts + ref_conflicts,
        "is_valid": (src_conflicts + ref_conflicts) == 0,
        "energy": energy,
        "gap_to_exact": gap,
    }


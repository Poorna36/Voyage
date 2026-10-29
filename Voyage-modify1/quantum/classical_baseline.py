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

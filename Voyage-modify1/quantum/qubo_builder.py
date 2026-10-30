"""
quantum/qubo_builder.py
=======================
Constructs the QUBO / Ising Hamiltonian for keypoint correspondence.

Problem Formulation
-------------------
Given:
  P = {p_i} — N_s keypoints in source image (Chandrayaan-2 OHRC/TMC/IIRS)
  Q = {q_j} — N_r keypoints in reference image (LRO NAC/WAC)

Binary decision variables:
  x_{i,j} = 1  iff  p_i matches q_j,  else 0

Objective (minimise):
  H(x) = - Σ_{i,j}     U(p_i, q_j)         * x_{i,j}          [Unary affinity]
          - Σ_{(i,j),(k,l)} W_{ijkl}         * x_{i,j} * x_{k,l} [Pairwise geometry]
          + λ_row * Σ_i (Σ_j x_{i,j} - 1)²                      [One-to-one: rows]
          + λ_col * Σ_j (Σ_i x_{i,j} - 1)²                      [One-to-one: cols]

Unary affinity U(p_i, q_j):
  - Descriptor cosine similarity (SIFT/RIFT descriptor vectors)
  - Scale compatibility: exp(-|log(s_i/s_j) - log(gsd_ratio)|)
  - Combined: alpha * desc_sim + (1 - alpha) * scale_compat

Pairwise geometric consistency W_{ijkl}:
  - Distance-ratio invariant: | ||p_i - p_k|| / ||p_j - p_l|| - 1 | < epsilon
  - Angle invariant:          | angle(p_i,p_k) - angle(q_j,q_l) | < delta
  - W_{ijkl} = exp(-dist_ratio_error - angle_error)

References
----------
- quantum/QUANTUM_COMPUTING_RESEARCH.txt §4 (QUBO formulation, §4.2)
- FEATURES.md F09, F12 (correspondence problem)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class QUBOBuilder:
    """
    Constructs the QUBO matrix Q such that H(x) = x^T Q x.

    The variable ordering is row-major:  variable index = i * N_r + j
    for the (i,j) pair.

    Parameters
    ----------
    lambda_penalty : float
        Lagrange multiplier for one-to-one constraint penalties.
        Typical range: 2.0 – 10.0. Higher values enforce hard constraints
        at the cost of energy landscape flatness.
    alpha : float
        Weight balancing descriptor affinity vs. geometric consistency.
        0 = pure geometry, 1 = pure descriptor.
    pair_weight : float
        Weight for pairwise geometric consistency term W.
    scale_sigma : float
        Sigma for Gaussian scale-compatibility kernel.
    angle_sigma_deg : float
        Sigma (degrees) for Gaussian angle-compatibility kernel.
    dist_ratio_eps : float
        Tolerance for distance-ratio geometric consistency test.
    max_pairs : int
        Cap on number of (i,j) candidate pairs fed to QUBO (memory guard).
        Pairs with U < unary_threshold are pruned first.
    unary_threshold : float
        Minimum unary affinity to include a candidate pair (pre-QUBO pruning).
    formulation_version : str
        'v1' (original) or 'v2' (similarity-invariant pairwise + dynamic lambda + MNN).
    ratio_thresh : float
        Lowe-style ratio test threshold for candidate filtering in v2.
    use_mutual_nn : bool
        Whether to enforce mutual nearest neighbour in candidate filtering (v2).
    use_ratio_test : bool
        Whether to enforce Lowe ratio test in candidate filtering (v2).
    dynamic_lambda : bool
        Whether to set lambda dynamically from maximum single-variable gain (v2).
    lambda_margin : float
        Multiplier above max single-variable gain for dynamic lambda (v2).
    """

    def __init__(
        self,
        lambda_penalty: float = 5.0,
        alpha: float = 0.6,
        pair_weight: float = 1.0,
        scale_sigma: float = 0.5,
        angle_sigma_deg: float = 30.0,
        dist_ratio_eps: float = 0.25,
        max_pairs: int = 200,
        unary_threshold: float = 0.10,
        formulation_version: str = "v2",
        ratio_thresh: float = 0.80,
        use_mutual_nn: bool = True,
        use_ratio_test: bool = True,
        dynamic_lambda: bool = True,
        lambda_margin: float = 1.25,
    ) -> None:
        self.lambda_penalty = float(lambda_penalty)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.pair_weight = float(pair_weight)
        self.scale_sigma = float(scale_sigma)
        self.angle_sigma_deg = float(angle_sigma_deg)
        self.dist_ratio_eps = float(dist_ratio_eps)
        self.max_pairs = int(max_pairs)
        self.unary_threshold = float(unary_threshold)
        self.formulation_version = str(formulation_version).lower()
        self.ratio_thresh = float(ratio_thresh)
        self.use_mutual_nn = bool(use_mutual_nn)
        self.use_ratio_test = bool(use_ratio_test)
        self.dynamic_lambda = bool(dynamic_lambda)
        self.lambda_margin = float(lambda_margin)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
        src_desc: Optional[np.ndarray] = None,
        ref_desc: Optional[np.ndarray] = None,
        src_scales: Optional[np.ndarray] = None,
        ref_scales: Optional[np.ndarray] = None,
        src_angles: Optional[np.ndarray] = None,
        ref_angles: Optional[np.ndarray] = None,
        gsd_ratio: float = 1.0,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Build the QUBO problem from keypoint arrays.

        Parameters
        ----------
        src_xy  : (N_s, 2) float32 — (col, row) source keypoints
        ref_xy  : (N_r, 2) float32 — (col, row) reference keypoints
        src_desc: (N_s, D) float32 — descriptor vectors (optional)
        ref_desc: (N_r, D) float32 — descriptor vectors (optional)
        src_scales, ref_scales: (N,) float32 — keypoint scale sizes
        src_angles, ref_angles: (N,) float32 — keypoint orientations (deg)
        gsd_ratio: float — src_GSD / ref_GSD
        candidate_mask: (N_s, N_r) bool — pre-filter for candidate pairs

        Returns
        -------
        dict with keys:
            'Q'           : (M, M) float64 QUBO matrix (M = number of active pairs)
            'pairs'       : (M, 2) int array — (i, j) index of each QUBO variable
            'unary'       : (N_s, N_r) float64 full unary affinity matrix
            'N_s', 'N_r'  : int
            'n_vars'      : int — M (QUBO size)
            'meta'        : dict with configuration snapshot
        """
        src_xy = np.asarray(src_xy, dtype=np.float64)
        ref_xy = np.asarray(ref_xy, dtype=np.float64)
        N_s, N_r = len(src_xy), len(ref_xy)

        log.debug("QUBOBuilder.build: N_s=%d, N_r=%d", N_s, N_r)

        # ── 1. Unary affinity matrix U (N_s × N_r) ──────────────────────────
        U = self._compute_unary(
            src_xy, ref_xy,
            src_desc, ref_desc,
            src_scales, ref_scales,
            src_angles, ref_angles,
            gsd_ratio,
        )

        # ── 2. Prune candidates by threshold & max_pairs ─────────────────────
        pairs = self._select_candidate_pairs(
            U, candidate_mask, src_desc=src_desc, ref_desc=ref_desc
        )
        M = len(pairs)
        log.debug("  Active QUBO variables (pairs): %d", M)

        if M == 0:
            return self._empty_result(N_s, N_r)

        # ── 3. Allocate QUBO matrix ──────────────────────────────────────────
        Q = np.zeros((M, M), dtype=np.float64)

        # ── 4. Diagonal: negative unary affinity (minimising → reward matches)
        for k, (i, j) in enumerate(pairs):
            Q[k, k] -= U[i, j]

        # ── 5. Off-diagonal: pairwise geometric consistency W ─────────────────
        pair_arr = np.array(pairs, dtype=np.int32)
        self._fill_pairwise(
            Q, pair_arr, src_xy, ref_xy,
            src_scales=src_scales, ref_scales=ref_scales,
            src_angles=src_angles, ref_angles=ref_angles,
            gsd_ratio=gsd_ratio,
        )

        # ── 6. One-to-one penalty ─────────────────────────────────────────────
        effective_lambda = self.lambda_penalty
        if self.formulation_version == "v2" and self.dynamic_lambda:
            max_gain = 0.0
            for k in range(M):
                unary_gain = U[pair_arr[k, 0], pair_arr[k, 1]]
                pairwise_gain = 0.0
                for l in range(M):
                    if k != l and pair_arr[k, 0] != pair_arr[l, 0] and pair_arr[k, 1] != pair_arr[l, 1]:
                        pairwise_gain += max(0.0, -Q[k, l])
                gain_k = unary_gain + pairwise_gain
                if gain_k > max_gain:
                    max_gain = gain_k
            effective_lambda = max(self.lambda_penalty, self.lambda_margin * max_gain)

        self._fill_one_to_one(Q, pair_arr, N_s, N_r, effective_lambda)

        return {
            "Q": Q,
            "pairs": pair_arr,
            "unary": U,
            "N_s": N_s,
            "N_r": N_r,
            "n_vars": M,
            "meta": {
                "formulation_version": self.formulation_version,
                "lambda_penalty": effective_lambda,
                "base_lambda": self.lambda_penalty,
                "alpha": self.alpha,
                "pair_weight": self.pair_weight,
                "unary_threshold": self.unary_threshold,
                "max_pairs": self.max_pairs,
                "gsd_ratio": gsd_ratio,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Unary affinity
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_unary(
        self,
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
        src_desc: Optional[np.ndarray],
        ref_desc: Optional[np.ndarray],
        src_scales: Optional[np.ndarray],
        ref_scales: Optional[np.ndarray],
        src_angles: Optional[np.ndarray],
        ref_angles: Optional[np.ndarray],
        gsd_ratio: float,
    ) -> np.ndarray:
        N_s, N_r = len(src_xy), len(ref_xy)
        U = np.zeros((N_s, N_r), dtype=np.float64)

        # Descriptor cosine similarity
        if src_desc is not None and ref_desc is not None:
            sd = np.asarray(src_desc, dtype=np.float64)
            rd = np.asarray(ref_desc, dtype=np.float64)
            # L2-normalise
            sd_norm = sd / (np.linalg.norm(sd, axis=1, keepdims=True) + 1e-9)
            rd_norm = rd / (np.linalg.norm(rd, axis=1, keepdims=True) + 1e-9)
            desc_sim = (sd_norm @ rd_norm.T + 1.0) / 2.0  # map [-1,1] → [0,1]
        else:
            # Without descriptors: uniform affinity (pure geometry mode)
            desc_sim = np.ones((N_s, N_r), dtype=np.float64) * 0.5

        # Scale compatibility
        if src_scales is not None and ref_scales is not None:
            ss = np.asarray(src_scales, dtype=np.float64)
            rs = np.asarray(ref_scales, dtype=np.float64)
            log_ratio = np.log(ss[:, None] / (rs[None, :] + 1e-9) + 1e-9)
            log_expected = np.log(max(gsd_ratio, 1e-9))
            scale_compat = np.exp(
                -((log_ratio - log_expected) ** 2) / (2 * self.scale_sigma ** 2)
            )
        else:
            scale_compat = np.ones((N_s, N_r), dtype=np.float64)

        # Angle compatibility
        if src_angles is not None and ref_angles is not None:
            sa = np.asarray(src_angles, dtype=np.float64)
            ra = np.asarray(ref_angles, dtype=np.float64)
            angle_diff = np.abs(sa[:, None] - ra[None, :]) % 360.0
            angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)
            angle_compat = np.exp(
                -(angle_diff ** 2) / (2 * self.angle_sigma_deg ** 2)
            )
        else:
            angle_compat = np.ones((N_s, N_r), dtype=np.float64)

        # Combined unary
        U = (
            self.alpha * desc_sim
            + (1.0 - self.alpha) * 0.5 * (scale_compat + angle_compat)
        )
        return np.clip(U, 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate pair selection
    # ─────────────────────────────────────────────────────────────────────────

    def _select_candidate_pairs(
        self,
        U: np.ndarray,
        candidate_mask: Optional[np.ndarray] = None,
        src_desc: Optional[np.ndarray] = None,
        ref_desc: Optional[np.ndarray] = None,
    ) -> list:
        """Return list of (i, j) pairs above threshold, capped at max_pairs."""
        N_s, N_r = U.shape
        if (
            self.formulation_version == "v2"
            and src_desc is not None
            and ref_desc is not None
            and (self.use_ratio_test or self.use_mutual_nn)
        ):
            return self._select_candidate_pairs_v2(U, src_desc, ref_desc, candidate_mask)

        if candidate_mask is not None:
            mask = np.asarray(candidate_mask, dtype=bool) & (U >= self.unary_threshold)
        else:
            mask = U >= self.unary_threshold

        pairs_i, pairs_j = np.where(mask)
        if len(pairs_i) == 0:
            return []

        # Sort by descending unary affinity, keep top max_pairs
        scores = U[pairs_i, pairs_j]
        order = np.argsort(-scores)
        pairs_i = pairs_i[order[: self.max_pairs]]
        pairs_j = pairs_j[order[: self.max_pairs]]

        return list(zip(pairs_i.tolist(), pairs_j.tolist()))

    def _select_candidate_pairs_v2(
        self,
        U: np.ndarray,
        src_desc: np.ndarray,
        ref_desc: np.ndarray,
        candidate_mask: Optional[np.ndarray],
    ) -> list:
        """
        v2 candidate selection: Lowe ratio test and/or Mutual Nearest Neighbour.
        Falls back to top unary affinity pairs if fewer than 4 pairs pass.
        """
        N_s, N_r = U.shape
        sd = np.asarray(src_desc, dtype=np.float32)
        rd = np.asarray(ref_desc, dtype=np.float32)

        # L2-normalise for cosine distance
        sd_norm = sd / (np.linalg.norm(sd, axis=1, keepdims=True) + 1e-9)
        rd_norm = rd / (np.linalg.norm(rd, axis=1, keepdims=True) + 1e-9)

        sim_mat = sd_norm @ rd_norm.T  # (N_s, N_r) in [-1, 1]
        dist_mat = 1.0 - sim_mat       # in [0, 2]

        ratio_passed = set()
        for i in range(N_s):
            dists = dist_mat[i]
            if len(dists) >= 2:
                idx_sorted = np.argpartition(dists, 1)[:2]
                idx_sorted = idx_sorted[np.argsort(dists[idx_sorted])]
                j1, j2 = idx_sorted[0], idx_sorted[1]
                if self.use_ratio_test:
                    if dists[j1] < self.ratio_thresh * dists[j2]:
                        ratio_passed.add((i, j1))
                else:
                    ratio_passed.add((i, j1))
            elif len(dists) == 1:
                ratio_passed.add((i, 0))

        bwd_nn = {}
        if self.use_mutual_nn:
            for j in range(N_r):
                dists = dist_mat[:, j]
                if len(dists) > 0:
                    bwd_nn[j] = int(np.argmin(dists))

        filtered_pairs = []
        for (i, j) in ratio_passed:
            if candidate_mask is not None and not candidate_mask[i, j]:
                continue
            if U[i, j] < self.unary_threshold:
                continue
            if self.use_mutual_nn:
                if bwd_nn.get(j) != i:
                    continue
            filtered_pairs.append((i, j))

        # Fallback to top unary if filtering produced fewer than 4 pairs
        if len(filtered_pairs) < 4:
            mask = U >= self.unary_threshold
            if candidate_mask is not None:
                mask = mask & candidate_mask
            pi, pj = np.where(mask)
            if len(pi) > 0:
                scores = U[pi, pj]
                order = np.argsort(-scores)
                existing = set(filtered_pairs)
                for idx in order:
                    pair = (int(pi[idx]), int(pj[idx]))
                    if pair not in existing:
                        filtered_pairs.append(pair)
                        existing.add(pair)
                        if len(filtered_pairs) >= self.max_pairs:
                            break

        # Sort by unary affinity descending and keep top max_pairs
        filtered_pairs.sort(key=lambda p: float(U[p[0], p[1]]), reverse=True)
        return filtered_pairs[: self.max_pairs]

    # ─────────────────────────────────────────────────────────────────────────
    # Pairwise geometric consistency
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_pairwise(
        self,
        Q: np.ndarray,
        pairs: np.ndarray,
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
        src_scales: Optional[np.ndarray] = None,
        ref_scales: Optional[np.ndarray] = None,
        src_angles: Optional[np.ndarray] = None,
        ref_angles: Optional[np.ndarray] = None,
        gsd_ratio: float = 1.0,
    ) -> None:
        """
        Fill off-diagonal QUBO terms encoding geometric consistency.
        In v1: absolute displacement angle and unscaled distance ratio.
        In v2: similarity-invariant distance ratio d_ref / (d_src * s) and
               relative orientation (psi - theta) difference.
        """
        M = len(pairs)
        w = self.pair_weight
        eps = self.dist_ratio_eps

        if self.formulation_version == "v1":
            for k in range(M):
                i, j = pairs[k]
                for l in range(k + 1, M):
                    p, q = pairs[l]
                    if i == p or j == q:
                        continue

                    d_src = np.linalg.norm(src_xy[i] - src_xy[p]) + 1e-9
                    d_ref = np.linalg.norm(ref_xy[j] - ref_xy[q]) + 1e-9
                    ratio_err = abs(d_src / d_ref - 1.0)

                    # Angle between displacement vectors
                    v_src = src_xy[p] - src_xy[i]
                    v_ref = ref_xy[q] - ref_xy[j]
                    cos_s = np.arctan2(v_src[1], v_src[0])
                    cos_r = np.arctan2(v_ref[1], v_ref[0])
                    angle_err = abs(cos_s - cos_r) % (2 * np.pi)
                    angle_err = min(angle_err, 2 * np.pi - angle_err) / np.pi

                    geo_score = np.exp(-(ratio_err / eps) ** 2) * np.exp(-(angle_err * 3.0) ** 2)
                    Q[k, l] -= w * geo_score
                    Q[l, k] -= w * geo_score
            return

        # Formulation v2: Similarity-invariant pairwise terms
        for k in range(M):
            i, j = pairs[k]
            for l in range(k + 1, M):
                p, q = pairs[l]
                if i == p or j == q:
                    continue

                d_src = np.linalg.norm(src_xy[i] - src_xy[p]) + 1e-9
                d_ref = np.linalg.norm(ref_xy[j] - ref_xy[q]) + 1e-9

                # Invariant scale ratio:
                if (
                    src_scales is not None
                    and ref_scales is not None
                    and i < len(src_scales)
                    and p < len(src_scales)
                    and j < len(ref_scales)
                    and q < len(ref_scales)
                ):
                    s_est = 0.5 * (
                        float(ref_scales[j]) / (float(src_scales[i]) + 1e-9)
                        + float(ref_scales[q]) / (float(src_scales[p]) + 1e-9)
                    )
                elif gsd_ratio > 0:
                    s_est = 1.0 / gsd_ratio
                else:
                    s_est = 1.0

                ratio_err = abs(d_ref / (d_src * s_est + 1e-9) - 1.0)

                # Invariant relative orientation:
                v_src = src_xy[p] - src_xy[i]
                v_ref = ref_xy[q] - ref_xy[j]
                psi_src = np.arctan2(v_src[1], v_src[0])
                psi_ref = np.arctan2(v_ref[1], v_ref[0])

                if (
                    src_angles is not None
                    and ref_angles is not None
                    and i < len(src_angles)
                    and p < len(src_angles)
                    and j < len(ref_angles)
                    and q < len(ref_angles)
                ):
                    th_s = np.radians(float(src_angles[i]))
                    th_r = np.radians(float(ref_angles[j]))
                    rel_s = (psi_src - th_s) % (2 * np.pi)
                    rel_r = (psi_ref - th_r) % (2 * np.pi)
                    angle_err = abs(rel_r - rel_s) % (2 * np.pi)
                    angle_err = min(angle_err, 2 * np.pi - angle_err) / np.pi
                else:
                    angle_err = abs(psi_src - psi_ref) % (2 * np.pi)
                    angle_err = min(angle_err, 2 * np.pi - angle_err) / np.pi

                geo_score = np.exp(-(ratio_err / eps) ** 2) * np.exp(-(angle_err * 3.0) ** 2)
                Q[k, l] -= w * geo_score
                Q[l, k] -= w * geo_score

    # ─────────────────────────────────────────────────────────────────────────
    # One-to-one matching penalty
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_one_to_one(
        self,
        Q: np.ndarray,
        pairs: np.ndarray,
        N_s: int,
        N_r: int,
        lam: Optional[float] = None,
    ) -> None:
        """
        Add Lagrange penalty for one-to-one constraint violations.

        For each source keypoint i:  penalty if Σ_j x_{i,j} > 1
        For each reference keypoint j: penalty if Σ_i x_{i,j} > 1
        """
        if lam is None:
            lam = self.lambda_penalty
        M = len(pairs)

        # Build index maps for fast lookup
        row_to_vars: dict = {}   # src index i → list of variable indices k
        col_to_vars: dict = {}   # ref index j → list of variable indices k
        for k, (i, j) in enumerate(pairs):
            row_to_vars.setdefault(i, []).append(k)
            col_to_vars.setdefault(j, []).append(k)

        # Diagonal adjustment (linear term from -2λ * Σ x_{i,j})
        for k in range(M):
            Q[k, k] -= 2.0 * lam

        # Off-diagonal: same source keypoint i (row conflict)
        for i, var_list in row_to_vars.items():
            for a in range(len(var_list)):
                for b in range(a + 1, len(var_list)):
                    k, l = var_list[a], var_list[b]
                    Q[k, l] += 2.0 * lam
                    Q[l, k] += 2.0 * lam

        # Off-diagonal: same reference keypoint j (column conflict)
        for j, var_list in col_to_vars.items():
            for a in range(len(var_list)):
                for b in range(a + 1, len(var_list)):
                    k, l = var_list[a], var_list[b]
                    Q[k, l] += 2.0 * lam
                    Q[l, k] += 2.0 * lam

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def energy(Q: np.ndarray, x: np.ndarray) -> float:
        """Evaluate QUBO energy H = x^T Q x for binary vector x."""
        x = np.asarray(x, dtype=np.float64)
        return float(x @ Q @ x)

    def _empty_result(self, N_s: int, N_r: int) -> Dict[str, Any]:
        return {
            "Q": np.zeros((0, 0), dtype=np.float64),
            "pairs": np.empty((0, 2), dtype=np.int32),
            "unary": np.zeros((N_s, N_r), dtype=np.float64),
            "N_s": N_s,
            "N_r": N_r,
            "n_vars": 0,
            "meta": {},
        }

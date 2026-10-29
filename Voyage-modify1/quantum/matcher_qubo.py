"""
quantum/matcher_qubo.py
=======================
M4 — Quantum QUBO Correspondence Matcher.

Integrates into the L2 Correspondence Engine as a pluggable BaseMatcher,
fully compliant with the pipeline contract (src/matching/base.py).

Architecture
------------
This matcher implements the 3-level Hybrid Classical–Quantum pipeline
described in quantum/QUANTUM_COMPUTING_RESEARCH.txt §5:

  Level 1 (Classical Front-End):
    - SIFT/RIFT keypoint detection and description (via OpenCV, optional)
    - ANMS SSC spatial uniformity filtering
    - Candidate pair pre-filtering (unary affinity threshold)

  Level 2 (Quantum Middle Layer):
    - QUBOBuilder: constructs the QUBO Ising Hamiltonian
    - Solver dispatch:
        'sqa'     → SimulatedQuantumAnnealer (default, no QPU required)
        'qaoa'    → QAOASolver (statevector, M ≤ 18 vars)
        'classical' → HungarianSolver (O(n³) baseline)
    - Solution: binary vector selecting globally-consistent correspondences

  Level 3 (Classical Back-End):
    - Convert selected (i,j) pairs → MatchResult (col, row) coordinates
    - Scale/angle metadata from original keypoints

Pipeline Integration
--------------------
  matcher_id = 'qubo_sqa'   when solver='sqa'
  matcher_id = 'qubo_qaoa'  when solver='qaoa'
  matcher_id = 'qubo_classical' when solver='classical'

  Layer: L2 (src/matching/ registry)
  Requires GPU: False (CPU-only, simulated / quantum-inspired)

References
----------
- src/matching/base.py (BaseMatcher contract)
- quantum/qubo_builder.py (QUBO formulation)
- quantum/annealer.py (SimulatedQuantumAnnealer)
- quantum/qaoa.py (QAOASolver)
- quantum/classical_baseline.py (HungarianSolver)
- quantum/QUANTUM_COMPUTING_RESEARCH.txt §4–5
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

# Pipeline base class
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.matching.base import BaseMatcher, MatchResult
from src.selection.anms import anms_ssc

# Quantum components
from .qubo_builder import QUBOBuilder
from .annealer import SimulatedQuantumAnnealer
from .qaoa import QAOASolver
from .classical_baseline import HungarianSolver, ClassicalSimulatedAnnealer

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    # Solver: 'sqa' | 'qaoa' | 'classical'
    "solver": "sqa",

    # Front-end keypoint extraction
    "num_keypoints": 150,      # ANMS budget per image (keep small for QUBO feasibility)
    "use_sift": True,          # Use SIFT for detection & description
    "ratio_thresh": 0.80,      # Lowe ratio test (relaxed — QUBO handles global selection)

    # QUBO builder parameters
    "formulation_version": "v2", # 'v1' (original) or 'v2' (similarity-invariant, dynamic lambda)
    "lambda_penalty": 5.0,     # One-to-one constraint strength
    "alpha": 0.6,              # Descriptor vs. geometry weight
    "pair_weight": 1.0,        # Pairwise geometric consistency weight
    "max_pairs": 150,          # Maximum QUBO variables (M)
    "unary_threshold": 0.15,   # Minimum affinity to include a pair
    "use_mutual_nn": True,     # Enforce mutual NN in v2 candidate selection
    "use_ratio_test": True,    # Enforce Lowe ratio in v2 candidate selection
    "dynamic_lambda": True,    # Dynamic lambda from max affinity gain
    "lambda_margin": 1.25,     # Margin over max affinity gain for lambda

    # SQA parameters
    "sqa_trotter": 16,
    "sqa_sweeps": 800,
    "sqa_reads": 5,
    "sqa_T_start": 5.0,
    "sqa_T_end": 0.01,
    "sqa_Gamma_start": 3.0,
    "sqa_Gamma_end": 0.001,

    # QAOA parameters
    "qaoa_p_layers": 2,
    "qaoa_restarts": 3,
    "qaoa_shots": 512,

    # Post-quantum filters
    "confidence_threshold": 0.20,   # Min affinity to include in MatchResult
    "max_log_scale_deviation": 0.5, # Scale consistency filter (relaxed for quantum)
}


class QUBOMatcher(BaseMatcher):
    """
    M4 — Hybrid Classical–Quantum QUBO Correspondence Matcher.

    Implements BaseMatcher contract. Returns MatchResult with (col, row)
    coordinates selected by the quantum / classical optimiser.

    matcher_id  = 'qubo_sqa' | 'qubo_qaoa' | 'qubo_classical'
    requires_gpu = False
    """

    requires_gpu = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}

        self.solver_name: str = str(cfg["solver"]).lower()
        self.num_keypoints: int = int(cfg["num_keypoints"])
        self.use_sift: bool = bool(cfg["use_sift"])
        self.ratio_thresh: float = float(cfg["ratio_thresh"])
        self.conf_thresh: float = float(cfg["confidence_threshold"])
        self.mode: str = str(cfg.get("mode", "full")).lower()
        self.pairwise_weight: float = float(cfg.get("pair_weight", 1.0))
        self.lambda_penalty: float = float(cfg.get("lambda_penalty", 5.0))
        self.formulation_version: str = str(cfg.get("formulation_version", "v2")).lower()
        self.max_log_scale_dev: float = float(cfg.get("max_log_scale_deviation", 0.5))

        # QUBO builder
        self.qubo_builder = QUBOBuilder(
            lambda_penalty=float(cfg["lambda_penalty"]),
            alpha=float(cfg["alpha"]),
            pair_weight=float(cfg["pair_weight"]),
            max_pairs=int(cfg["max_pairs"]),
            unary_threshold=float(cfg["unary_threshold"]),
            formulation_version=self.formulation_version,
            ratio_thresh=float(cfg.get("ratio_thresh", 0.80)),
            use_mutual_nn=bool(cfg.get("use_mutual_nn", True)),
            use_ratio_test=bool(cfg.get("use_ratio_test", True)),
            dynamic_lambda=bool(cfg.get("dynamic_lambda", True)),
            lambda_margin=float(cfg.get("lambda_margin", 1.25)),
        )

        # Solvers
        self._sqa = SimulatedQuantumAnnealer(
            n_trotter=int(cfg["sqa_trotter"]),
            n_sweeps=int(cfg["sqa_sweeps"]),
            n_reads=int(cfg["sqa_reads"]),
            T_start=float(cfg["sqa_T_start"]),
            T_end=float(cfg["sqa_T_end"]),
            Gamma_start=float(cfg["sqa_Gamma_start"]),
            Gamma_end=float(cfg["sqa_Gamma_end"]),
        )
        self._qaoa = QAOASolver(
            p_layers=int(cfg["qaoa_p_layers"]),
            n_optimizer_restarts=int(cfg["qaoa_restarts"]),
            n_shots=int(cfg["qaoa_shots"]),
            fallback_to_annealer=True,
            fallback_sqa_params={
                "n_trotter": int(cfg["sqa_trotter"]),
                "n_sweeps": int(cfg["sqa_sweeps"]),
                "n_reads": int(cfg["sqa_reads"]),
                "T_start": float(cfg["sqa_T_start"]),
                "T_end": float(cfg["sqa_T_end"]),
                "Gamma_start": float(cfg["sqa_Gamma_start"]),
                "Gamma_end": float(cfg["sqa_Gamma_end"]),
            },
        )
        self._hungarian = HungarianSolver(
            confidence_threshold=self.conf_thresh,
        )
        self._sa_classical = ClassicalSimulatedAnnealer(
            n_sweeps=int(cfg["sqa_sweeps"]),
            n_reads=int(cfg["sqa_reads"]),
            T_start=float(cfg["sqa_T_start"]),
            T_end=float(cfg["sqa_T_end"]),
        )

    @property
    def matcher_id(self) -> str:
        return f"qubo_{self.solver_name}"

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface (BaseMatcher contract)
    # ─────────────────────────────────────────────────────────────────────────

    def match(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float = 1.0,
        mask_src: Optional[np.ndarray] = None,
        mask_ref: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> MatchResult:
        """
        Full Hybrid Classical–Quantum correspondence pipeline.

        Level 1: SIFT detection → ANMS → description
        Level 2: QUBO build → quantum/classical solve
        Level 3: binary solution → MatchResult
        """
        t0 = time.time()
        try:
            return self._match_impl(src, ref, gsd_ratio, mask_src, mask_ref)
        except Exception as exc:
            log.error("QUBOMatcher.match failed: %s", exc, exc_info=True)
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason=str(exc),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Q0b: QUBO Consistency Filter (Hybrid Level 2 Filter)
    # ─────────────────────────────────────────────────────────────────────────

    def filter_matches(
        self,
        raw_matches: MatchResult,
        gsd_ratio: float = 1.0,
        cap: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[MatchResult, Dict[str, Any]]:
        """
        Q0b: Take raw candidate matches and filter them for geometric consistency
        and one-to-one uniqueness using QUBO Ising optimization.

        Parameters
        ----------
        raw_matches : MatchResult from any upstream matcher
        gsd_ratio   : source GSD / reference GSD
        cap         : max candidate variables (defaults to 16 for QAOA, 64 for SQA)
        seed        : random seed

        Returns
        -------
        (filtered_result, info_dict)
        """
        t0 = time.time()
        N = raw_matches.count
        if N == 0:
            return raw_matches, {"energy": 0.0, "n_vars": 0, "runtime_s": 0.0, "kept": 0}

        default_cap = 16 if self.solver_name == "qaoa" else 64
        actual_cap = cap or default_cap

        # Truncate to top candidates by confidence if needed
        src_xy = np.copy(raw_matches.src_xy)
        ref_xy = np.copy(raw_matches.ref_xy)
        conf = np.copy(raw_matches.confidence)
        scale = np.copy(raw_matches.scale) if raw_matches.scale is not None else None
        angle = np.copy(raw_matches.angle_deg) if raw_matches.angle_deg is not None else None

        if N > actual_cap:
            idx = np.argsort(-conf)[:actual_cap]
            src_xy = src_xy[idx]
            ref_xy = ref_xy[idx]
            conf = conf[idx]
            if scale is not None:
                scale = scale[idx]
            if angle is not None:
                angle = angle[idx]
            M = actual_cap
        else:
            M = N

        # Build QUBO Hamiltonian
        Q = np.zeros((M, M), dtype=np.float64)

        # Unary term: reward high confidence -> minimize -conf
        for i in range(M):
            Q[i, i] = -float(conf[i])

        # Pairwise term: enforce one-to-one constraint and relative distance consistency
        lambda_pen = float(self.lambda_penalty)
        pair_w = float(self.pairwise_weight)

        for i in range(M):
            for j in range(i + 1, M):
                # 1. Spatial uniqueness (one-to-one)
                d_src_ij = float(np.linalg.norm(src_xy[i] - src_xy[j]))
                d_ref_ij = float(np.linalg.norm(ref_xy[i] - ref_xy[j]))

                if d_src_ij < 2.0 or d_ref_ij < 2.0:
                    Q[i, j] += lambda_pen
                else:
                    # 2. Geometric distance ratio consistency
                    if d_src_ij > 5.0 and d_ref_ij > 5.0:
                        ratio = d_src_ij / (d_ref_ij * gsd_ratio + 1e-6)
                        if 0.70 <= ratio <= 1.40:
                            # Distance ratio preserved -> negative energy reward
                            Q[i, j] -= pair_w * float(np.exp(-((ratio - 1.0) ** 2) / 0.10))
                        else:
                            # Inconsistent scale distortion -> penalty
                            Q[i, j] += pair_w * 0.50

        # Solve via configured solver
        solve_res = self._solve(
            Q=Q,
            unary=conf.reshape(-1, 1),
            pairs=np.array([[i, i] for i in range(M)]),
            src_xy=src_xy,
            ref_xy=ref_xy,
        )

        solution = np.asarray(solve_res.get("solution", np.zeros(M, dtype=int)), dtype=int)
        kept_idx = np.where(solution == 1)[0]

        # If solver returned all zeros, keep top-ranked match as conservative fallback
        if len(kept_idx) == 0 and M > 0:
            kept_idx = np.array([int(np.argmax(conf))])

        filtered_src = src_xy[kept_idx]
        filtered_ref = ref_xy[kept_idx]
        filtered_conf = conf[kept_idx]
        filtered_scale = scale[kept_idx] if scale is not None else None
        filtered_angle = angle[kept_idx] if angle is not None else None

        runtime_s = time.time() - t0

        result = MatchResult(
            src_xy=filtered_src,
            ref_xy=filtered_ref,
            confidence=filtered_conf,
            scale=filtered_scale,
            angle_deg=filtered_angle,
            runtime_s=runtime_s,
            matcher_params={
                "matcher_id": f"qubo_{self.solver_name}_filter",
                "solver": self.solver_name,
                "n_vars": M,
                "n_kept": len(kept_idx),
                "energy": float(solve_res.get("energy", 0.0)),
                "runtime_s": runtime_s,
            },
        )

        info = {
            "solver": self.solver_name,
            "n_vars": M,
            "n_kept": len(kept_idx),
            "kept_fraction": float(len(kept_idx) / max(1, M)),
            "energy": float(solve_res.get("energy", 0.0)),
            "runtime_s": runtime_s,
            "solution": solution.tolist(),
        }

        return result, info

    # ─────────────────────────────────────────────────────────────────────────
    # Implementation
    # ─────────────────────────────────────────────────────────────────────────

    def _match_impl(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float,
        mask_src: Optional[np.ndarray],
        mask_ref: Optional[np.ndarray],
    ) -> MatchResult:
        t0 = time.time()

        # ── Level 1: Classical front-end ──────────────────────────────────────
        kp_src, kp_ref, des_src, des_ref = self._extract_keypoints(
            src, ref, mask_src, mask_ref
        )

        if len(kp_src) < 2 or len(kp_ref) < 2:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="insufficient_keypoints",
            )

        src_xy = np.array([[k.pt[0], k.pt[1]] for k in kp_src], dtype=np.float32)
        ref_xy = np.array([[k.pt[0], k.pt[1]] for k in kp_ref], dtype=np.float32)
        src_scales = np.array([k.size for k in kp_src], dtype=np.float32)
        ref_scales = np.array([k.size for k in kp_ref], dtype=np.float32)
        src_angles = np.array([k.angle for k in kp_src], dtype=np.float32)
        ref_angles = np.array([k.angle for k in kp_ref], dtype=np.float32)

        log.debug(
            "QUBOMatcher: N_src=%d, N_ref=%d keypoints", len(kp_src), len(kp_ref)
        )

        # ── Level 2: QUBO build ───────────────────────────────────────────────
        qubo_data = self.qubo_builder.build(
            src_xy=src_xy,
            ref_xy=ref_xy,
            src_desc=des_src,
            ref_desc=des_ref,
            src_scales=src_scales,
            ref_scales=ref_scales,
            src_angles=src_angles,
            ref_angles=ref_angles,
            gsd_ratio=gsd_ratio,
        )

        Q = qubo_data["Q"]
        pairs = qubo_data["pairs"]
        unary = qubo_data["unary"]
        M = qubo_data["n_vars"]

        log.debug("QUBO built: M=%d variables", M)

        if M == 0:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="no_candidate_pairs_above_threshold",
            )

        # ── Level 2: Quantum / classical solve ───────────────────────────────
        solve_result = self._solve(Q, unary, pairs, src_xy, ref_xy)

        # ── Level 3: Extract matches from solution ────────────────────────────
        solution = solve_result["solution"]  # (M,) binary
        matched = self._solution_to_matches(
            solution, pairs, src_xy, ref_xy, unary,
            src_scales, ref_scales, src_angles, ref_angles,
            gsd_ratio,
        )

        if matched["src_xy"].shape[0] == 0:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="no_matches_after_quantum_solve",
            )

        runtime = time.time() - t0
        log.debug(
            "QUBOMatcher(%s): %d matches, energy=%.4f, t=%.3fs",
            self.solver_name, len(matched["src_xy"]),
            solve_result.get("energy", 0.0), runtime,
        )

        return MatchResult(
            src_xy=matched["src_xy"],
            ref_xy=matched["ref_xy"],
            confidence=matched["confidence"],
            scale=matched["scale"],
            angle_deg=matched["angle_deg"],
            runtime_s=runtime,
            matcher_params={
                "matcher_id": self.matcher_id,
                "solver": self.solver_name,
                "qubo_energy": solve_result.get("energy", None),
                "qubo_n_vars": M,
                "qubo_solution": solution,
                "qubo_pairs": pairs,
                "n_src_kp": len(kp_src),
                "n_ref_kp": len(kp_ref),
                "n_matches": len(matched["src_xy"]),
                "gsd_ratio": gsd_ratio,
                "qubo_meta": qubo_data["meta"],
                "solver_meta": {
                    k: v for k, v in solve_result.items()
                    if k not in ("solution",)
                },
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Level 1: Keypoint extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_keypoints(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        mask_src: Optional[np.ndarray],
        mask_ref: Optional[np.ndarray],
    ):
        """Detect and describe keypoints using SIFT + ANMS SSC."""
        import cv2

        src_gray = self._to_gray(src)
        ref_gray = self._to_gray(ref)

        m_src = (mask_src > 0).astype(np.uint8) * 255 if mask_src is not None else None
        m_ref = (mask_ref > 0).astype(np.uint8) * 255 if mask_ref is not None else None

        sift = cv2.SIFT_create(nfeatures=0)

        kp_src_raw = sift.detect(src_gray, m_src)
        kp_ref_raw = sift.detect(ref_gray, m_ref)

        if not kp_src_raw or not kp_ref_raw:
            return [], [], None, None

        # ANMS SSC for spatial uniformity
        kp_src = anms_ssc(kp_src_raw, self.num_keypoints, src_gray.shape[:2])
        kp_ref = anms_ssc(kp_ref_raw, self.num_keypoints, ref_gray.shape[:2])

        # Describe
        kp_src, des_src = sift.compute(src_gray, kp_src)
        kp_ref, des_ref = sift.compute(ref_gray, kp_ref)

        return kp_src, kp_ref, des_src, des_ref

    # ─────────────────────────────────────────────────────────────────────────
    # Level 2: Solver dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _solve(
        self,
        Q: np.ndarray,
        unary: np.ndarray,
        pairs: np.ndarray,
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
    ) -> Dict[str, Any]:
        """Dispatch to the configured quantum/classical solver."""
        if self.solver_name == "sqa":
            return self._sqa.solve(Q)
        elif self.solver_name == "qaoa":
            return self._qaoa.solve(Q)
        elif self.solver_name in ("classical", "hungarian"):
            result = self._hungarian.solve(unary, pairs, src_xy, ref_xy, Q=Q)
            # Re-format to match solver contract
            return {
                "solution": result["solution"],
                "energy": result["energy"],
                "linear_cost_energy": result.get("linear_cost_energy"),
                "runtime_s": result["runtime_s"],
                "solver": result["solver"],
            }
        elif self.solver_name in ("classical_sa", "sa_classical", "simulated_annealing"):
            return self._sa_classical.solve(Q)
        else:
            raise ValueError(f"Unknown solver: {self.solver_name!r}")

    # ─────────────────────────────────────────────────────────────────────────
    # Level 3: Solution → MatchResult arrays
    # ─────────────────────────────────────────────────────────────────────────

    def _solution_to_matches(
        self,
        solution: np.ndarray,
        pairs: np.ndarray,
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
        unary: np.ndarray,
        src_scales: np.ndarray,
        ref_scales: np.ndarray,
        src_angles: np.ndarray,
        ref_angles: np.ndarray,
        gsd_ratio: float,
    ) -> Dict[str, np.ndarray]:
        """Convert binary QUBO solution to matched coordinate arrays."""
        selected = np.where(solution == 1)[0]

        if len(selected) == 0:
            return {
                "src_xy": np.empty((0, 2), dtype=np.float32),
                "ref_xy": np.empty((0, 2), dtype=np.float32),
                "confidence": np.empty(0, dtype=np.float32),
                "scale": np.empty(0, dtype=np.float32),
                "angle_deg": np.empty(0, dtype=np.float32),
            }

        src_list, ref_list, conf_list, scale_list, angle_list = [], [], [], [], []
        seen_src = set()
        seen_ref = set()

        # Sort selected by unary affinity descending so highest confidence matches are evaluated first
        selected_aff = [float(unary[int(pairs[k, 0]), int(pairs[k, 1])]) for k in selected]
        order = np.argsort(-np.array(selected_aff))
        selected_sorted = selected[order]

        for k in selected_sorted:
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            aff = float(unary[i, j])

            if aff < self.conf_thresh:
                continue

            # In v2: greedy one-to-one conflict resolution (keep higher-confidence match)
            if self.formulation_version == "v2":
                if i in seen_src or j in seen_ref:
                    continue
                seen_src.add(i)
                seen_ref.add(j)

            src_list.append(src_xy[i])
            ref_list.append(ref_xy[j])
            conf_list.append(aff)

            # Scale ratio
            s_src = float(src_scales[i]) if i < len(src_scales) else 1.0
            s_ref = float(ref_scales[j]) if j < len(ref_scales) else 1.0
            scale_list.append(s_src / (s_ref + 1e-9))

            # Angle difference
            a_src = float(src_angles[i]) if i < len(src_angles) else 0.0
            a_ref = float(ref_angles[j]) if j < len(ref_angles) else 0.0
            angle_list.append(a_src - a_ref)

        if not src_list:
            return {
                "src_xy": np.empty((0, 2), dtype=np.float32),
                "ref_xy": np.empty((0, 2), dtype=np.float32),
                "confidence": np.empty(0, dtype=np.float32),
                "scale": np.empty(0, dtype=np.float32),
                "angle_deg": np.empty(0, dtype=np.float32),
            }

        out_src = np.array(src_list, dtype=np.float32)
        out_ref = np.array(ref_list, dtype=np.float32)
        out_conf = np.array(conf_list, dtype=np.float32)
        out_scale = np.array(scale_list, dtype=np.float32)
        out_angle = np.array(angle_list, dtype=np.float32)

        # Scale-consistency filter (ARCHITECTURE.md §4)
        keep = self._scale_consistency_mask(
            out_scale,
            np.ones(len(out_scale)),
            gsd_ratio,
            self.max_log_scale_dev,
        )

        return {
            "src_xy": out_src[keep],
            "ref_xy": out_ref[keep],
            "confidence": out_conf[keep],
            "scale": out_scale[keep],
            "angle_deg": out_angle[keep],
        }

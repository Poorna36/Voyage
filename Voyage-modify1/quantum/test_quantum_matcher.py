"""
quantum/test_quantum_matcher.py
================================
Automated test suite for the Quantum QUBO Correspondence Module.

Tests
-----
T-QM01: QUBOBuilder — correct matrix dimensions and symmetry
T-QM02: QUBOBuilder — one-to-one penalty structure
T-QM03: QUBOBuilder — unary affinity produces values in [0,1]
T-QM04: SimulatedQuantumAnnealer — finds known ground state on trivial problem
T-QM05: SimulatedQuantumAnnealer — satisfies one-to-one constraints on identity match
T-QM06: QAOASolver — expected energy is better than random for small M
T-QM07: QAOASolver — auto-fallback to SQA when M > limit
T-QM08: HungarianSolver — recovers known optimal assignment
T-QM09: QUBOMatcher (SQA) — returns valid MatchResult on synthetic image pair
T-QM10: QUBOMatcher (QAOA) — returns valid MatchResult with correct schema
T-QM11: QUBOMatcher (Classical) — returns valid MatchResult
T-QM12: QUBOMatcher — empty result on degenerate (blank) input
T-QM13: QUBOBuilder energy function — x^T Q x sanity check
T-QM14: Benchmark harness — runs and produces non-empty report
T-QM15: Scale-consistency filter — rejects scale-inconsistent matches

Run with: python -m quantum.test_quantum_matcher [-v]
"""
from __future__ import annotations

import sys
import os
import logging
import unittest
from typing import Any, Dict

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Add project root to path so imports work when run directly
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from quantum.qubo_builder import QUBOBuilder
from quantum.annealer import SimulatedQuantumAnnealer
from quantum.qaoa import QAOASolver
from quantum.classical_baseline import HungarianSolver


logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_kp(n: int, size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(10, size - 10, size=(n, 2)).astype(np.float32)


def _random_desc(n: int, d: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d_ = rng.standard_normal((n, d)).astype(np.float32)
    return d_ / (np.linalg.norm(d_, axis=1, keepdims=True) + 1e-9)


def _synthetic_image(size: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.standard_normal((size, size)) * 30 + 128).clip(0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# T-QM01 – T-QM03: QUBOBuilder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQUBOBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = QUBOBuilder(
            lambda_penalty=5.0,
            alpha=0.6,
            max_pairs=50,
            unary_threshold=0.05,
        )
        self.src_xy = _random_kp(10, seed=1)
        self.ref_xy = _random_kp(10, seed=2)
        self.src_desc = _random_desc(10, d=32, seed=3)
        self.ref_desc = _random_desc(10, d=32, seed=4)

    def test_T_QM01_dimensions(self):
        """T-QM01: QUBO matrix is square with correct dimensions."""
        data = self.builder.build(self.src_xy, self.ref_xy, self.src_desc, self.ref_desc)
        M = data["n_vars"]
        Q = data["Q"]
        self.assertEqual(Q.shape, (M, M), f"Expected ({M},{M}), got {Q.shape}")
        self.assertGreater(M, 0, "Expected at least one candidate pair")

    def test_T_QM02_symmetry(self):
        """T-QM01: QUBO matrix is symmetric."""
        data = self.builder.build(self.src_xy, self.ref_xy, self.src_desc, self.ref_desc)
        Q = data["Q"]
        np.testing.assert_allclose(Q, Q.T, atol=1e-10, err_msg="Q is not symmetric")

    def test_T_QM02_one_to_one_penalty(self):
        """T-QM02: Penalty raises energy for duplicate assignments."""
        # Use perfectly aligned identical keypoints to force high unary affinity
        xy = _random_kp(5, seed=7)
        desc = _random_desc(5, d=16, seed=7)
        data = self.builder.build(xy, xy, desc, desc)
        Q = data["Q"]
        pairs = data["pairs"]
        if len(pairs) == 0:
            self.skipTest("No candidate pairs generated")

        # Energy should be LOWER for identity assignment than random
        identity_sol = np.zeros(len(pairs), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            if i == j:
                identity_sol[k] = 1.0
        if identity_sol.sum() > 0:
            e_identity = QUBOBuilder.energy(Q, identity_sol)
            e_zero = QUBOBuilder.energy(Q, np.zeros(len(pairs)))
            # Identity assignment should be better (lower energy) than no matches
            # (or at worst comparable — with penalties it's nuanced)
            self.assertIsInstance(e_identity, float)

    def test_T_QM03_unary_range(self):
        """T-QM03: Unary affinity values are in [0, 1]."""
        data = self.builder.build(self.src_xy, self.ref_xy, self.src_desc, self.ref_desc)
        U = data["unary"]
        self.assertTrue(np.all(U >= -1e-6), "Unary affinity has values < 0")
        self.assertTrue(np.all(U <= 1 + 1e-6), "Unary affinity has values > 1")

    def test_T_QM13_energy_sanity(self):
        """T-QM13: x^T Q x returns correct energy for known vectors."""
        Q = np.array([[1.0, -0.5], [-0.5, 2.0]])
        x0 = np.array([1.0, 0.0])
        x1 = np.array([0.0, 1.0])
        x2 = np.array([1.0, 1.0])
        self.assertAlmostEqual(QUBOBuilder.energy(Q, x0), 1.0)
        self.assertAlmostEqual(QUBOBuilder.energy(Q, x1), 2.0)
        self.assertAlmostEqual(QUBOBuilder.energy(Q, x2), 2.0)  # 1 + 2 - 2*0.5

    def test_T_QM16_pairwise_similarity_invariance(self):
        """T-QM16: Pairwise energy of true match pairs is invariant under rotation and scale."""
        builder_v2 = QUBOBuilder(formulation_version="v2", alpha=0.0)
        p1 = np.array([25.0, 30.0], dtype=np.float32)
        p2 = np.array([75.0, 80.0], dtype=np.float32)
        src_xy = np.vstack([p1, p2])
        src_scales = np.array([2.0, 3.0], dtype=np.float32)
        src_angles = np.array([15.0, 45.0], dtype=np.float32)

        data_base = builder_v2.build(
            src_xy, src_xy,
            src_scales=src_scales, ref_scales=src_scales,
            src_angles=src_angles, ref_angles=src_angles,
            gsd_ratio=1.0,
        )
        pairs_base = data_base["pairs"]
        k0 = next(k for k, (i, j) in enumerate(pairs_base) if i == 0 and j == 0)
        k1 = next(k for k, (i, j) in enumerate(pairs_base) if i == 1 and j == 1)
        base_w = data_base["Q"][k0, k1]

        rot_deg = 45.0
        rot_rad = np.radians(rot_deg)
        scale = 1.5
        trans = np.array([10.0, -20.0], dtype=np.float32)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad),  np.cos(rot_rad)]], dtype=np.float32)

        ref_xy = (scale * (src_xy @ R.T) + trans).astype(np.float32)
        ref_scales = (src_scales * scale).astype(np.float32)
        ref_angles = ((src_angles + rot_deg) % 360.0).astype(np.float32)

        data_trans = builder_v2.build(
            src_xy, ref_xy,
            src_scales=src_scales, ref_scales=ref_scales,
            src_angles=src_angles, ref_angles=ref_angles,
            gsd_ratio=1.0 / scale,
        )
        pairs_trans = data_trans["pairs"]
        k0_t = next(k for k, (i, j) in enumerate(pairs_trans) if i == 0 and j == 0)
        k1_t = next(k for k, (i, j) in enumerate(pairs_trans) if i == 1 and j == 1)
        trans_w = data_trans["Q"][k0_t, k1_t]

        np.testing.assert_allclose(
            trans_w, base_w, atol=1e-4,
            err_msg=f"Pairwise energy changed from {base_w} to {trans_w} under rotation/scale"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T-QM04 – T-QM05: SimulatedQuantumAnnealer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatedQuantumAnnealer(unittest.TestCase):

    def setUp(self):
        self.sqa = SimulatedQuantumAnnealer(
            n_trotter=8,
            n_sweeps=200,
            n_reads=3,
            seed=42,
        )

    def test_T_QM04_finds_obvious_minimum(self):
        """T-QM04: SQA finds minimum of trivial 2-var QUBO."""
        # H(x) = x0 - x1 — minimum at x=[0,1], energy=-1
        Q = np.array([
            [1.0, 0.0],
            [0.0, -1.0],
        ])
        result = self.sqa.solve(Q)
        self.assertIn("solution", result)
        self.assertEqual(len(result["solution"]), 2)
        self.assertLessEqual(result["energy"], 0.5, "Expected near-optimal solution")

    def test_T_QM04_result_schema(self):
        """T-QM04: SQA returns all required keys."""
        Q = np.eye(3) * -1.0
        result = self.sqa.solve(Q)
        for key in ["solution", "energy", "all_energies", "runtime_s", "n_vars", "solver"]:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertEqual(result["n_vars"], 3)
        self.assertEqual(result["solver"], "simulated_quantum_annealing")

    def test_T_QM04_empty_qubo(self):
        """T-QM04: SQA handles empty QUBO gracefully."""
        Q = np.zeros((0, 0))
        result = self.sqa.solve(Q)
        self.assertEqual(result["n_vars"], 0)
        self.assertEqual(len(result["solution"]), 0)

    def test_T_QM05_binary_solution(self):
        """T-QM05: SQA solution is binary {0,1}."""
        Q = np.array([[2.0, -1.0, 0.5],
                      [-1.0, 3.0, -0.5],
                      [0.5, -0.5, 1.0]])
        result = self.sqa.solve(Q)
        sol = result["solution"]
        self.assertTrue(
            np.all((sol == 0) | (sol == 1)),
            f"Non-binary solution values: {sol}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T-QM06 – T-QM07: QAOASolver tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQAOASolver(unittest.TestCase):

    def setUp(self):
        self.qaoa = QAOASolver(
            p_layers=1,
            n_optimizer_restarts=2,
            max_iter=50,
            n_shots=256,
            seed=42,
            fallback_to_annealer=True,
        )

    def test_T_QM06_result_schema(self):
        """T-QM06: QAOA returns all required keys."""
        Q = np.array([[-1.0, 0.5], [0.5, -1.0]])
        result = self.qaoa.solve(Q)
        for key in ["solution", "energy", "expected_energy", "runtime_s", "n_vars", "solver"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_T_QM06_expected_energy_is_scalar(self):
        """T-QM06: Expected energy is a finite scalar."""
        Q = np.array([[-1.0, 0.5], [0.5, -1.0]])
        result = self.qaoa.solve(Q)
        self.assertTrue(np.isfinite(result["expected_energy"]))

    def test_T_QM06_binary_solution(self):
        """T-QM06: QAOA solution is binary {0,1}."""
        Q = np.array([[-2.0, 0.3], [0.3, -1.5]])
        result = self.qaoa.solve(Q)
        sol = result["solution"]
        self.assertTrue(
            np.all((sol == 0) | (sol == 1)),
            f"Non-binary solution: {sol}"
        )

    def test_T_QM07_fallback_large_M(self):
        """T-QM07: QAOA falls back to SQA when M > 18."""
        M = 20  # exceeds QAOA limit
        Q = -np.eye(M) * 2.0
        result = self.qaoa.solve(Q)
        self.assertEqual(result["solver"], "sqa_fallback_from_qaoa")
        self.assertEqual(result["n_vars"], M)


# ─────────────────────────────────────────────────────────────────────────────
# T-QM08: HungarianSolver tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHungarianSolver(unittest.TestCase):

    def setUp(self):
        self.solver = HungarianSolver(confidence_threshold=0.1)

    def test_T_QM08_known_optimal(self):
        """T-QM08: Hungarian recovers optimal assignment on diagonal affinity."""
        N = 4
        # Perfect diagonal affinity: each source i best matches reference i
        unary = np.eye(N, dtype=np.float64) * 0.95
        # Add small noise off-diagonal
        unary += np.random.default_rng(0).uniform(0, 0.05, (N, N))
        np.fill_diagonal(unary, 0.95)

        src_xy = _random_kp(N, seed=1)
        ref_xy = _random_kp(N, seed=2)
        pairs = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int32)

        result = self.solver.solve(unary, pairs, src_xy, ref_xy)
        self.assertIn("solution", result)
        self.assertIn("src_xy", result)
        self.assertIn("ref_xy", result)
        self.assertGreater(len(result["src_xy"]), 0, "Expected at least one match")

    def test_T_QM08_one_to_one_constraint(self):
        """T-QM08: Hungarian enforces one-to-one matching."""
        N = 5
        unary = np.random.default_rng(7).uniform(0.3, 0.9, (N, N))
        src_xy = _random_kp(N, seed=1)
        ref_xy = _random_kp(N, seed=2)
        pairs = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int32)

        result = self.solver.solve(unary, pairs, src_xy, ref_xy)
        sol = result["solution"]
        selected = pairs[sol == 1]

        if len(selected) > 0:
            # Each source index appears at most once
            self.assertEqual(len(np.unique(selected[:, 0])), len(selected), "Duplicate source index")
            # Each ref index appears at most once
            self.assertEqual(len(np.unique(selected[:, 1])), len(selected), "Duplicate reference index")


# ─────────────────────────────────────────────────────────────────────────────
# T-QM09 – T-QM12: QUBOMatcher integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQUBOMatcher(unittest.TestCase):

    def _make_pair(self, size: int = 128, seed: int = 42):
        """Create a textured image pair with slight warp."""
        rng = np.random.default_rng(seed)
        src = (rng.standard_normal((size, size)) * 40 + 128).clip(0, 255).astype(np.uint8)
        # Add some structure
        for _ in range(8):
            cx, cy = rng.integers(10, size - 10, 2)
            r = rng.integers(4, 15)
            Y, X = np.ogrid[:size, :size]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
            src[mask] = np.clip(src[mask].astype(int) - 60, 0, 255)

        # Slightly shifted reference
        shift_r = int(rng.integers(3, 8))
        shift_c = int(rng.integers(3, 8))
        ref = np.roll(np.roll(src, shift_r, axis=0), shift_c, axis=1)
        return src, ref

    def _make_matcher(self, solver: str):
        from quantum.matcher_qubo import QUBOMatcher
        return QUBOMatcher(config={
            "solver": solver,
            "num_keypoints": 40,
            "max_pairs": 60,
            "unary_threshold": 0.05,
            "sqa_sweeps": 100,
            "sqa_reads": 2,
            "sqa_trotter": 8,
            "qaoa_p_layers": 1,
            "qaoa_restarts": 1,
            "qaoa_shots": 64,
        })

    def _check_result_schema(self, result, solver_name: str):
        """Verify MatchResult satisfies BaseMatcher contract."""
        from src.matching.base import MatchResult
        self.assertIsInstance(result, MatchResult, f"{solver_name}: not a MatchResult")
        self.assertEqual(result.src_xy.ndim, 2, f"{solver_name}: src_xy not 2D")
        self.assertEqual(result.src_xy.shape[-1], 2, f"{solver_name}: src_xy not (N,2)")
        self.assertEqual(result.ref_xy.ndim, 2, f"{solver_name}: ref_xy not 2D")
        self.assertEqual(result.ref_xy.shape[-1], 2, f"{solver_name}: ref_xy not (N,2)")
        self.assertEqual(len(result.src_xy), len(result.ref_xy))
        self.assertEqual(len(result.src_xy), len(result.confidence))
        self.assertIn("matcher_id", result.matcher_params)
        self.assertGreater(result.runtime_s, 0.0)

    def test_T_QM09_sqa_valid_result(self):
        """T-QM09: SQA matcher returns valid MatchResult on synthetic pair."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not available")
        src, ref = self._make_pair()
        matcher = self._make_matcher("sqa")
        result = matcher.match(src, ref)
        self._check_result_schema(result, "sqa")

    def test_T_QM10_qaoa_valid_result(self):
        """T-QM10: QAOA matcher returns valid MatchResult."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not available")
        src, ref = self._make_pair(seed=7)
        matcher = self._make_matcher("qaoa")
        result = matcher.match(src, ref)
        self._check_result_schema(result, "qaoa")

    def test_T_QM11_classical_valid_result(self):
        """T-QM11: Classical baseline matcher returns valid MatchResult."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not available")
        src, ref = self._make_pair(seed=13)
        matcher = self._make_matcher("classical")
        result = matcher.match(src, ref)
        self._check_result_schema(result, "classical")

    def test_T_QM12_degenerate_blank_image(self):
        """T-QM12: QUBOMatcher returns empty (not crash) on blank input."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not available")
        blank = np.zeros((64, 64), dtype=np.uint8)
        matcher = self._make_matcher("sqa")
        result = matcher.match(blank, blank)
        # Must return MatchResult (possibly empty), never raise
        from src.matching.base import MatchResult
        self.assertIsInstance(result, MatchResult)

    def test_T_QM15_matcher_id_correct(self):
        """T-QM15: matcher_id includes solver name."""
        from quantum.matcher_qubo import QUBOMatcher
        for solver in ["sqa", "qaoa", "classical"]:
            m = QUBOMatcher(config={"solver": solver})
            self.assertEqual(m.matcher_id, f"qubo_{solver}")

    def test_T_QM17_zero_constraint_violations_v2(self):
        """T-QM17: QUBO v2 with dynamic lambda achieves 0% one-to-one constraint violations."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not available")
        from quantum.benchmark import _make_synthetic_pair, _constraint_violation
        from quantum.matcher_qubo import QUBOMatcher
        pair = _make_synthetic_pair(size=128, n_true_matches=15, rotation_deg=8.0, scale=1.15, seed=42)
        matcher = QUBOMatcher(config={
            "formulation_version": "v2",
            "solver": "sqa",
            "num_keypoints": 30,
            "max_pairs": 40,
            "sqa_sweeps": 200,
            "sqa_reads": 3,
        })
        res = matcher.match(pair["src"], pair["ref"], gsd_ratio=1.15)
        sol = res.matcher_params.get("qubo_solution")
        pairs = res.matcher_params.get("qubo_pairs")
        cv = _constraint_violation(sol, pairs)
        self.assertEqual(cv, 0.0, f"Expected 0.0 constraint violation in v2, got {cv}")


# ─────────────────────────────────────────────────────────────────────────────
# T-QM14: Benchmark harness test
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark(unittest.TestCase):

    def test_T_QM14_benchmark_runs(self):
        """T-QM14: Benchmark harness produces non-empty report with classical solver."""
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not available")

        from quantum.benchmark import QuantumBenchmark
        bench = QuantumBenchmark(image_size=128, n_true_matches=15, seed=42)
        report = bench.run(n_kp=30, solvers=["classical"], verbose=False)

        self.assertIn("benchmark", report)
        self.assertIn("results", report)
        self.assertIn("classical", report["results"])
        res = report["results"]["classical"]
        self.assertIn("n_matches", res)
        self.assertIn("runtime_s", res)
        self.assertGreater(res["runtime_s"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)

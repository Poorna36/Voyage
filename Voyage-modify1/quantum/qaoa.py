"""
quantum/qaoa.py
===============
QAOA (Quantum Approximate Optimization Algorithm) solver for QUBO problems.

This is a statevector-level simulation of QAOA on a gate-based quantum
computer. It runs entirely on NumPy — no Qiskit or Cirq required.

QAOA Overview
-------------
QAOA is a variational quantum algorithm for combinatorial optimisation.
Given a QUBO with M variables, the circuit acts on an M-qubit register:

  1. Initialise: |+⟩^⊗M  (equal superposition)

  2. Repeat p layers:
     a. Apply problem unitary:  U_P(γ) = exp(-i γ H_P)
        where H_P is the problem Hamiltonian (diagonal in Z-basis)
     b. Apply mixer unitary:    U_M(β) = exp(-i β H_M)
        where H_M = -Σ_i X_i  (transverse-field mixer)

  3. Measure in computational basis → sample solution.

  4. Optimise (γ, β) to minimise expected energy ⟨H_P⟩.

Complexity note: Full statevector requires 2^M amplitudes — feasible only
for M ≤ 18 on a workstation. For larger problems we use the annealer.

References
----------
- Farhi et al. (2014), "A Quantum Approximate Optimization Algorithm",
  arXiv:1411.4028.
- quantum/QUANTUM_COMPUTING_RESEARCH.txt §4.3 (QAOA formulation)
- Blekos et al. (2024), "A review on Quantum Approximate Optimization
  Algorithm and its variants", Physics Reports.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Maximum number of QUBO variables for statevector QAOA (memory guard)
_MAX_QAOA_VARS = 18


class QAOASolver:
    """
    Statevector QAOA solver for QUBO.

    min_{x ∈ {0,1}^M}  H(x) = x^T Q x

    Parameters
    ----------
    p_layers : int
        QAOA circuit depth p. Higher p → better approximation ratio.
        p=1 gives ~0.5 approximation; p≥3 is typically sufficient.
    n_optimizer_restarts : int
        Number of random restarts for the classical parameter optimiser.
    max_iter : int
        Maximum iterations per optimiser restart (Nelder–Mead).
    n_shots : int
        Number of measurement samples to draw from final state.
    seed : int
        Random seed.
    fallback_to_annealer : bool
        If True, automatically fall back to SQA when M > _MAX_QAOA_VARS.
    """

    def __init__(
        self,
        p_layers: int = 2,
        n_optimizer_restarts: int = 5,
        max_iter: int = 200,
        n_shots: int = 1024,
        seed: int = 42,
        fallback_to_annealer: bool = True,
        fallback_sqa_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.p = int(p_layers)
        self.n_restarts = int(n_optimizer_restarts)
        self.max_iter = int(max_iter)
        self.n_shots = int(n_shots)
        self.seed = int(seed)
        self.fallback_to_annealer = fallback_to_annealer
        self.fallback_sqa_params = fallback_sqa_params

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self, Q: np.ndarray) -> Dict[str, Any]:
        """
        Find approximate ground state of QUBO matrix Q via QAOA.

        Parameters
        ----------
        Q : (M, M) float64 QUBO matrix

        Returns
        -------
        dict with keys:
            'solution'      : (M,) int array in {0, 1}
            'energy'        : float
            'expected_energy': float — variational ⟨H_P⟩ at optimal (γ, β)
            'optimal_gamma' : list of p gamma values
            'optimal_beta'  : list of p beta values
            'n_shots'       : int
            'runtime_s'     : float
            'n_vars'        : int M
            'solver'        : str
        """
        t0 = time.time()
        Q = np.asarray(Q, dtype=np.float64)
        M = Q.shape[0]

        if M == 0:
            return self._empty_result(M, t0)

        # Auto-fallback for large M
        if M > _MAX_QAOA_VARS:
            log.info(
                "QAOA: M=%d > %d, falling back to SimulatedQuantumAnnealing",
                M, _MAX_QAOA_VARS,
            )
            if self.fallback_to_annealer:
                from .annealer import SimulatedQuantumAnnealer
                sqa_kwargs = dict(self.fallback_sqa_params or {})
                sqa_kwargs.setdefault("seed", self.seed)
                sqa = SimulatedQuantumAnnealer(**sqa_kwargs)
                result = sqa.solve(Q)
                result["solver"] = "sqa_fallback_from_qaoa"
                return result
            else:
                return self._empty_result(M, t0, reason="M_exceeds_qaoa_limit")

        Q_sym = (Q + Q.T) / 2.0
        rng = np.random.default_rng(self.seed)

        # Normalise Q for well-conditioned variational optimization (energies kept ~ O(1-10))
        scale = float(np.abs(Q_sym).max())
        if scale < 1e-12:
            scale = 1.0
        Q_norm = Q_sym / scale

        # Diagonal of H_P (problem Hamiltonian eigenvalues in Z-basis)
        h_norm = self._qubo_to_ising_diagonal(Q_norm, M)
        h_diag = h_norm * scale

        # ── Variational optimisation of (γ, β) ───────────────────────────────
        best_exp_e_norm = np.inf
        best_gamma: List[float] = []
        best_beta: List[float] = []

        for restart in range(self.n_restarts):
            gamma0 = rng.uniform(0, 2 * np.pi, self.p).tolist()
            beta0 = rng.uniform(0, np.pi, self.p).tolist()
            params0 = gamma0 + beta0

            result_opt = self._nelder_mead(
                lambda params: self._expected_energy(params, h_norm, M),
                params0,
                max_iter=self.max_iter,
                rng=rng,
            )
            exp_e_norm = float(result_opt["fun"])
            log.debug("  QAOA restart %d/%d: ⟨H_norm⟩=%.4f", restart + 1, self.n_restarts, exp_e_norm)

            if exp_e_norm < best_exp_e_norm:
                best_exp_e_norm = exp_e_norm
                best_gamma = result_opt["x"][: self.p]
                best_beta = result_opt["x"][self.p :]

        # ── Sample from optimal state ─────────────────────────────────────────
        state = self._build_state(best_gamma, best_beta, h_norm, M)
        probs = np.abs(state) ** 2
        p_sum = probs.sum()
        probs = probs / p_sum if p_sum > 0 else np.ones_like(probs) / len(probs)
        samples = rng.choice(2 ** M, size=self.n_shots, p=probs)

        # Metrics: expected energy in unnormalized scale, approx ratio, ground state prob
        best_exp_e = float(np.dot(probs, h_diag))
        ground_energy = float(h_diag.min())
        approx_ratio = float(best_exp_e / ground_energy) if abs(ground_energy) > 1e-12 else 1.0
        is_ground = np.isclose(h_diag, ground_energy, atol=1e-6)
        ground_prob = float(probs[is_ground].sum())

        # Find sample with lowest QUBO energy
        best_x: Optional[np.ndarray] = None
        best_e = np.inf
        for idx in samples:
            e = float(h_diag[idx])
            if e < best_e:
                best_e = e
                best_x = self._index_to_binary(idx, M)

        if best_x is None:
            best_x = np.zeros(M, dtype=np.int32)

        runtime = time.time() - t0
        log.debug("QAOA done: energy=%.4f, M=%d, t=%.3fs", best_e, M, runtime)

        return {
            "solution": best_x.astype(np.int32),
            "energy": best_e,
            "expected_energy": best_exp_e,
            "ground_state_energy": ground_energy,
            "approximation_ratio": approx_ratio,
            "ground_state_probability": ground_prob,
            "optimal_gamma": list(best_gamma),
            "optimal_beta": list(best_beta),
            "n_shots": self.n_shots,
            "runtime_s": runtime,
            "n_vars": M,
            "solver": "qaoa_statevector",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # QAOA circuit (statevector)
    # ─────────────────────────────────────────────────────────────────────────

    def _qubo_to_ising_diagonal(self, Q: np.ndarray, M: int) -> np.ndarray:
        """
        Convert QUBO H(x) = x^T Q x to the diagonal of the Ising Hamiltonian
        H_P in the 2^M computational basis (vectorised).
        """
        idx = np.arange(2 ** M)
        X = ((idx[:, None] >> np.arange(M)) & 1).astype(np.float64)
        return ((X @ Q) * X).sum(axis=1)

    def _build_state(
        self,
        gamma: List[float],
        beta: List[float],
        h_diag: np.ndarray,
        M: int,
    ) -> np.ndarray:
        """
        Build the QAOA statevector for parameters (γ, β).

        Starts from |+⟩^⊗M = (1/√N) * [1, 1, ..., 1],
        then applies p layers of (U_P, U_M).
        """
        N = 2 ** M
        state = np.ones(N, dtype=np.complex128) / np.sqrt(N)

        for layer in range(self.p):
            g = gamma[layer]
            b = beta[layer]

            # Problem unitary: U_P(γ) = diag(exp(-i γ h_diag))
            state = state * np.exp(-1j * g * h_diag)

            # Mixer unitary: U_M(β) = ⊗_i exp(-i β X_i) (vectorised)
            state = self._apply_mixer(state, b, M)

        return state

    def _apply_mixer(
        self, state: np.ndarray, beta: float, M: int
    ) -> np.ndarray:
        """
        Apply the mixer unitary U_M(β) = ⊗_i exp(-i β X_i) to the statevector (vectorised).
        """
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = state.reshape((2,) * M)
        for ax in range(M):
            a, b = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * a + s * b, s * a + c * b], axis=ax)
        return psi.reshape(-1)

    def _expected_energy(
        self, params: List[float], h_diag: np.ndarray, M: int
    ) -> float:
        """Compute ⟨ψ(γ,β)| H_P |ψ(γ,β)⟩ for variational optimisation."""
        gamma = params[: self.p]
        beta = params[self.p :]
        state = self._build_state(gamma, beta, h_diag, M)
        probs = np.abs(state) ** 2
        return float(np.dot(probs, h_diag))

    # ─────────────────────────────────────────────────────────────────────────
    # Minimal Nelder-Mead optimiser (no scipy dependency)
    # ─────────────────────────────────────────────────────────────────────────

    def _nelder_mead(
        self,
        f,
        x0: List[float],
        max_iter: int,
        rng: np.random.Generator,
        alpha: float = 1.0,
        gamma_nm: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Nelder-Mead simplex minimisation.
        Returns dict with 'x' (best params) and 'fun' (best function value).
        """
        n = len(x0)
        # Initialise simplex
        simplex = [np.array(x0)]
        for i in range(n):
            xi = np.array(x0)
            xi[i] += 0.2 + rng.uniform(-0.05, 0.05)
            simplex.append(xi)

        fvals = np.array([f(x) for x in simplex])

        for iteration in range(max_iter):
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[o] for o in order]
            fvals = fvals[order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Expand
                xe = centroid + gamma_nm * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contract
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])

            # Convergence check
            if np.max(np.abs(fvals - fvals[0])) < tol:
                break

        return {"x": simplex[0].tolist(), "fun": float(fvals[0])}

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _index_to_binary(idx: int, M: int) -> np.ndarray:
        """Convert integer index to binary vector of length M (LSB first)."""
        return np.array([(idx >> i) & 1 for i in range(M)], dtype=np.float64)

    @staticmethod
    def _empty_result(M: int, t0: float, reason: str = "") -> Dict[str, Any]:
        return {
            "solution": np.zeros(M, dtype=np.int32),
            "energy": 0.0,
            "expected_energy": 0.0,
            "optimal_gamma": [],
            "optimal_beta": [],
            "n_shots": 0,
            "runtime_s": time.time() - t0,
            "n_vars": M,
            "solver": "qaoa_statevector",
            "failure_reason": reason,
        }

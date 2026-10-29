"""
quantum/annealer.py
===================
Self-contained Simulated Quantum Annealing (SQA) solver for QUBO problems.

This module implements Simulated Quantum Annealing using the Path-Integral
Monte Carlo (PIMC) formulation of quantum annealing. It does NOT require
D-Wave hardware or any quantum SDK — it runs entirely on CPU/NumPy.

Physics Background
------------------
Classical Simulated Annealing (SA) uses thermal fluctuations to escape local
minima by accepting uphill moves with probability exp(-ΔE / T).

Quantum Annealing (QA) adds a transverse-field term Γ(t) * Σ_i σ^x_i to the
Hamiltonian, which enables quantum tunnelling through energy barriers.

In PIMC (our simulation):
  - The quantum system is represented as P "Trotter slices" (replicas).
  - Each slice is a classical binary spin configuration.
  - Slices are coupled by an effective tunnel coupling J_t = -T/2 * ln(tanh(Γ/(P*T))).
  - The energy of a Trotter configuration includes both the problem Hamiltonian
    on each slice AND ferromagnetic coupling between adjacent slices.

Schedule:
  - Temperature T: high → low  (thermal annealing)
  - Transverse field Γ: high → low  (quantum field annealing)

References
----------
- Finnila et al. (1994), "Quantum annealing: A new method for minimizing
  multidimensional functions", Chem. Phys. Lett. 219, 343–348.
- Kadowaki & Nishimori (1998), Phys. Rev. E 58, 5355.
- quantum/QUANTUM_COMPUTING_RESEARCH.txt §4.3, §5
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class SimulatedQuantumAnnealer:
    """
    Path-Integral Monte Carlo Simulated Quantum Annealing for QUBO.

    Solves:  min_{x ∈ {0,1}^M}  H(x) = x^T Q x

    Parameters
    ----------
    n_trotter : int
        Number of Trotter slices (replicas). More slices = better quantum
        tunnelling simulation, higher cost. Typical: 8–32.
    n_sweeps : int
        Number of Monte Carlo sweeps per Trotter update step.
    T_start, T_end : float
        Initial and final temperatures for the thermal schedule.
    Gamma_start, Gamma_end : float
        Initial and final transverse-field strengths.
    n_reads : int
        Number of independent annealing runs (best solution selected).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_trotter: int = 16,
        n_sweeps: int = 1000,
        T_start: float = 5.0,
        T_end: float = 0.01,
        Gamma_start: float = 3.0,
        Gamma_end: float = 0.001,
        n_reads: int = 5,
        seed: int = 42,
    ) -> None:
        self.n_trotter = int(n_trotter)
        self.n_sweeps = int(n_sweeps)
        self.T_start = float(T_start)
        self.T_end = float(T_end)
        self.Gamma_start = float(Gamma_start)
        self.Gamma_end = float(Gamma_end)
        self.n_reads = int(n_reads)
        self.seed = int(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self, Q: np.ndarray) -> Dict[str, Any]:
        """
        Find the approximate ground state of QUBO matrix Q.

        Parameters
        ----------
        Q : (M, M) float64 — QUBO matrix (symmetric or upper triangular)

        Returns
        -------
        dict with keys:
            'solution'      : (M,) int array in {0, 1}
            'energy'        : float — H(solution) = solution^T Q solution
            'all_energies'  : list of energies per read
            'runtime_s'     : float
            'n_vars'        : int M
            'solver'        : 'simulated_quantum_annealing'
        """
        t0 = time.time()
        Q = np.asarray(Q, dtype=np.float64)
        M = Q.shape[0]

        if M == 0:
            return self._empty_result(0, t0)

        # Symmetrise
        Q_sym = (Q + Q.T) / 2.0

        rng = np.random.default_rng(self.seed)
        best_x: Optional[np.ndarray] = None
        best_e = np.inf
        all_energies = []

        for read in range(self.n_reads):
            x, e = self._anneal(Q_sym, rng)
            all_energies.append(float(e))
            if e < best_e:
                best_e = e
                best_x = x.copy()
            log.debug("  SQA read %d/%d: energy=%.4f", read + 1, self.n_reads, e)

        runtime = time.time() - t0
        log.debug("SQA done: best_energy=%.4f, M=%d, t=%.3fs", best_e, M, runtime)

        return {
            "solution": best_x.astype(np.int32),
            "energy": best_e,
            "all_energies": all_energies,
            "runtime_s": runtime,
            "n_vars": M,
            "solver": "simulated_quantum_annealing",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Core annealing loop (PIMC)
    # ─────────────────────────────────────────────────────────────────────────

    def _anneal(
        self, Q: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float]:
        """
        One annealing run using Path-Integral Monte Carlo.

        Returns (best_x, best_energy) for this run.
        """
        M = Q.shape[0]
        P = self.n_trotter

        # Initialise Trotter slices: (P, M) random binary spins
        spins = rng.integers(0, 2, size=(P, M), dtype=np.int32)

        # Precompute: diagonal of Q (h_i) and upper triangle (J_ij)
        # H(x) = x^T Q x = Σ_i Q_ii x_i + 2 Σ_{i<j} Q_ij x_i x_j
        h = np.diag(Q).copy()           # (M,) — linear biases
        J = np.triu(Q, k=1).copy()      # (M, M) — quadratic couplings (i<j)

        best_x = spins[0].copy()
        best_e = self._energy_classical(spins[0], Q)

        # Annealing schedule: log-space temperature, linear Gamma
        T_schedule = np.geomspace(self.T_start, self.T_end, self.n_sweeps)
        G_schedule = np.linspace(self.Gamma_start, self.Gamma_end, self.n_sweeps)

        for step in range(self.n_sweeps):
            T = T_schedule[step]
            Gamma = G_schedule[step]

            # Effective tunnel coupling between adjacent Trotter slices
            # J_t = -T/2 * ln(tanh(Γ/(P*T)))  [Suzuki-Trotter mapping]
            arg = Gamma / (P * T + 1e-12)
            arg = np.clip(arg, 1e-12, 50.0)
            J_tunnel = -T / 2.0 * np.log(np.tanh(arg) + 1e-12)

            # Single spin flip proposal for each spin in each Trotter slice
            for s in range(P):
                spin_s = spins[s]
                # Neighbour slices (periodic boundary)
                prev_s = spins[(s - 1) % P]
                next_s = spins[(s + 1) % P]

                # Compute ΔE for flipping each spin i
                for i in rng.permutation(M):
                    x_i = spin_s[i]
                    x_i_new = 1 - x_i

                    # Classical energy change in this slice
                    # ΔE_classical = Q_ii * (x_new - x_old) + 2 * Σ_{j≠i} Q_ij * x_j * (x_new - x_old)
                    delta = x_i_new - x_i  # ±1
                    dE_classical = (h[i] + 2.0 * (Q[i, :] @ spin_s - Q[i, i] * x_i)) * delta

                    # Tunnel coupling with adjacent Trotter slices
                    # ΔE_tunnel = J_t * (x_new - x_old) * (-(prev + next))
                    dE_tunnel = -J_tunnel * delta * (prev_s[i] + next_s[i] - 2 * x_i * 2 + 2)
                    # Simplified: coupling prefers aligned neighbours
                    dE_tunnel = J_tunnel * delta * (1 - 2 * prev_s[i]) + J_tunnel * delta * (1 - 2 * next_s[i])

                    dE = dE_classical + dE_tunnel

                    # Metropolis acceptance
                    if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                        spin_s[i] = x_i_new

                # Track best energy (use replica 0 as representative)
                if s == 0:
                    e_now = self._energy_classical(spin_s, Q)
                    if e_now < best_e:
                        best_e = e_now
                        best_x = spin_s.copy()

        return best_x, best_e

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _energy_classical(x: np.ndarray, Q: np.ndarray) -> float:
        """Evaluate QUBO energy H = x^T Q x."""
        x = np.asarray(x, dtype=np.float64)
        return float(x @ Q @ x)

    @staticmethod
    def _empty_result(M: int, t0: float) -> Dict[str, Any]:
        return {
            "solution": np.zeros(M, dtype=np.int32),
            "energy": 0.0,
            "all_energies": [],
            "runtime_s": time.time() - t0,
            "n_vars": M,
            "solver": "simulated_quantum_annealing",
        }

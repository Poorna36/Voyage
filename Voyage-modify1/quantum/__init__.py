"""
quantum/__init__.py
===================
Simulated and Quantum-Inspired Correspondence Module for SIH 2026 PS-26166.

Implements correspondence matching via QUBO formulation, solved via:
  - Simulated Quantum Annealing (PIMC on CPU)
  - QAOA statevector simulation (NumPy, M <= 18)
  - Classical Hungarian and Simulated Annealing baselines

Pipeline Integration: Layer 2 (L2) — Pluggable matcher (M4)
  matcher_id = 'qubo_qa'   (Quantum Annealing)
  matcher_id = 'qubo_qaoa' (QAOA gate-level)

References
----------
- quantum/QUANTUM_COMPUTING_RESEARCH.txt §4 (QUBO formulation)
- src/matching/base.py (BaseMatcher contract)
- docs/ARCHITECTURE.md §4
"""

from .qubo_builder import QUBOBuilder
from .annealer import SimulatedQuantumAnnealer
from .qaoa import QAOASolver
from .matcher_qubo import QUBOMatcher

__all__ = [
    "QUBOBuilder",
    "SimulatedQuantumAnnealer",
    "QAOASolver",
    "QUBOMatcher",
]

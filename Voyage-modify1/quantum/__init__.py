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
from .backends import QuantumBackend, get_backend, list_backends
from .selection_qaoa import select_qaoa
from .kernel_classifier import QuantumKernelClassifier
from .grover_consensus import GroverConsensusSearch
from .qft_phase_corr import qft_phase_correlation, apply_2d_qft
from .image_encoding import encode_frqi, decode_frqi, encode_neqr, decode_neqr, quantum_edge_detection
from .pqc import PQCSigner
from .records import QuantumRunRecord, generate_run_id
from .service import QuantumService

__all__ = [
    "QUBOBuilder",
    "SimulatedQuantumAnnealer",
    "QAOASolver",
    "QUBOMatcher",
    "QuantumBackend",
    "get_backend",
    "list_backends",
    "select_qaoa",
    "QuantumKernelClassifier",
    "GroverConsensusSearch",
    "qft_phase_correlation",
    "apply_2d_qft",
    "encode_frqi",
    "decode_frqi",
    "encode_neqr",
    "decode_neqr",
    "quantum_edge_detection",
    "PQCSigner",
    "QuantumRunRecord",
    "generate_run_id",
    "QuantumService",
]

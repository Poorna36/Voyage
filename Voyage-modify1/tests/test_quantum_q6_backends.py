"""
tests/test_quantum_q6_backends.py
=================================
Unit tests for Q6 Backend Abstraction Layer (backends.py).
"""
import numpy as np
import pytest

from quantum.backends import NumpyStatevectorBackend, get_backend, list_backends


def test_numpy_statevector_backend():
    backend = NumpyStatevectorBackend()
    avail, reason = backend.available()
    assert avail is True

    # Solve 3-qubit QUBO
    Q = np.array([
        [-1.0, 2.0, 0.0],
        [0.0, -1.0, 2.0],
        [0.0, 0.0, -1.0],
    ])
    res = backend.run_qaoa(Q, p=1, seed=42)
    assert "solution" in res
    assert len(res["solution"]) == 3
    assert res["backend"] == "numpy_statevector"


def test_backend_registry():
    b_list = list_backends()
    assert len(b_list) >= 4
    backend_ids = [b["id"] for b in b_list]
    assert "numpy_statevector" in backend_ids
    assert "qiskit_aer" in backend_ids
    assert "dwave_neal" in backend_ids

    # Safe lookup
    adapter = get_backend("numpy_statevector")
    assert adapter.name == "numpy_statevector"

    # Missing lookup raises KeyError
    with pytest.raises(KeyError):
        get_backend("nonexistent_quantum_chip")

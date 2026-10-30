"""
tests/test_quantum_simulator_aer.py
===================================
Tests for Build Spec v2 §3D: IBM-Compatible Local Quantum Simulator / Emulator.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.backends import QiskitAerBackend, get_backend


@pytest.fixture
def aer_backend():
    return QiskitAerBackend()


def test_qiskit_aer_availability(aer_backend):
    avail, reason = aer_backend.available()
    assert avail is True
    assert "AerSimulator ready" in reason


def test_qiskit_aer_circuit_spec_ideal(aer_backend):
    spec = {
        "n_qubits": 2,
        "gates": [
            {"gate": "H", "target": 0},
            {"gate": "CNOT", "control": 0, "target": 1},
        ],
    }
    res = aer_backend.run_circuit(spec, shots=500, mode="ideal_simulator")

    assert res["backend_type"] == "simulator"
    assert "ideal" in res["backend_name"]
    assert res["mode"] == "ideal_simulator"
    assert res["noise_model_applied"] is False
    assert res["shots"] == 500
    assert "counts" in res
    assert res["circuit_metadata"]["n_qubits"] == 2
    assert "cx" in res["circuit_metadata"]["count_ops"] or "h" in res["circuit_metadata"]["count_ops"]
    # Bell state counts on noiseless simulator should only contain '00' and '11'
    counts = res["counts"]
    for k in counts.keys():
        assert k in ["00", "11"]


def test_qiskit_aer_circuit_noisy(aer_backend):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    res = aer_backend.run_circuit(qc, shots=500, mode="noisy_simulator")

    assert res["backend_type"] == "simulator"
    assert res["mode"] == "noisy_simulator"
    assert res["noise_model_applied"] is True
    assert "depolarizing_p1" in res["noise_config"]
    assert res["execution_time_ms"] > 0.0
    assert len(res["counts"]) > 0


def test_qiskit_aer_fake_ibm_backend(aer_backend):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    res = aer_backend.run_circuit(qc, shots=200, mode="fake_ibm_backend")

    assert res["backend_type"] == "simulator"
    assert res["mode"] == "fake_ibm_backend"
    assert res["noise_model_applied"] is True
    assert res["backend_name"] == "fake_ibm_perth"
    assert "counts" in res


def test_transparency_reporting_guardrail(aer_backend):
    """Under no circumstances may simulator report backend_type as hardware."""
    modes = ["ideal_simulator", "noisy_simulator", "fake_ibm_backend"]
    spec = {"n_qubits": 1, "gates": [{"gate": "H", "target": 0}]}

    for m in modes:
        res = aer_backend.run_circuit(spec, shots=50, mode=m)
        assert res["backend_type"] == "simulator"
        assert res["backend_type"] != "hardware"


def test_qaoa_on_aer(aer_backend):
    Q = np.array([
        [-1.0, 2.0],
        [0.0, -1.0],
    ])
    res = aer_backend.run_qaoa(Q, p=1, shots=256, seed=42)

    assert "solution" in res
    assert len(res["solution"]) == 2
    assert res["backend_type"] == "simulator"
    assert "energy" in res

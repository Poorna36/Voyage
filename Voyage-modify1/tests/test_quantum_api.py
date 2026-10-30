"""
tests/test_quantum_api.py
=========================
FastAPI test client tests for /api/quantum/* routes and validation rules.
"""
import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


def test_quantum_status_endpoint():
    res = client.get("/api/quantum/status")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is True
    assert "modules" in data
    assert "backends" in data


def test_quantum_run_unknown_pair_404():
    # Strict validation: Never synthesize, return 404
    req = {
        "pair_id": "nonexistent_pair_99999",
        "module": "Q1_selection",
        "backend": "numpy_statevector",
        "mode": "shadow",
    }
    res = client.post("/api/quantum/run", json=req)
    assert res.status_code == 404


def test_quantum_run_invalid_module_422():
    req = {
        "pair_id": "boguslawsky",
        "module": "invalid_module_xyz",
        "backend": "numpy_statevector",
        "mode": "shadow",
    }
    res = client.post("/api/quantum/run", json=req)
    assert res.status_code == 422


def test_quantum_run_unpromoted_active_mode_403():
    # Active mode refused without promotion report
    req = {
        "pair_id": "boguslawsky",
        "module": "Q1_selection",
        "backend": "numpy_statevector",
        "mode": "active",
    }
    res = client.post("/api/quantum/run", json=req)
    assert res.status_code == 403


def test_quantum_circuit_endpoint():
    res = client.get("/api/quantum/circuit/Q1_selection")
    assert res.status_code == 200
    data = res.json()
    assert "qubits" in data
    assert "gate_sequence" in data


def test_quantum_sign_and_verify_endpoints():
    sign_res = client.post("/api/quantum/sign", json={"payload": "TEST_LUNAR_PRODUCT_GEO"})
    assert sign_res.status_code == 200
    s_data = sign_res.json()
    assert "sig_b64" in s_data
    assert "sha3_256" in s_data

    verify_res = client.post("/api/quantum/verify", json={
        "payload": "TEST_LUNAR_PRODUCT_GEO",
        "signature_b64": s_data["sig_b64"],
    })
    assert verify_res.status_code == 200
    v_data = verify_res.json()
    assert v_data["valid"] is True

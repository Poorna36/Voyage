"""
tests/test_quantum_q7_pqc.py
============================
Unit tests for Q7 Post-Quantum Signatures & Tamper Detection (pqc.py).
"""
import pytest

from quantum.pqc import PQCSigner


def test_pqc_sign_and_verify():
    signer = PQCSigner()
    payload = b"CHANDRAYAAN2_OHRC_NAC_REGISTERED_GEOTIFF_PRODUCT"

    sig_res = signer.sign_bytes(payload)
    assert "sha3_256" in sig_res
    assert "sig_b64" in sig_res

    # Valid verification
    is_valid, msg = signer.verify(payload, sig_res["sig_b64"])
    assert is_valid is True


def test_pqc_tamper_detection():
    signer = PQCSigner()
    payload = b"ORIGINAL_AUTHENTIC_CARTOGRAPHIC_DATA"

    sig_res = signer.sign_bytes(payload)

    # Corrupt 1 byte
    tampered_payload = b"ORIGINAL_AUTHENTIC_CARTOGRAPHIC_DATB"
    is_valid, msg = signer.verify(tampered_payload, sig_res["sig_b64"])

    assert is_valid is False
    assert "Invalid signature" in msg or "tampered" in msg

"""
tests/test_quantum_v2_api.py
============================
Tests for Build Spec v2 §8 REST API Endpoints.
Uses httpx.AsyncClient for clean in-process event-loop execution.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.anyio
async def test_v2_quantum_status(async_client):
    res = await async_client.get("/api/v1/quantum/status")
    assert res.status_code == 200
    data = res.json()
    assert "pipeline_mode" in data
    assert "components" in data
    assert "matcher_selector" in data["components"]
    assert "simulator" in data["components"]
    assert "backends" in data


@pytest.mark.anyio
async def test_v2_select_matcher_endpoint(async_client):
    req = {
        "pair_id": "test_pair_v2",
        "features": {
            "pair_id": "test_pair_v2",
            "sensor_pair_enc": 0,
            "gsd_ratio": 1.0,
            "delta_solar_azimuth": 10.0,
            "crater_density": 0.5,
            "src_texture_contrast": 25.0,
            "ref_texture_contrast": 22.0,
        },
        "strategy": "rules",
    }
    res = await async_client.post("/api/v1/quantum/select-matcher", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["selected_matcher"] == "sift"
    assert "Selected sift" in data["explanation"]
    assert data["confidence"] > 0.5


@pytest.mark.anyio
async def test_v2_select_matcher_override(async_client):
    req = {
        "pair_id": "test_pair_v2",
        "features": {},
        "override_matcher": "rift2",
    }
    res = await async_client.post("/api/v1/quantum/select-matcher", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["selected_matcher"] == "rift2"
    assert data["manual_override"] is True


@pytest.mark.anyio
async def test_v2_qnn_classify(async_client):
    req = {
        "features": [
            [0.1, 0.2, 0.1, 0.1],
            [0.8, 0.9, 0.6, 0.7],
        ],
        "threshold": 0.5,
    }
    res = await async_client.post("/api/v1/quantum/qnn-classify", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["n_samples"] == 2
    assert len(data["scores"]) == 2
    assert len(data["labels"]) == 2


@pytest.mark.anyio
async def test_v2_qcnn_extract(async_client):
    patch = [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
    ]
    req = {
        "patch": patch,
        "descriptor_dim": 4,
    }
    res = await async_client.post("/api/v1/quantum/qcnn-extract", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["dim"] == 4
    assert len(data["descriptor"]) == 4
    assert abs(data["norm"] - 1.0) < 1e-3


@pytest.mark.anyio
async def test_v2_qaoa_optimize(async_client):
    req = {
        "src_pts": [
            [10.0, 10.0],
            [50.0, 10.0],
            [10.0, 50.0],
            [50.0, 50.0],
            [90.0, 10.0],
        ],
        "ref_pts": [
            [20.0, 15.0],
            [60.0, 15.0],
            [20.0, 55.0],
            [60.0, 55.0],
            [100.0, 15.0],
        ],
        "candidate_pairs": [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
        ],
        "affinities": [0.9, 0.85, 0.88, 0.92, 0.80],
        "p_layers": 1,
        "use_qaoa": True,
    }
    res = await async_client.post("/api/v1/quantum/qaoa-optimize", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["n_candidates"] == 5
    assert data["n_selected"] >= 4
    assert "constraint_diagnostics" in data
    assert data["constraint_diagnostics"]["is_valid"] is True


@pytest.mark.anyio
async def test_v2_ablation_summary(async_client):
    res = await async_client.get("/api/v1/quantum/ablation-summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_configs"] == 6
    assert len(data["configurations"]) == 6
    assert len(data["markdown_table"]) > 0

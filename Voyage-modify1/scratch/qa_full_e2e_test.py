"""
qa_full_e2e_test.py
===================
Comprehensive End-to-End QA Validation Suite for Voyage Quantum Pipeline.
Validates:
  - System Health and Service Info
  - Datasets & Manifest Routes
  - Pipeline Configuration & Matchers
  - Science Landing Zone & Spectral Telemetry
  - Pipeline Run Execution & Calibration
  - Quantum v1 and v2 Endpoints (§3A, §3B, §3C, §3D, §3E)
  - Post-Quantum Cryptographic Signatures
"""
import sys
import time
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

def run_qa_checks():
    checks_passed = 0
    checks_total = 0

    def assert_check(name, condition, details=""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {details}")

    print("=" * 70)
    print("VOYAGE FULL END-TO-END QA AUDIT")
    print("=" * 70)

    # 1. Health Endpoint
    res = client.get("/api/health")
    assert_check("GET /api/health", res.status_code == 200 and res.json().get("status") == "online")

    # 2. Datasets
    res = client.get("/api/datasets/")
    assert_check("GET /api/datasets/", res.status_code == 200 and isinstance(res.json(), list))

    res = client.get("/api/datasets/stats")
    assert_check("GET /api/datasets/stats", res.status_code == 200 and "total_pairs" in res.json())

    # 3. Config
    # /api/config/ returns {"configs": [...yaml files...]}
    res = client.get("/api/config/")
    assert_check("GET /api/config/", res.status_code == 200 and "configs" in res.json() and isinstance(res.json()["configs"], list))

    # /api/config/matchers returns flat dict of matcher name -> config
    res = client.get("/api/config/matchers")
    assert_check("GET /api/config/matchers", res.status_code == 200 and "sift" in res.json())

    # 4. Science Routes
    res = client.get("/api/science/craters/")
    assert_check("GET /api/science/craters/", res.status_code == 200 and len(res.json()) >= 6)

    # name field is "Boguslawsky Crater" (full name), id is "boguslawsky"
    res = client.get("/api/science/craters/boguslawsky")
    cdata = res.json()
    assert_check(
        "GET /api/science/craters/boguslawsky",
        res.status_code == 200 and cdata.get("id") == "boguslawsky" and "Boguslawsky" in cdata.get("name", "")
    )

    # SLZ response uses go_no_go, not verdict
    res = client.get("/api/science/slz/boguslawsky")
    slz = res.json()
    assert_check(
        "GET /api/science/slz/boguslawsky",
        res.status_code == 200 and "go_no_go" in slz and slz.get("overall_safety_score") is not None
    )

    # Spectral response uses "data" key for band array
    res = client.get("/api/science/spectral/boguslawsky")
    spec = res.json()
    assert_check(
        "GET /api/science/spectral/boguslawsky",
        res.status_code == 200 and "data" in spec and isinstance(spec["data"], list) and len(spec["data"]) > 0
    )

    # Keypoints endpoint returns list (may be empty when no processed data exists)
    res = client.get("/api/science/keypoints/boguslawsky")
    assert_check(
        "GET /api/science/keypoints/boguslawsky",
        res.status_code == 200 and isinstance(res.json(), list)
    )

    # 5. Pipeline Run
    run_req = {"pair_id": "boguslawsky", "matcher": "lightglue"}
    res = client.post("/api/pipeline/run", json=run_req)
    data = res.json()
    assert_check(
        "POST /api/pipeline/run (LightGlue)",
        res.status_code == 200 and data.get("status") == "completed" and data.get("metrics", {}).get("rmse_px") is not None
    )

    # 6. Pipeline History
    res = client.get("/api/pipeline/history")
    assert_check("GET /api/pipeline/history", res.status_code == 200 and len(res.json()) >= 1)

    # 7. Quantum Status (v1 and v2)
    res = client.get("/api/quantum/status")
    q_data = res.json()
    assert_check(
        "GET /api/quantum/status",
        res.status_code == 200 and "components" in q_data and "backends" in q_data
    )

    res = client.get("/api/v1/quantum/status")
    assert_check(
        "GET /api/v1/quantum/status (Spec v2 alias)",
        res.status_code == 200 and res.json().get("pipeline_mode") is not None
    )

    # 8. Intelligent Matcher Selector (§3A)
    sel_req = {
        "pair_id": "boguslawsky",
        "features": {
            "sensor_pair_enc": 0,
            "gsd_ratio": 1.0,
            "delta_solar_azimuth": 8.0,
            "crater_density": 0.4,
            "src_texture_contrast": 26.0,
            "ref_texture_contrast": 24.0,
        },
        "strategy": "rules",
    }
    res = client.post("/api/quantum/select-matcher", json=sel_req)
    ms_data = res.json()
    assert_check(
        "POST /api/quantum/select-matcher (Rule-based §3A)",
        res.status_code == 200 and ms_data.get("selected_matcher") in ["sift", "rift2", "lightglue", "crater"] and ms_data.get("confidence") > 0.5
    )

    # Override test
    res = client.post("/api/quantum/select-matcher", json={"override_matcher": "rift2"})
    assert_check(
        "POST /api/quantum/select-matcher (Manual Override)",
        res.status_code == 200 and res.json().get("selected_matcher") == "rift2" and res.json().get("manual_override") is True
    )

    # 9. QNN Match Reliability (§3B)
    qnn_req = {
        "features": [
            [0.1, 0.2, 0.1, 0.05],
            [0.85, 0.9, 0.5, 0.7],
        ],
        "threshold": 0.5,
    }
    res = client.post("/api/quantum/qnn-classify", json=qnn_req)
    qnn_data = res.json()
    assert_check(
        "POST /api/quantum/qnn-classify (§3B)",
        res.status_code == 200 and qnn_data.get("n_samples") == 2 and len(qnn_data.get("scores", [])) == 2
    )

    # 10. QCNN Patch Features (§3C)
    qcnn_req = {
        "patch": [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.8, 0.9, 0.3],
            [0.3, 0.9, 0.8, 0.2],
            [0.4, 0.3, 0.2, 0.1],
        ],
        "descriptor_dim": 4,
    }
    res = client.post("/api/quantum/qcnn-extract", json=qcnn_req)
    qcnn_data = res.json()
    assert_check(
        "POST /api/quantum/qcnn-extract (§3C)",
        res.status_code == 200 and qcnn_data.get("dim") == 4 and abs(qcnn_data.get("norm", 0) - 1.0) < 0.05
    )

    # 11. QAOA Optimization (§3E)
    qaoa_req = {
        "src_pts": [[10, 10], [50, 10], [50, 50], [10, 50], [25, 25], [30, 30]],
        "ref_pts": [[15, 15], [55, 15], [55, 55], [15, 55], [30, 30], [200, 200]],
        "candidate_pairs": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        "affinities": [0.9, 0.9, 0.9, 0.9, 0.85, 0.3],
        "p_layers": 1,
        "use_qaoa": True,
    }
    res = client.post("/api/quantum/qaoa-optimize", json=qaoa_req)
    qaoa_data = res.json()
    assert_check(
        "POST /api/quantum/qaoa-optimize (§3E)",
        res.status_code == 200
        and qaoa_data.get("n_selected", 0) >= 4
        and qaoa_data.get("constraint_status") == "feasible"
        and "constraint_diagnostics" in qaoa_data
    )

    # 12. Circuit Visualization
    res = client.get("/api/quantum/circuit/Q1_selection")
    assert_check("GET /api/quantum/circuit/Q1_selection", res.status_code == 200 and "gate_sequence" in res.json())

    res = client.get("/api/quantum/circuit/Q4_phase_corr")
    assert_check("GET /api/quantum/circuit/Q4_phase_corr", res.status_code == 200 and "gate_sequence" in res.json())

    # 13. Post-Quantum Signing & Tamper Verification
    sign_res = client.post("/api/quantum/sign", json={"payload": "VOYAGE_LUNAR_CORE_CHECKPOINT_2026"})
    sig_data = sign_res.json()
    assert_check(
        "POST /api/quantum/sign",
        sign_res.status_code == 200 and "sig_b64" in sig_data and "sha3_256" in sig_data
    )

    verify_res = client.post("/api/quantum/verify", json={
        "payload": "VOYAGE_LUNAR_CORE_CHECKPOINT_2026",
        "signature_b64": sig_data["sig_b64"],
    })
    v_data = verify_res.json()
    assert_check(
        "POST /api/quantum/verify (Authentic payload)",
        verify_res.status_code == 200 and v_data.get("valid") is True
    )

    # Tampered payload check
    tamper_res = client.post("/api/quantum/verify", json={
        "payload": "TAMPERED_DATA_XYZ",
        "signature_b64": sig_data["sig_b64"],
    })
    assert_check(
        "POST /api/quantum/verify (Tampered payload rejected)",
        tamper_res.status_code == 200 and tamper_res.json().get("valid") is False
    )

    print("=" * 70)
    print(f"QA RESULT: {checks_passed} / {checks_total} checks passed.")
    print("=" * 70)
    assert checks_passed == checks_total, f"{checks_total - checks_passed} checks failed"

if __name__ == "__main__":
    run_qa_checks()


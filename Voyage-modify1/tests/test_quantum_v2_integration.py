"""
tests/test_quantum_v2_integration.py
====================================
End-to-end integration and regression tests for Build Spec v2 (§9, §11).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.config_validator import validate_quantum_config, load_and_validate_quantum_config
from quantum.matcher_selector import RuleBasedMatcherSelector, get_available_matchers
from quantum.qnn_reliability import QNNReliabilityClassifier, extract_match_features
from quantum.qcnn_features import QCNNFeatureExtractor
from quantum.qaoa_correspondence import QAOACorrespondenceOptimizer
from quantum.records import QuantumRunRecord, generate_run_id

# qiskit_aer's C extension crashes on Python 3.14 / Win32 (fatal exception 0xc0000008).
# Guard the import and skip the single test that exercises QiskitAerBackend directly.
_QiskitAerBackend = None
_AER_SKIP = None
if sys.platform == "win32":
    _AER_SKIP = "qiskit-aer C extension crashes on Python 3.14 / Win32 (0xc0000008)"
else:
    try:
        from quantum.backends import QiskitAerBackend as _QiskitAerBackend  # type: ignore[assignment]
    except Exception as _e:
        _AER_SKIP = f"qiskit-aer unavailable: {_e}"


@pytest.fixture
def synthetic_pair_data():
    rng = np.random.default_rng(101)
    n = 20
    src_pts = rng.uniform(50.0, 400.0, size=(n, 2)).astype(np.float32)
    # Rigid transformation: shift + 2 deg rotation
    theta = np.radians(2.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], dtype=np.float32)
    shift = np.array([25.0, -15.0], dtype=np.float32)
    ref_pts = (src_pts @ R.T + shift + rng.normal(0, 0.2, size=(n, 2))).astype(np.float32)

    # Add 4 outlier matches
    ref_pts[-4:] = rng.uniform(50.0, 400.0, size=(4, 2)).astype(np.float32)

    candidate_pairs = np.array([[i, i] for i in range(n)], dtype=np.int32)
    affinities = np.array([0.92] * (n - 4) + [0.35] * 4, dtype=np.float32)

    meta = {
        "pair_id": "test_lunar_integration_pair_001",
        "sensor_pair_enc": 0,
        "gsd_ratio": 1.0,
        "delta_solar_azimuth": 8.5,
        "crater_density": 0.4,
        "src_texture_contrast": 26.0,
        "ref_texture_contrast": 24.0,
        "terrain_class_enc": 0,
    }

    return src_pts, ref_pts, candidate_pairs, affinities, meta


def test_pure_classical_zero_quantum_config():
    """Verify classical mode validates without requiring any quantum dependencies."""
    cfg = {
        "quantum": {
            "enabled": False,
            "mode": "classical",
        }
    }
    validated = validate_quantum_config(cfg)
    assert validated["enabled"] is False
    assert validated["mode"] == "classical"


def test_matcher_selector_individual_enabled(synthetic_pair_data):
    _, _, _, _, meta = synthetic_pair_data
    ms = RuleBasedMatcherSelector()
    res = ms.select(meta)

    assert res.selected_matcher in get_available_matchers()
    assert res.confidence >= 0.65
    assert len(res.explanation) > 0


def test_qnn_reliability_individual_enabled(synthetic_pair_data):
    src_pts, ref_pts, pairs, affinities, _ = synthetic_pair_data
    clf = QNNReliabilityClassifier(num_features=4)

    # Extract 4-feature vector for candidate pairs
    feats = np.zeros((len(pairs), 4), dtype=np.float64)
    for i in range(len(pairs)):
        feats[i] = [
            1.0 - affinities[i],  # descriptor distance
            0.6 if i < 16 else 0.95,  # ratio
            0.1,  # scale ratio
            0.05 if i < 16 else 0.8,  # angle diff
        ]

    scores = clf.predict_proba(feats)
    assert len(scores) == len(pairs)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_qcnn_patch_features_individual_enabled():
    qcnn = QCNNFeatureExtractor(num_qubits=4, descriptor_dim=4)
    patch = np.random.uniform(0.1, 0.9, size=(4, 4))
    desc = qcnn.extract_descriptor(patch)

    assert len(desc) == 4
    assert abs(np.linalg.norm(desc) - 1.0) < 1e-4


def test_qaoa_correspondence_individual_enabled(synthetic_pair_data):
    src_pts, ref_pts, pairs, affinities, _ = synthetic_pair_data
    sub_pairs = pairs[:8]
    sub_aff = affinities[:8]

    opt = QAOACorrespondenceOptimizer(p_layers=1, max_vars=8)
    res = opt.optimize(src_pts, ref_pts, sub_pairs, sub_aff, use_qaoa=True)

    assert res.n_selected >= 4
    assert res.constraint_diagnostics["is_valid"] is True
    assert res.homography_result is not None
    assert res.homography_result["success"] is True


def test_local_aer_simulator_individual_enabled():
    if _AER_SKIP:
        pytest.skip(_AER_SKIP)
    aer = _QiskitAerBackend(mode="ideal_simulator")
    spec = {"n_qubits": 2, "gates": [{"gate": "H", "target": 0}, {"gate": "CX", "control": 0, "target": 1}]}
    res = aer.run_circuit(spec, shots=200)

    assert res["backend_type"] == "simulator"
    assert res["noise_model_applied"] is False
    assert len(res["counts"]) > 0


def test_all_components_combined_hybrid(synthetic_pair_data):
    """
    Test full end-to-end hybrid workflow:
    1. Matcher Selector selects matcher
    2. QNN filters candidate matches
    3. QAOA correspondence optimizer selects mutually consistent set
    4. RANSAC verifies homography and inliers
    5. Saves full QuantumRunRecord with git commit and repro command
    """
    src_pts, ref_pts, candidate_pairs, affinities, meta = synthetic_pair_data

    # Step 1: Matcher Selector
    ms = RuleBasedMatcherSelector()
    ms_res = ms.select(meta)
    assert ms_res.selected_matcher in ["sift", "rift2", "lightglue", "crater"]

    # Step 2: QNN Reliability filtering
    qnn = QNNReliabilityClassifier(num_features=4)
    feats = np.zeros((len(candidate_pairs), 4), dtype=np.float64)
    for i in range(len(candidate_pairs)):
        feats[i] = [
            float(1.0 - affinities[i]),
            0.6 if i < 16 else 0.95,
            0.1,
            0.05 if i < 16 else 0.8,
        ]
    qnn_probs = qnn.predict_proba(feats)
    # Keep candidates passing threshold
    keep = qnn_probs >= 0.4
    filtered_pairs = candidate_pairs[keep]
    filtered_affinities = affinities[keep]
    assert len(filtered_pairs) >= 8

    # Step 3: QAOA Correspondence Optimization
    qaoa_opt = QAOACorrespondenceOptimizer(p_layers=1, max_vars=12)
    qaoa_res = qaoa_opt.optimize(
        src_pts,
        ref_pts,
        filtered_pairs[:12],
        filtered_affinities[:12],
        use_qaoa=True,
    )

    assert qaoa_res.n_selected >= 4
    assert qaoa_res.constraint_diagnostics["is_valid"] is True
    assert qaoa_res.homography_result["success"] is True

    # Step 4: Record output verification
    run_id = generate_run_id("test_hybrid")
    rec = QuantumRunRecord(
        run_id=run_id,
        pair_id=meta["pair_id"],
        module="combined_hybrid",
        mode="active",
        pipeline_mode="hybrid",
        active_components=["matcher_selector", "qnn_reliability", "qaoa_correspondence"],
        backend_type="simulator",
        backend_mode="ideal_simulator",
        repro_command="pytest tests/test_quantum_v2_integration.py",
        metrics={
            "n_inliers": qaoa_res.homography_result["n_inliers"],
            "inlier_ratio": qaoa_res.homography_result["inlier_ratio"],
            "qaoa_energy": qaoa_res.qaoa_energy,
        },
        decision_logs={
            "matcher": ms_res.selected_matcher,
            "explanation": ms_res.explanation,
            "diagnostics": qaoa_res.constraint_diagnostics,
        },
    )

    d = rec.to_dict()
    assert d["pipeline_mode"] == "hybrid"
    assert d["backend_type"] == "simulator"
    assert len(d["git_commit"]) > 0
    assert "n_inliers" in d["metrics"]

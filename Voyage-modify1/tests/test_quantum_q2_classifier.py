"""
tests/test_quantum_q2_classifier.py
===================================
Unit tests for Q2 Quantum-Kernel Classifier (kernel_classifier.py).
"""
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from quantum.kernel_classifier import (
    QuantumKernelClassifier,
    compute_quantum_gram_matrix,
    run_q2_benchmark,
)


def test_gram_matrix_properties():
    # 5 samples in 8 dimensions
    X = np.random.uniform(0.0, 2.0 * np.pi, size=(5, 8))
    K, states = compute_quantum_gram_matrix(X, reps=1)

    # 1. Symmetry
    assert np.allclose(K, K.T)

    # 2. Unit diagonal: K_ii = |<phi|phi>|^2 = 1.0
    assert np.allclose(np.diag(K), 1.0)

    # 3. Positive Semi-Definiteness (eigenvalues >= -1e-8)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-8)


def test_quantum_classifier_fit_predict():
    X = np.random.normal(size=(20, 8))
    y = (X[:, 0] > 0).astype(int)

    clf = QuantumKernelClassifier(reps=1, C=1.0, seed=42)
    clf.fit(X, y)
    preds = clf.predict(X[:5])

    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_run_q2_benchmark():
    res = run_q2_benchmark(task="inlier", n_samples=30, seed=42)
    assert "quantum_accuracy" in res
    assert "classical_accuracy" in res
    assert res["qubits"] == 8
    assert res.get("benchmark_label") == "SYNTHETIC SMOKE TEST - not evidence of real-data performance"


def test_q2_groupkfold_no_scene_leakage():
    """Verify that GroupKFold strictly prevents scene/group leakage between train and test."""
    groups = np.array(["scene_A"] * 20 + ["scene_B"] * 25 + ["scene_C"] * 30 + ["scene_D"] * 20 + ["scene_E"] * 25)
    X = np.random.normal(size=(len(groups), 7))
    y = np.random.randint(0, 2, size=len(groups))

    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        assert len(overlap) == 0, f"Fold {fold} leaked groups across train and test: {overlap}"


def test_q2_known_homography_labels():
    """Verify inlier labels on a known homography under symmetric reprojection threshold 3 px."""
    H_true = np.array([
        [1.05, -0.05, 12.0],
        [0.05,  1.05, -8.0],
        [0.00,  0.00,  1.0],
    ], dtype=np.float64)
    H_inv = np.linalg.inv(H_true)

    # 10 true inliers exactly transformed by H_true
    src_inliers = np.array([[50.0, 50.0], [100.0, 80.0], [150.0, 200.0], [200.0, 120.0]], dtype=np.float64)
    ones = np.ones((len(src_inliers), 1))
    src_h = np.hstack([src_inliers, ones])
    ref_h = (H_true @ src_h.T).T
    ref_inliers = ref_h[:, :2] / ref_h[:, 2:3]

    # 4 outliers: perturb ref points by 20-50 px
    src_outliers = np.copy(src_inliers)
    ref_outliers = ref_inliers + np.array([[30.0, -25.0], [-40.0, 20.0], [50.0, 50.0], [-35.0, -40.0]])

    all_src = np.vstack([src_inliers, src_outliers])
    all_ref = np.vstack([ref_inliers, ref_outliers])

    # Compute symmetric reprojection error
    p_fwd = np.hstack([all_src, np.ones((len(all_src), 1))]) @ H_true.T
    err_fwd = np.linalg.norm(p_fwd[:, :2] / (p_fwd[:, 2:3] + 1e-12) - all_ref, axis=1)

    p_bwd = np.hstack([all_ref, np.ones((len(all_ref), 1))]) @ H_inv.T
    err_bwd = np.linalg.norm(p_bwd[:, :2] / (p_bwd[:, 2:3] + 1e-12) - all_src, axis=1)

    err_sym = np.sqrt(0.5 * (err_fwd**2 + err_bwd**2))
    labels = (err_sym <= 3.0).astype(int)

    # First 4 are inliers (label 1), last 4 are outliers (label 0)
    assert np.array_equal(labels[:4], [1, 1, 1, 1])
    assert np.array_equal(labels[4:], [0, 0, 0, 0])

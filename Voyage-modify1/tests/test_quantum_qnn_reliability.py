"""
tests/test_quantum_qnn_reliability.py
=====================================
Tests for Build Spec v2 §3B: QNN Candidate-Match Reliability Classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.qnn_reliability import (
    QNNReliabilityClassifier,
    extract_match_features,
    compute_metrics,
    compute_calibration_data,
    train_and_evaluate_qnn,
)


def test_extract_match_features_shapes_and_bounds():
    des1 = np.random.rand(128).astype(np.float32)
    des2 = des1 + np.random.normal(0, 0.05, 128).astype(np.float32)
    des_2nd = np.random.rand(128).astype(np.float32)

    feat = extract_match_features(
        src_pt=np.array([10.0, 10.0]),
        ref_pt=np.array([12.0, 11.0]),
        des_src=des1,
        des_ref=des2,
        des_ref_2nd=des_2nd,
        scale_src=1.5,
        scale_ref=1.2,
        angle_src_deg=30.0,
        angle_ref_deg=45.0,
    )

    assert feat.shape == (4,)
    # All features must be normalized in [0, 1]
    assert np.all(feat >= 0.0)
    assert np.all(feat <= 1.0)


def test_qnn_classifier_init_and_forward():
    clf = QNNReliabilityClassifier(num_features=4)
    X = np.random.rand(5, 4)
    probs = clf.predict_proba(X)

    assert probs.shape == (5,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    preds = clf.predict(X, threshold=0.5)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})


def test_qnn_fit_and_loss_decrease():
    rng = np.random.default_rng(42)
    # Synthetic dataset: inliers have lower distance/ratio
    X_pos = rng.uniform(0.0, 0.35, size=(12, 4))
    X_neg = rng.uniform(0.65, 1.0, size=(12, 4))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 12 + [0] * 12)

    clf = QNNReliabilityClassifier(num_features=4)
    init_probs = clf.predict_proba(X)
    init_loss = np.mean((init_probs - y) ** 2)

    clf.fit(X, y, maxiter=25)
    trained_probs = clf.predict_proba(X)
    trained_loss = np.mean((trained_probs - y) ** 2)

    # Training should decrease error
    assert trained_loss <= init_loss


def test_metrics_and_calibration():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3])

    metrics = compute_metrics(y_true, y_prob)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["brier_score"] < 0.1

    calib = compute_calibration_data(y_true, y_prob, n_bins=3)
    assert "bin_centers" in calib
    assert "bin_accuracies" in calib
    assert len(calib["bin_centers"]) > 0


def test_train_and_evaluate_qnn_benchmark():
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.0, 1.0, size=(16, 4))
    y_train = (X_train[:, 0] + X_train[:, 1] < 1.0).astype(int)

    X_test = rng.uniform(0.0, 1.0, size=(8, 4))
    y_test = (X_test[:, 0] + X_test[:, 1] < 1.0).astype(int)

    report = train_and_evaluate_qnn(X_train, y_train, X_test, y_test, maxiter=15)

    assert "precision" in report.qnn_metrics
    assert "brier_score" in report.qnn_metrics
    assert "precision" in report.logistic_regression_metrics
    assert "precision" in report.decision_tree_metrics
    assert report.n_samples == 8

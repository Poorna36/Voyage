"""
tests/test_quantum_q2_classifier.py
===================================
Unit tests for Q2 Quantum-Kernel Classifier (kernel_classifier.py).
"""
import numpy as np
import pytest

from quantum.kernel_classifier import QuantumKernelClassifier, compute_quantum_gram_matrix, run_q2_benchmark


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

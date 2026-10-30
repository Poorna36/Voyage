"""
quantum/kernel_classifier.py
============================
Q2 — Quantum-Kernel Classifier (L1.5 Router and L4 Inlier Scoring).

Computes a Quantum Kernel Gram Matrix using an 8-qubit feature map:
  1. H^\\otimes 8
  2. R_z(x_i) on each qubit i
  3. Ring of ZZ interactions: exp(-i (pi - x_i)(pi - x_j) Z_i Z_j)
  4. Repeat reps = 2 layers

Kernel:
  K(x, x') = |<phi(x') | phi(x)>|^2

Two applications:
  - Q2a Router: DEFERRED (router not trained: requires real PairRecords).
  - Q2b Inlier Scorer: Classify matches as inlier/outlier from 8 per-match features.

Classical Twins:
  - RBF-SVM (sklearn.svm.SVC)
  - Logistic Regression / LightGBM baseline
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

log = logging.getLogger(__name__)


def simulate_quantum_feature_state(x: np.ndarray, reps: int = 2) -> np.ndarray:
    """
    Simulate the statevector |phi(x)> for an 8-dimensional feature vector x in [0, 2*pi].
    Uses 8 qubits -> 256 complex amplitudes.
    """
    n_qubits = 8
    dim = 1 << n_qubits  # 256

    # 1. Start in |00000000>
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0

    # Hadamard tensor product creates uniform superposition: 1/sqrt(256) everywhere
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Precompute bit values for all 256 basis states
    # basis_bits[s, q] = bit q of basis state s
    basis_bits = np.zeros((dim, n_qubits), dtype=int)
    for s in range(dim):
        for q in range(n_qubits):
            basis_bits[s, q] = (s >> q) & 1

    # Spin values: s_z = 1 - 2*bit in {+1, -1}
    spins = 1.0 - 2.0 * basis_bits

    for _ in range(reps):
        # 2. Single-qubit phase rotation R_z(x_i)
        # Phase applied to basis state s is sum_q (x_q * spins[s, q])
        phase_single = np.sum(spins * (x.reshape(1, n_qubits)), axis=1)
        psi = psi * np.exp(-0.5j * phase_single)

        # 3. Ring of ZZ interactions: exp(-i (pi - x_i)(pi - x_j) Z_i Z_j)
        phase_zz = np.zeros(dim, dtype=float)
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            coeff = (np.pi - x[i]) * (np.pi - x[j])
            phase_zz += coeff * (spins[:, i] * spins[:, j])
        psi = psi * np.exp(-1.0j * phase_zz)

    # Normalization check
    norm = np.linalg.norm(psi)
    if norm > 1e-12:
        psi = psi / norm
    return psi


def compute_quantum_gram_matrix(X: np.ndarray, reps: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full Gram matrix K_ij = |<phi(x_j) | phi(x_i)>|^2 for all samples in X.
    X: (N, 8) in [0, 2*pi]
    Returns (Gram_matrix, statevectors)
    """
    N = len(X)
    dim = 256
    statevectors = np.zeros((N, dim), dtype=complex)

    for i in range(N):
        statevectors[i] = simulate_quantum_feature_state(X[i], reps=reps)

    # Overlap matrix: S_ij = <phi(x_i) | phi(x_j)> = Psi @ Psi^dagger
    S = statevectors @ np.conjugate(statevectors.T)
    # Kernel K_ij = |S_ij|^2
    K = np.abs(S) ** 2

    # Enforce exact symmetry and unit diagonal
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K, statevectors


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine using a Quantum Feature Map Kernel.
    """
    def __init__(self, reps: int = 2, C: float = 1.0, seed: int = 42):
        self.reps = reps
        self.C = C
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.train_statevectors: Optional[np.ndarray] = None
        self.svm: Optional[SVC] = None
        self.is_fitted = False

    def _prepare_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            if X_scaled.shape[1] > 8:
                self.pca = PCA(n_components=8, random_state=self.seed)
                X_reduced = self.pca.fit_transform(X_scaled)
            elif X_scaled.shape[1] < 8:
                # Pad with zeros to 8 dims
                pad = np.zeros((len(X_scaled), 8 - X_scaled.shape[1]))
                X_reduced = np.hstack([X_scaled, pad])
            else:
                X_reduced = X_scaled
        else:
            X_scaled = self.scaler.transform(X)
            if self.pca is not None:
                X_reduced = self.pca.transform(X_scaled)
            elif X_scaled.shape[1] < 8:
                pad = np.zeros((len(X_scaled), 8 - X_scaled.shape[1]))
                X_reduced = np.hstack([X_scaled, pad])
            else:
                X_reduced = X_scaled

        # Map to [0, 2*pi] using sigmoid scaling
        X_angles = 2.0 * np.pi / (1.0 + np.exp(-X_reduced))
        return X_angles

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumKernelClassifier:
        t0 = time.time()
        X_angles = self._prepare_features(X, fit=True)
        K_train, self.train_statevectors = compute_quantum_gram_matrix(X_angles, reps=self.reps)

        self.svm = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.seed)
        self.svm.fit(K_train, y)
        self.is_fitted = True
        self.train_time_s = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.svm is None or self.train_statevectors is None:
            raise RuntimeError("Model is not fitted.")
        X_angles = self._prepare_features(X, fit=False)
        N_test = len(X_angles)
        test_statevectors = np.zeros((N_test, 256), dtype=complex)
        for i in range(N_test):
            test_statevectors[i] = simulate_quantum_feature_state(X_angles[i], reps=self.reps)

        # Cross kernel: K_cross[i, j] = |<test_i | train_j>|^2
        S = test_statevectors @ np.conjugate(self.train_statevectors.T)
        K_cross = np.abs(S) ** 2
        return self.svm.predict(K_cross)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.svm is None or self.train_statevectors is None:
            raise RuntimeError("Model is not fitted.")
        X_angles = self._prepare_features(X, fit=False)
        N_test = len(X_angles)
        test_statevectors = np.zeros((N_test, 256), dtype=complex)
        for i in range(N_test):
            test_statevectors[i] = simulate_quantum_feature_state(X_angles[i], reps=self.reps)

        S = test_statevectors @ np.conjugate(self.train_statevectors.T)
        K_cross = np.abs(S) ** 2
        return self.svm.predict_proba(K_cross)


def run_q2_benchmark(
    task: str = "inlier",
    n_samples: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    # SYNTHETIC SMOKE TEST - not evidence of real-data performance
    Run evaluation comparing Quantum Kernel vs Classical RBF-SVM Twin.
    task: "router" (13 features -> 3 classes) or "inlier" (8 features -> binary inlier/outlier).
    """
    rng = np.random.default_rng(seed)

    if task == "router":
        return {
            "router_status": "router not trained: requires real PairRecords",
            "benchmark_label": "SYNTHETIC SMOKE TEST - not evidence of real-data performance",
            "task": "router",
            "status": "DEFERRED",
            "qubits": 8,
            "quantum_accuracy": 0.0,
            "classical_accuracy": 0.0,
        }
    else:
        # 8 match features -> binary inlier
        X = rng.uniform(0.0, 1.0, size=(n_samples, 8))
        # Inliers have low residual and high confidence
        score = X[:, 5] * 2.0 - X[:, 4] * 2.5 + rng.normal(0, 0.3, n_samples)
        y = (score > 0.3).astype(int)

    # Train / Test split (70 / 30)
    n_train = int(0.70 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # 1. Quantum Kernel Model
    t0 = time.time()
    qk_model = QuantumKernelClassifier(reps=2, C=1.0, seed=seed)
    qk_model.fit(X_train, y_train)
    y_pred_qk = qk_model.predict(X_test)
    qk_time = time.time() - t0
    acc_qk = float(accuracy_score(y_test, y_pred_qk))

    # 2. Classical Twin: RBF-SVM
    t_twin_0 = time.time()
    twin_svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    scaler = StandardScaler()
    twin_svm.fit(scaler.fit_transform(X_train), y_train)
    y_pred_twin = twin_svm.predict(scaler.transform(X_test))
    twin_time = time.time() - t_twin_0
    acc_twin = float(accuracy_score(y_test, y_pred_twin))

    # AUC if binary
    auc_qk = None
    auc_twin = None
    if len(np.unique(y)) == 2:
        try:
            proba_qk = qk_model.predict_proba(X_test)[:, 1]
            auc_qk = float(roc_auc_score(y_test, proba_qk))
            proba_twin = twin_svm.predict_proba(scaler.transform(X_test))[:, 1]
            auc_twin = float(roc_auc_score(y_test, proba_twin))
        except Exception:
            pass

    return {
        "benchmark_label": "SYNTHETIC SMOKE TEST - not evidence of real-data performance",
        "task": task,
        "n_samples": n_samples,
        "n_train": n_train,
        "n_test": len(X_test),
        "qubits": 8,
        "reps": 2,
        "quantum_accuracy": round(acc_qk, 4),
        "classical_accuracy": round(acc_twin, 4),
        "quantum_auc": round(auc_qk, 4) if auc_qk is not None else None,
        "classical_auc": round(auc_twin, 4) if auc_twin is not None else None,
        "quantum_runtime_s": round(qk_time, 4),
        "classical_twin_runtime_s": round(twin_time, 4),
    }

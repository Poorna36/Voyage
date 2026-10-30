"""
quantum/qcnn_features.py
========================
Quantum Convolutional Neural Network (QCNN) for Small-Patch Feature Extraction (Build Spec v2 §3C).

Extracts compact descriptor representations from small lunar surface image patches
(e.g., around keypoints, crater rims) for cross-sensor or extreme-illumination pairs.

Architecture:
  - Input: Small grayscale image patch (4x4 or 8x8 pixels, normalized to [0, 1]).
  - Encoding: 4-qubit hierarchical angle encoding.
  - Quantum Convolution Layers: Parameterized two-qubit unitary operations with
    convolutional translational invariance across qubit pairs.
  - Quantum Pooling Layers: Controlled reduction operations reducing active qubit count (4 -> 2).
  - Measurement Layer: Multi-basis Pauli expectation values (Z, X) producing a
    compact 4-8 dimensional L2-normalized feature descriptor.
  - Metric Learning: Contrastive / Triplet loss training (anchor, positive, negative).
  - Classical Baselines:
      - Classical PCA patch descriptor
      - Classical simple Conv2D patch descriptor
  - Cross-Sensor Evaluation with Geo-Scene Disjoint Splitting.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger("quantum.qcnn_features")


def preprocess_patch(patch: np.ndarray, target_size: int = 4) -> np.ndarray:
    """
    Ensure patch is a 2D float64 array of shape (target_size, target_size)
    normalized to [0.0, 1.0].
    """
    p = np.asarray(patch, dtype=np.float64)
    if p.ndim == 3:
        p = np.mean(p, axis=2)

    h, w = p.shape
    if (h, w) != (target_size, target_size):
        # Downsample or resize to target_size x target_size
        row_bins = np.array_split(np.arange(h), target_size)
        col_bins = np.array_split(np.arange(w), target_size)
        resized = np.zeros((target_size, target_size), dtype=np.float64)
        for i, r_idx in enumerate(row_bins):
            for j, c_idx in enumerate(col_bins):
                resized[i, j] = float(np.mean(p[np.ix_(r_idx, c_idx)]))
        p = resized

    # Normalize to [0, 1]
    p_min, p_max = float(np.min(p)), float(np.max(p))
    if p_max > p_min:
        p_norm = (p - p_min) / (p_max - p_min)
    else:
        p_norm = np.zeros_like(p)

    return p_norm


@dataclass
class QCNNEvaluationResult:
    """Evaluation report comparing QCNN descriptors with classical baselines."""
    qcnn_triplet_accuracy: float
    qcnn_pos_dist_mean: float
    qcnn_neg_dist_mean: float
    pca_triplet_accuracy: float
    classical_cnn_triplet_accuracy: float
    descriptor_dim: int
    n_triplets: int
    training_runtime_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QCNNFeatureExtractor:
    """
    Quantum Convolutional Neural Network patch feature extractor.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        descriptor_dim: int = 4,
        seed: int = 42,
    ):
        self.num_qubits = num_qubits
        self.descriptor_dim = descriptor_dim
        self.seed = seed
        self.num_weights = 10  # 6 for conv, 4 for pool/ansatz
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.2, size=self.num_weights)

    def _extract_quadrant_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract 4 quadrant angle values in [0, pi] from 4x4 patch."""
        p = preprocess_patch(patch, target_size=4)
        # Quadrant means
        q0 = float(np.mean(p[0:2, 0:2]))
        q1 = float(np.mean(p[0:2, 2:4]))
        q2 = float(np.mean(p[2:4, 0:2]))
        q3 = float(np.mean(p[2:4, 2:4]))
        # Map [0, 1] -> [0, pi]
        return np.array([q0, q1, q2, q3], dtype=np.float64) * math.pi

    def _build_and_simulate_qcnn(
        self,
        angles: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Build and simulate the QCNN circuit using Qiskit statevector.
        Returns L2-normalized descriptor vector.
        """
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, SparsePauliOp

        qc = QuantumCircuit(self.num_qubits)

        # 1. Angle Encoding
        for i in range(self.num_qubits):
            qc.ry(angles[i], i)

        # 2. Quantum Convolution Layer (translationally invariant 2-qubit unitaries)
        w = weights
        # Even pairs
        qc.ry(w[0], 0); qc.ry(w[1], 1); qc.cx(0, 1)
        qc.ry(w[0], 2); qc.ry(w[1], 3); qc.cx(2, 3)
        # Odd pairs (periodic boundary)
        qc.ry(w[2], 1); qc.ry(w[3], 2); qc.cx(1, 2)
        qc.ry(w[4], 3); qc.ry(w[5], 0); qc.cx(3, 0)

        # 3. Quantum Pooling Layer (entangle and trace out qubits 1 and 3)
        qc.cx(0, 1)
        qc.ry(w[6], 0)
        qc.cx(2, 3)
        qc.ry(w[7], 2)

        # 4. Measurement of remaining active qubits (0 and 2)
        # Observables: Z0, X0, Z2, X2
        sv = Statevector.from_instruction(qc)

        op_z0 = SparsePauliOp.from_list([("IIIZ", 1.0)])  # qubit 0 is rightmost in Qiskit
        op_x0 = SparsePauliOp.from_list([("IIIX", 1.0)])
        op_z2 = SparsePauliOp.from_list([("IZII", 1.0)])  # qubit 2
        op_x2 = SparsePauliOp.from_list([("IXII", 1.0)])

        val_z0 = float(sv.expectation_value(op_z0).real)
        val_x0 = float(sv.expectation_value(op_x0).real)
        val_z2 = float(sv.expectation_value(op_z2).real)
        val_x2 = float(sv.expectation_value(op_x2).real)

        desc = np.array([val_z0, val_x0, val_z2, val_x2], dtype=np.float64)
        if self.descriptor_dim > 4:
            op_zz = SparsePauliOp.from_list([("IZIZ", 1.0)])
            op_xx = SparsePauliOp.from_list([("IXIX", 1.0)])
            val_zz = float(sv.expectation_value(op_zz).real)
            val_xx = float(sv.expectation_value(op_xx).real)
            desc = np.concatenate([desc, [val_zz, val_xx]])

        norm = float(np.linalg.norm(desc))
        if norm > 1e-6:
            desc = desc / norm
        return desc[:self.descriptor_dim]

    def extract_descriptor(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract compact L2-normalized descriptor from an input image patch.
        """
        angles = self._extract_quadrant_features(patch)
        return self._build_and_simulate_qcnn(angles, self.weights)

    def extract_batch(self, patches: List[np.ndarray]) -> np.ndarray:
        """Extract descriptors for a list of patches."""
        return np.array([self.extract_descriptor(p) for p in patches], dtype=np.float64)

    def fit_triplet(
        self,
        anchors: List[np.ndarray],
        positives: List[np.ndarray],
        negatives: List[np.ndarray],
        margin: float = 0.2,
        maxiter: int = 30,
    ) -> "QCNNFeatureExtractor":
        """
        Train QCNN feature extractor weights using Triplet Loss.
        L = sum max(0, ||d_a - d_p||^2 - ||d_a - d_n||^2 + margin)
        """
        assert len(anchors) == len(positives) == len(negatives)
        n = len(anchors)
        if n == 0:
            return self

        # Precompute angle features
        a_angles = [self._extract_quadrant_features(p) for p in anchors]
        p_angles = [self._extract_quadrant_features(p) for p in positives]
        n_angles = [self._extract_quadrant_features(p) for p in negatives]

        def triplet_loss(w: np.ndarray) -> float:
            total_loss = 0.0
            for i in range(n):
                d_a = self._build_and_simulate_qcnn(a_angles[i], w)
                d_p = self._build_and_simulate_qcnn(p_angles[i], w)
                d_n = self._build_and_simulate_qcnn(n_angles[i], w)
                dist_pos_sq = float(np.sum((d_a - d_p) ** 2))
                dist_neg_sq = float(np.sum((d_a - d_n) ** 2))
                total_loss += max(0.0, dist_pos_sq - dist_neg_sq + margin)
            return float(total_loss / n)

        res = minimize(triplet_loss, self.weights, method="COBYLA", options={"maxiter": maxiter})
        self.weights = res.x
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Classical Baselines for Comparison
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalPCADescriptor:
    """PCA baseline descriptor on flattened image patches."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, patches: List[np.ndarray]) -> "ClassicalPCADescriptor":
        X = np.array([preprocess_patch(p, 4).ravel() for p in patches], dtype=np.float64)
        self.pca.fit(X)
        self.is_fitted = True
        return self

    def extract(self, patch: np.ndarray) -> np.ndarray:
        X = preprocess_patch(patch, 4).ravel().reshape(1, -1)
        if not self.is_fitted:
            vec = X[0, :self.n_components]
        else:
            vec = self.pca.transform(X)[0]
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-6 else vec


class ClassicalSimpleCNNDescriptor:
    """Simple 2D convolution + pooling classical descriptor baseline."""

    def __init__(self, descriptor_dim: int = 4):
        self.descriptor_dim = descriptor_dim
        # Fixed spatial filters: horizontal edges, vertical edges, diagonal, blur
        self.filters = np.array([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],     # Sobel Y
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],     # Sobel X
            [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],      # Diagonal
            [[1, 1, 1], [1, 2, 1], [1, 1, 1]],        # Low-pass
        ], dtype=np.float64) / 4.0

    def extract(self, patch: np.ndarray) -> np.ndarray:
        p = preprocess_patch(patch, 4)
        p_padded = np.pad(p, 1, mode="edge")
        feats = []
        for f in self.filters:
            # 3x3 conv response at center
            resp = np.sum(p_padded[1:4, 1:4] * f)
            feats.append(float(resp))
        vec = np.array(feats[:self.descriptor_dim], dtype=np.float64)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-6 else vec


def evaluate_qcnn_benchmark(
    train_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    test_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    descriptor_dim: int = 4,
    maxiter: int = 25,
) -> QCNNEvaluationResult:
    """
    Train QCNN and evaluate against Classical PCA and Classical Simple CNN descriptors.
    """
    t0 = time.time()
    anchors_tr = [t[0] for t in train_triplets]
    positives_tr = [t[1] for t in train_triplets]
    negatives_tr = [t[2] for t in train_triplets]

    anchors_te = [t[0] for t in test_triplets]
    positives_te = [t[1] for t in test_triplets]
    negatives_te = [t[2] for t in test_triplets]

    # 1. Train QCNN
    qcnn = QCNNFeatureExtractor(descriptor_dim=descriptor_dim)
    qcnn.fit_triplet(anchors_tr, positives_tr, negatives_tr, maxiter=maxiter)

    # 2. Fit PCA
    pca = ClassicalPCADescriptor(n_components=descriptor_dim)
    pca.fit(anchors_tr + positives_tr + negatives_tr)

    # 3. Simple CNN baseline
    cnn = ClassicalSimpleCNNDescriptor(descriptor_dim=descriptor_dim)

    # Evaluate on test triplets
    q_correct = 0
    pca_correct = 0
    cnn_correct = 0
    q_pos_dists = []
    q_neg_dists = []

    for a, p, n in test_triplets:
        # QCNN
        da = qcnn.extract_descriptor(a)
        dp = qcnn.extract_descriptor(p)
        dn = qcnn.extract_descriptor(n)
        d_pos = float(np.linalg.norm(da - dp))
        d_neg = float(np.linalg.norm(da - dn))
        q_pos_dists.append(d_pos)
        q_neg_dists.append(d_neg)
        if d_pos < d_neg:
            q_correct += 1

        # PCA
        da_pca = pca.extract(a)
        dp_pca = pca.extract(p)
        dn_pca = pca.extract(n)
        if np.linalg.norm(da_pca - dp_pca) < np.linalg.norm(da_pca - dn_pca):
            pca_correct += 1

        # CNN
        da_cnn = cnn.extract(a)
        dp_cnn = cnn.extract(p)
        dn_cnn = cnn.extract(n)
        if np.linalg.norm(da_cnn - dp_cnn) < np.linalg.norm(da_cnn - dn_cnn):
            cnn_correct += 1

    n_te = max(1, len(test_triplets))
    runtime = time.time() - t0

    return QCNNEvaluationResult(
        qcnn_triplet_accuracy=round(float(q_correct / n_te), 4),
        qcnn_pos_dist_mean=round(float(np.mean(q_pos_dists)), 4) if q_pos_dists else 0.0,
        qcnn_neg_dist_mean=round(float(np.mean(q_neg_dists)), 4) if q_neg_dists else 0.0,
        pca_triplet_accuracy=round(float(pca_correct / n_te), 4),
        classical_cnn_triplet_accuracy=round(float(cnn_correct / n_te), 4),
        descriptor_dim=descriptor_dim,
        n_triplets=n_te,
        training_runtime_s=round(runtime, 3),
    )

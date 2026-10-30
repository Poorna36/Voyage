"""
tests/test_quantum_qcnn_features.py
===================================
Tests for Build Spec v2 §3C: QCNN for Small-Patch Feature Extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.qcnn_features import (
    QCNNFeatureExtractor,
    ClassicalPCADescriptor,
    ClassicalSimpleCNNDescriptor,
    preprocess_patch,
    evaluate_qcnn_benchmark,
)


def test_preprocess_patch_shapes_and_normalization():
    p8 = np.random.uniform(10.0, 200.0, size=(8, 8))
    res4 = preprocess_patch(p8, target_size=4)
    assert res4.shape == (4, 4)
    assert np.min(res4) >= 0.0
    assert np.max(res4) <= 1.0


def test_qcnn_descriptor_extraction():
    qcnn = QCNNFeatureExtractor(num_qubits=4, descriptor_dim=4)
    patch = np.random.rand(4, 4)
    desc = qcnn.extract_descriptor(patch)

    assert desc.shape == (4,)
    # Must be L2 normalized
    assert abs(np.linalg.norm(desc) - 1.0) < 1e-4


def test_classical_baselines_extraction():
    pca = ClassicalPCADescriptor(n_components=4)
    patches = [np.random.rand(4, 4) for _ in range(5)]
    pca.fit(patches)
    desc_pca = pca.extract(patches[0])
    assert desc_pca.shape == (4,)
    assert abs(np.linalg.norm(desc_pca) - 1.0) < 1e-4

    cnn = ClassicalSimpleCNNDescriptor(descriptor_dim=4)
    desc_cnn = cnn.extract(patches[0])
    assert desc_cnn.shape == (4,)
    assert abs(np.linalg.norm(desc_cnn) - 1.0) < 1e-4


def test_qcnn_triplet_training_and_benchmark():
    rng = np.random.default_rng(42)

    # Generate synthetic triplets with geo-scene grouping
    # Scene A: bright crater center
    scene_a_base = np.zeros((4, 4))
    scene_a_base[1:3, 1:3] = 1.0

    # Scene B: dark maria patch
    scene_b_base = np.zeros((4, 4)) + 0.1

    # Train triplets from Scene A and B
    train_triplets = []
    for _ in range(6):
        a = scene_a_base + rng.normal(0, 0.05, (4, 4))
        p = scene_a_base + rng.normal(0, 0.05, (4, 4))  # positive
        n = scene_b_base + rng.normal(0, 0.05, (4, 4))  # negative
        train_triplets.append((a, p, n))

    # Test triplets from disjoint Scene C and D
    scene_c_base = np.zeros((4, 4))
    scene_c_base[0, :] = 1.0
    scene_d_base = np.zeros((4, 4))
    scene_d_base[:, 0] = 1.0

    test_triplets = []
    for _ in range(4):
        a = scene_c_base + rng.normal(0, 0.05, (4, 4))
        p = scene_c_base + rng.normal(0, 0.05, (4, 4))
        n = scene_d_base + rng.normal(0, 0.05, (4, 4))
        test_triplets.append((a, p, n))

    res = evaluate_qcnn_benchmark(train_triplets, test_triplets, maxiter=15)

    assert res.n_triplets == 4
    assert res.qcnn_triplet_accuracy >= 0.50
    assert res.pca_triplet_accuracy >= 0.0
    assert res.classical_cnn_triplet_accuracy >= 0.0
    assert res.qcnn_pos_dist_mean >= 0.0
    assert res.qcnn_neg_dist_mean >= 0.0

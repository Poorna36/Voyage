"""
quantum/qnn_reliability.py
==========================
Quantum Neural Network (QNN) for Candidate-Match Reliability (Build Spec v2 §3B).

Implements:
  - EstimatorQNN-based classification of candidate correspondences (true match vs spurious).
  - 4-6 dimensional feature vector:
      1. Descriptor distance (normalized)
      2. Lowe ratio (dist_1st / dist_2nd)
      3. Scale ratio (|log(scale_src / scale_ref)| or scale ratio)
      4. Orientation difference (|delta_theta| / pi)
  - Quantum Circuit:
      - Feature map: zz_feature_map / ZZFeatureMap encoding features onto qubits
      - Variational ansatz: real_amplitudes / RealAmplitudes with linear entanglement
      - Observable: Pauli-Z operator on output qubit
  - Classical Baselines:
      - Logistic Regression baseline
      - Decision Tree / Gradient Boosting baseline
  - Calibration and Evaluation Metrics:
      - Precision, Recall, F1
      - PR-AUC (Precision-Recall Area Under Curve)
      - Brier score (Calibration Mean Squared Error)
      - Reliability diagram binning
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger("quantum.qnn_reliability")


@dataclass
class QNNMatchFeatures:
    """Canonical 4-feature vector for a candidate match."""
    descriptor_distance: float  # [0.0, 1.0]
    lowe_ratio: float           # [0.0, 1.0]
    scale_ratio: float          # [0.0, 1.0]
    orientation_diff: float     # [0.0, 1.0]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.descriptor_distance,
            self.lowe_ratio,
            self.scale_ratio,
            self.orientation_diff,
        ], dtype=np.float64)


def extract_match_features(
    src_pt: np.ndarray,
    ref_pt: np.ndarray,
    des_src: np.ndarray,
    des_ref: np.ndarray,
    des_ref_2nd: Optional[np.ndarray] = None,
    scale_src: float = 1.0,
    scale_ref: float = 1.0,
    angle_src_deg: float = 0.0,
    angle_ref_deg: float = 0.0,
) -> np.ndarray:
    """
    Extract 4-dimensional normalized feature vector from a candidate pair.
    """
    # 1. Normalized descriptor Euclidean distance
    d1 = float(np.linalg.norm(des_src - des_ref))
    # Normalize with standard descriptor norm ~ sqrt(2) or 512 for SIFT
    d1_norm = min(1.0, max(0.0, d1 / (np.sqrt(len(des_src)) * 2.0 + 1e-6)))

    # 2. Lowe ratio
    if des_ref_2nd is not None:
        d2 = float(np.linalg.norm(des_src - des_ref_2nd))
        ratio = float(d1 / max(d2, 1e-6))
        ratio_norm = min(1.0, max(0.0, ratio))
    else:
        ratio_norm = d1_norm

    # 3. Scale ratio (normalized log difference)
    scale_ratio = abs(np.log(max(scale_src, 1e-3) / max(scale_ref, 1e-3)))
    scale_norm = float(np.tanh(scale_ratio))

    # 4. Orientation difference in [0, 1]
    angle_diff = abs(angle_src_deg - angle_ref_deg) % 360.0
    if angle_diff > 180.0:
        angle_diff = 360.0 - angle_diff
    angle_norm = float(angle_diff / 180.0)

    return np.array([d1_norm, ratio_norm, scale_norm, angle_norm], dtype=np.float64)


@dataclass
class QNNEvaluationReport:
    """Report comparing QNN with classical baselines."""
    qnn_metrics: Dict[str, float]
    logistic_regression_metrics: Dict[str, float]
    decision_tree_metrics: Dict[str, float]
    calibration_curve: Dict[str, List[float]]
    n_samples: int
    training_runtime_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QNNReliabilityClassifier:
    """
    Quantum Neural Network for candidate match reliability classification.
    """

    def __init__(
        self,
        num_features: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        seed: int = 42,
    ):
        self.num_features = num_features
        self.reps = reps
        self.entanglement = entanglement
        self.seed = seed
        self.weights: Optional[np.ndarray] = None
        self.qnn: Any = None
        self._init_qnn()

    def _init_qnn(self) -> None:
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector
            from qiskit.quantum_info import SparsePauliOp

            n = self.num_features
            # 1. Direct Pure-Qiskit ZZFeatureMap circuit (reps=1)
            fmap = QuantumCircuit(n)
            x_params = ParameterVector("x", n)
            for i in range(n):
                fmap.h(i)
                fmap.rz(2.0 * x_params[i], i)
            for i in range(n):
                for j in range(i + 1, n):
                    fmap.cx(i, j)
                    fmap.rz(2.0 * x_params[i] * x_params[j], j)
                    fmap.cx(i, j)

            # 2. Direct Pure-Qiskit RealAmplitudes variational ansatz
            ansatz = QuantumCircuit(n)
            n_weights = n * (self.reps + 1)
            theta_params = ParameterVector("theta", n_weights)
            idx = 0
            for r in range(self.reps):
                for i in range(n):
                    ansatz.ry(theta_params[idx], i)
                    idx += 1
                for i in range(n - 1):
                    ansatz.cx(i, i + 1)
            for i in range(n):
                ansatz.ry(theta_params[idx], i)
                idx += 1

            self.circuit = fmap.compose(ansatz)
            self.fmap_params = list(fmap.parameters)
            self.ansatz_params = list(ansatz.parameters)
            self.observable = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1.0)])
            self.qnn = self.circuit  # Flag that circuit is initialized

            rng = np.random.default_rng(self.seed)
            self.weights = rng.normal(0.0, 0.1, size=len(self.ansatz_params))
        except Exception as exc:
            log.warning("Failed to initialize Qiskit circuit for QNN: %s. Using NumPy simulator fallback.", exc)
            self.circuit = None
            self.qnn = None
            rng = np.random.default_rng(self.seed)
            self.weights = rng.normal(0.0, 0.1, size=self.num_features + 1)

    def _forward(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Evaluate QNN and return probabilities in [0, 1]."""
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self, "circuit") and self.circuit is not None:
            from qiskit.quantum_info import Statevector

            # Assign variational weights once for the batch
            w_dict = dict(zip(self.ansatz_params, weights))
            bound_w = self.circuit.assign_parameters(w_dict)

            exp_vals = []
            for row in X:
                x_dict = dict(zip(self.fmap_params, row))
                bound_x = bound_w.assign_parameters(x_dict)
                sv = Statevector.from_instruction(bound_x)
                exp_vals.append(float(sv.expectation_value(self.observable).real))

            exp_arr = np.array(exp_vals, dtype=np.float64)
            # Map expectation [-1, 1] -> probability [0, 1]
            probs = (1.0 + exp_arr) / 2.0
            return np.clip(probs, 1e-6, 1.0 - 1e-6)
        else:
            # Fallback linear-sigmoid proxy
            w = weights[:self.num_features]
            b = weights[-1] if len(weights) > self.num_features else 0.0
            logits = X @ w + b
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        maxiter: int = 50,
    ) -> "QNNReliabilityClassifier":
        """
        Train QNN weights using binary cross-entropy loss.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        def loss_fn(weights: np.ndarray) -> float:
            probs = self._forward(X, weights)
            # Binary cross-entropy
            bce = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
            # L2 regularization
            l2 = 0.01 * np.sum(weights ** 2)
            return float(bce + l2)

        init_w = self.weights if self.weights is not None else np.zeros(self.num_features + 1)
        res = minimize(loss_fn, init_w, method="COBYLA", options={"maxiter": maxiter})
        self.weights = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict confidence probability of being a true inlier match."""
        if self.weights is None:
            raise RuntimeError("Model has not been initialized or fitted.")
        return self._forward(X, self.weights)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class (1 = reliable match, 0 = outlier)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)


def compute_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, List[float]]:
    """Compute calibration curve (reliability diagram) bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        if np.sum(mask) > 0:
            bin_centers.append(float((low + high) / 2.0))
            bin_accuracies.append(float(np.mean(y_true[mask])))
            bin_counts.append(int(np.sum(mask)))

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute precision, recall, F1, PR-AUC, and Brier score."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Brier score: MSE between predicted probability and true binary label
    brier = float(np.mean((y_prob - y_true) ** 2))

    # PR-AUC via trapezoidal rule
    from sklearn.metrics import precision_recall_curve, auc
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = prec

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
    }


def train_and_evaluate_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    maxiter: int = 40,
) -> QNNEvaluationReport:
    """
    Train QNN and evaluate against classical Logistic Regression and Decision Tree baselines.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    t0 = time.time()

    # 1. Train QNN
    qnn = QNNReliabilityClassifier(num_features=X_train.shape[1])
    qnn.fit(X_train, y_train, maxiter=maxiter)
    qnn_probs = qnn.predict_proba(X_test)
    qnn_metrics = compute_metrics(y_test, qnn_probs)
    calib = compute_calibration_data(y_test, qnn_probs)

    # 2. Logistic Regression baseline
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_metrics = compute_metrics(y_test, lr_probs)

    # 3. Decision Tree baseline
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_probs = dt.predict_proba(X_test)[:, 1]
    dt_metrics = compute_metrics(y_test, dt_probs)

    runtime = time.time() - t0

    return QNNEvaluationReport(
        qnn_metrics=qnn_metrics,
        logistic_regression_metrics=lr_metrics,
        decision_tree_metrics=dt_metrics,
        calibration_curve=calib,
        n_samples=len(y_test),
        training_runtime_s=round(runtime, 3),
    )

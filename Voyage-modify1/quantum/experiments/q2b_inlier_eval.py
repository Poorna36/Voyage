import sys
import os
import math
import time
sys.path.insert(0, os.path.abspath("Voyage-modify1"))

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from quantum.benchmark import make_benchmark_pair, ALL_CONDITIONS


def simulate_7qubit_statevector(x: np.ndarray, reps: int = 2) -> np.ndarray:
    """
    7-qubit feature statevector for 7 real features (dim = 128).
    NO constant padding.
    """
    n_qubits = 7
    dim = 1 << n_qubits  # 128

    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    basis_bits = np.zeros((dim, n_qubits), dtype=int)
    for s in range(dim):
        for q in range(n_qubits):
            basis_bits[s, q] = (s >> q) & 1
    spins = 1.0 - 2.0 * basis_bits

    for _ in range(reps):
        # R_z(x_i)
        phase_single = np.sum(spins * (x.reshape(1, n_qubits)), axis=1)
        psi = psi * np.exp(-0.5j * phase_single)

        # Ring of ZZ interactions
        phase_zz = np.zeros(dim, dtype=float)
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            coeff = (np.pi - x[i]) * (np.pi - x[j])
            phase_zz += coeff * (spins[:, i] * spins[:, j])
        psi = psi * np.exp(-1.0j * phase_zz)

    norm = np.linalg.norm(psi)
    return psi / norm if norm > 1e-12 else psi


def compute_precision_at_95_recall(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # Find points where recall >= 0.95
    idx = np.where(recalls >= 0.95)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.max(precisions[idx]))


def extract_benchmark_dataset(max_per_scene: int = 25):
    sift = cv2.SIFT_create(nfeatures=200)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    X_list = []
    y_list = []
    groups_list = []

    scene_stats = {}

    for cond in ALL_CONDITIONS:
        scene_stats[cond] = {"counts": [], "pos_fracs": []}
        for seed in range(10):
            pair = make_benchmark_pair(cond, seed=seed)
            src, ref, H_true = pair["src"], pair["ref"], pair["H_true"]
            H_inv = np.linalg.inv(H_true)

            kp1, des1 = sift.detectAndCompute(src, None)
            kp2, des2 = sift.detectAndCompute(ref, None)
            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            valid = [m for m in matches if len(m) == 2]
            if not valid:
                continue

            src_pts = np.array([kp1[m[0].queryIdx].pt for m in valid], dtype=np.float32)
            ref_pts = np.array([kp2[m[0].trainIdx].pt for m in valid], dtype=np.float32)

            # Symmetric reprojection error under H_true
            p_fwd = np.hstack([src_pts, np.ones((len(src_pts), 1))]) @ H_true.T
            err_fwd = np.linalg.norm(p_fwd[:, :2] / (p_fwd[:, 2:3] + 1e-12) - ref_pts, axis=1)

            p_bwd = np.hstack([ref_pts, np.ones((len(ref_pts), 1))]) @ H_inv.T
            err_bwd = np.linalg.norm(p_bwd[:, :2] / (p_bwd[:, 2:3] + 1e-12) - src_pts, axis=1)

            err_sym = np.sqrt(0.5 * (err_fwd**2 + err_bwd**2))
            labels = (err_sym <= 3.0).astype(int)

            scene_stats[cond]["counts"].append(len(valid))
            scene_stats[cond]["pos_fracs"].append(float(np.mean(labels)))

            # Extract 7 real match features
            n_m = len(valid)
            feat = np.zeros((n_m, 7), dtype=np.float32)
            for i, m in enumerate(valid):
                m1, m2 = m[0], m[1]
                k1 = kp1[m1.queryIdx]
                k2 = kp2[m1.trainIdx]

                confidence = k1.response
                scale_ratio = k1.size / (k2.size + 1e-6)
                angle_diff = min(abs(k1.angle - k2.angle), 360.0 - abs(k1.angle - k2.angle)) / 180.0
                norm_x = k1.pt[0] / 256.0
                norm_y = k1.pt[1] / 256.0
                disp = np.linalg.norm(np.array(k1.pt) - np.array(k2.pt)) / (256.0 * math.sqrt(2.0))
                lowe_ratio = m1.distance / (m2.distance + 1e-6)

                feat[i] = [confidence, scale_ratio, angle_diff, norm_x, norm_y, disp, lowe_ratio]

            # Subsample per scene if needed to balance dataset
            if max_per_scene is not None and n_m > max_per_scene:
                rng_scene = np.random.default_rng(seed * 100)
                sub_idx = rng_scene.choice(n_m, size=max_per_scene, replace=False)
                feat = feat[sub_idx]
                labels = labels[sub_idx]

            scene_id = f"{cond}_{seed}"
            X_list.append(feat)
            y_list.append(labels)
            groups_list.extend([scene_id] * len(labels))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups_all = np.array(groups_list)

    return X_all, y_all, groups_all, scene_stats


def run_q2b_cross_validation(X, y, groups, n_splits=5, n_repeats=3):
    models = ["Quantum_Kernel_SVM", "RBF_SVM", "Logistic_Regression", "HistGradientBoosting", "Lowe_Ratio_Baseline"]
    results = {m: {"roc_auc": [], "pr_auc": [], "bal_acc": [], "p_at_95r": [], "t_train": [], "t_pred": []} for m in models}

    unique_groups = np.unique(groups)
    rng_perm = np.random.default_rng(42)

    # Track paired scene scores for bootstrap CI
    paired_diffs_roc = []
    paired_diffs_pr = []

    total_folds = n_splits * n_repeats
    fold_counter = 0

    for rep in range(n_repeats):
        shuffled_groups = rng_perm.permutation(unique_groups)
        group_to_shuff = {g: i for i, g in enumerate(shuffled_groups)}
        shuff_group_ids = np.array([group_to_shuff[g] for g in groups])

        gkf = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in gkf.split(X, y, groups=shuff_group_ids):
            fold_counter += 1
            X_train_raw, y_train = X[train_idx], y[train_idx]
            X_test_raw, y_test = X[test_idx], y[test_idx]

            # Fit scaler on TRAIN ONLY
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # Map to angles in [0, 2*pi] for quantum kernel: sigmoid
            X_train_q = 2.0 * np.pi / (1.0 + np.exp(-X_train))
            X_test_q = 2.0 * np.pi / (1.0 + np.exp(-X_test))

            # 1. Quantum Kernel SVM
            t0 = time.time()
            psi_train = np.array([simulate_7qubit_statevector(x) for x in X_train_q])
            K_train = np.abs(psi_train @ psi_train.conj().T) ** 2
            q_svm = SVC(kernel="precomputed", probability=True, C=1.0)
            q_svm.fit(K_train, y_train)
            t_train_q = time.time() - t0

            t0 = time.time()
            psi_test = np.array([simulate_7qubit_statevector(x) for x in X_test_q])
            K_test = np.abs(psi_test @ psi_train.conj().T) ** 2
            y_score_q = q_svm.predict_proba(K_test)[:, 1]
            y_pred_q = q_svm.predict(K_test)
            t_pred_q = time.time() - t0

            # 2. RBF-SVM
            t0 = time.time()
            rbf_svm = SVC(kernel="rbf", probability=True, C=1.0)
            rbf_svm.fit(X_train, y_train)
            t_train_rbf = time.time() - t0

            t0 = time.time()
            y_score_rbf = rbf_svm.predict_proba(X_test)[:, 1]
            y_pred_rbf = rbf_svm.predict(X_test)
            t_pred_rbf = time.time() - t0

            # 3. Logistic Regression
            t0 = time.time()
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            t_train_lr = time.time() - t0

            t0 = time.time()
            y_score_lr = lr.predict_proba(X_test)[:, 1]
            y_pred_lr = lr.predict(X_test)
            t_pred_lr = time.time() - t0

            # 4. HistGradientBoosting
            t0 = time.time()
            hgb = HistGradientBoostingClassifier(random_state=42)
            hgb.fit(X_train, y_train)
            t_train_hgb = time.time() - t0

            t0 = time.time()
            y_score_hgb = hgb.predict_proba(X_test)[:, 1]
            y_pred_hgb = hgb.predict(X_test)
            t_pred_hgb = time.time() - t0

            # 5. Lowe ratio threshold baseline (feature index 6 is raw Lowe ratio)
            # High score for low ratio: score = 1.0 - lowe_ratio
            lowe_test = X_test_raw[:, 6]
            y_score_lowe = 1.0 - np.clip(lowe_test, 0.0, 1.0)
            y_pred_lowe = (lowe_test <= 0.80).astype(int)

            scores_map = {
                "Quantum_Kernel_SVM": (y_pred_q, y_score_q, t_train_q, t_pred_q),
                "RBF_SVM": (y_pred_rbf, y_score_rbf, t_train_rbf, t_pred_rbf),
                "Logistic_Regression": (y_pred_lr, y_score_lr, t_train_lr, t_pred_lr),
                "HistGradientBoosting": (y_pred_hgb, y_score_hgb, t_train_hgb, t_pred_hgb),
                "Lowe_Ratio_Baseline": (y_pred_lowe, y_score_lowe, 0.0, 0.0),
            }

            for m in models:
                yp, ys, tt, tp = scores_map[m]
                roc = roc_auc_score(y_test, ys)
                pr = average_precision_score(y_test, ys)
                bacc = balanced_accuracy_score(y_test, yp)
                p95 = compute_precision_at_95_recall(y_test, ys)

                results[m]["roc_auc"].append(roc)
                results[m]["pr_auc"].append(pr)
                results[m]["bal_acc"].append(bacc)
                results[m]["p_at_95r"].append(p95)
                results[m]["t_train"].append(tt)
                results[m]["t_pred"].append(tp)

            # Paired difference vs RBF-SVM for this fold
            paired_diffs_roc.append(roc_auc_score(y_test, y_score_q) - roc_auc_score(y_test, y_score_rbf))
            paired_diffs_pr.append(average_precision_score(y_test, y_score_q) - average_precision_score(y_test, y_score_rbf))

    # Bootstrap 95% CI on paired differences
    rng_b = np.random.default_rng(42)
    n_boot = 10000
    boot_roc = [np.mean(rng_b.choice(paired_diffs_roc, size=len(paired_diffs_roc), replace=True)) for _ in range(n_boot)]
    boot_pr = [np.mean(rng_b.choice(paired_diffs_pr, size=len(paired_diffs_pr), replace=True)) for _ in range(n_boot)]

    ci_roc = np.percentile(boot_roc, [2.5, 97.5])
    ci_pr = np.percentile(boot_pr, [2.5, 97.5])

    return results, (ci_roc, ci_pr, np.mean(paired_diffs_roc), np.mean(paired_diffs_pr))


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 80)
    print("  Q2b INLIER SCORER EVALUATION")
    print("  NOTICE: BENCHMARK-DERIVED (synthetic warps), not real Chandrayaan-2 data")
    print("=" * 80)

    X, y, groups, scene_stats = extract_benchmark_dataset(max_per_scene=20)
    print(f"\nExtracted total samples: {len(y)} across {len(np.unique(groups))} scenes.")
    print(f"Overall positive class balance: {np.mean(y)*100:.2f}% (inliers <= 3px)")

    print("\n--- Per-Condition Raw Candidates & Positive Fraction ---")
    for cond in ALL_CONDITIONS:
        cs = scene_stats[cond]
        print(f"  {cond:>20}: {np.mean(cs['counts']):5.1f} ± {np.std(cs['counts']):4.1f} candidates | {np.mean(cs['pos_fracs'])*100:5.1f}% positive")

    print("\nRunning 5-fold 3-repeat GroupKFold across models...")
    res, (ci_roc, ci_pr, diff_roc, diff_pr) = run_q2b_cross_validation(X, y, groups, n_splits=5, n_repeats=3)

    print("\n" + "=" * 115)
    print(f"{'Model':<24} | {'ROC-AUC':<15} | {'PR-AUC':<15} | {'Bal Acc':<15} | {'Prec @ 95% Rec':<15} | {'Train(s)':<9} | {'Pred(s)':<8}")
    print("-" * 115)
    for m in res:
        r = res[m]
        print(f"{m:<24} | {np.mean(r['roc_auc']):.4f} ± {np.std(r['roc_auc']):.4f} | {np.mean(r['pr_auc']):.4f} ± {np.std(r['pr_auc']):.4f} | {np.mean(r['bal_acc']):.4f} ± {np.std(r['bal_acc']):.4f} | {np.mean(r['p_at_95r']):.4f} ± {np.std(r['p_at_95r']):.4f} | {np.mean(r['t_train']):.3f}s   | {np.mean(r['t_pred']):.3f}s")
    print("=" * 115)

    print(f"\n--- Paired Difference vs RBF-SVM (Quantum - RBF) ---")
    print(f"  Delta ROC-AUC: {diff_roc:+.4f} | 95% Bootstrap CI: [{ci_roc[0]:+.4f}, {ci_roc[1]:+.4f}]")
    print(f"  Delta PR-AUC:  {diff_pr:+.4f} | 95% Bootstrap CI: [{ci_pr[0]:+.4f}, {ci_pr[1]:+.4f}]")
    if ci_roc[0] <= 0 <= ci_roc[1] and ci_pr[0] <= 0 <= ci_pr[1]:
        print("  VERDICT: no measurable difference (95% CI includes 0)")
    else:
        print("  VERDICT: statistically significant difference")

    print(f"\nCompleted in {time.time()-t_start:.2f}s")

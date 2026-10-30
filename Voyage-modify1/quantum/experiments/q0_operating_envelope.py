import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import cv2
import time

def build_qubo(s_xy, r_xy, conf, gsd_ratio=1.0, lambda_pen=5.0, pair_w=1.0):
    M = len(s_xy)
    Q = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        Q[i, i] = -float(conf[i])
    for i in range(M):
        for j in range(i + 1, M):
            d_s = float(np.linalg.norm(s_xy[i] - s_xy[j]))
            d_r = float(np.linalg.norm(r_xy[i] - r_xy[j]))
            if d_s < 2.0 or d_r < 2.0:
                Q[i, j] += lambda_pen
            else:
                if d_s > 5.0 and d_r > 5.0:
                    ratio = d_s / (d_r * gsd_ratio + 1e-6)
                    if 0.70 <= ratio <= 1.40:
                        Q[i, j] -= pair_w * float(np.exp(-((ratio - 1.0) ** 2) / 0.10))
                    else:
                        Q[i, j] += pair_w * 0.50
    return Q

def solve_qubo_fast_sa(Q, seed=42, n_reads=5, n_sweeps=300):
    M = Q.shape[0]
    Q_sym = (Q + Q.T) / 2.0
    h = np.diag(Q_sym).copy()
    rng = np.random.default_rng(seed)
    best_x = None
    best_e = np.inf
    T_schedule = np.geomspace(5.0, 0.01, n_sweeps)
    for _ in range(n_reads):
        x = rng.integers(0, 2, size=M, dtype=np.int32)
        cur_e = float(x @ Q_sym @ x)
        read_best_x = x.copy()
        read_best_e = cur_e
        for T in T_schedule:
            for i in rng.permutation(M):
                delta = 1 - 2 * x[i]
                dE = (h[i] + 2.0 * (Q_sym[i] @ x - h[i] * x[i])) * delta
                if dE <= 0.0 or rng.random() < np.exp(-dE / T):
                    x[i] = 1 - x[i]
                    cur_e += dE
                    if cur_e < read_best_e:
                        read_best_e = cur_e
                        read_best_x = x.copy()
        if read_best_e < best_e:
            best_e = read_best_e
            best_x = read_best_x.copy()
    return best_x, best_e

canvases = [100, 200, 400, 800]
near_miss_s = [5, 10, 20, 30, 50, 100]
outlier_types = [("near-miss", s) for s in near_miss_s] + [("gross", 0)]

results = []

print("Running 4 canvases x 7 outlier models x 10 seeds = 280 runs...")
t_start = time.time()

for C in canvases:
    for model_name, s_val in outlier_types:
        model_label = f"near(s={s_val})" if model_name == "near-miss" else "gross"
        
        qubo_precs, qubo_recs = [], []
        magsac_precs, magsac_recs = [], []
        keep_all_is_gs_count = 0
        keep_all_better_than_inliers_count = 0
        
        for seed_idx in range(10):
            seed = seed_idx * 1000 + 42
            rng = np.random.default_rng(seed)
            N = 24
            
            # 18 inliers, 6 outliers
            s_xy = rng.uniform(0.1 * C, 0.9 * C, size=(N, 2)).astype(np.float32)
            shift = np.array([12.0, -8.0], dtype=np.float32)
            r_xy = s_xy + shift
            
            if model_name == "near-miss":
                r_xy[-6:] += rng.normal(0, float(s_val), size=(6, 2)).astype(np.float32)
            else: # gross
                r_xy[-6:] = rng.uniform(0.1 * C, 0.9 * C, size=(6, 2)).astype(np.float32)
                
            conf = rng.uniform(0.4, 0.95, size=N).astype(np.float32)
            
            # Ground truth inliers are 0..17
            gt_mask = np.zeros(N, dtype=bool)
            gt_mask[:18] = True
            
            # Build QUBO
            Q = build_qubo(s_xy, r_xy, conf)
            
            # Check energies
            x_all = np.ones(N)
            x_inl = np.zeros(N)
            x_inl[:18] = 1.0
            
            e_all = float(x_all @ Q @ x_all)
            e_inl = float(x_inl @ Q @ x_inl)
            
            if e_all < e_inl:
                keep_all_better_than_inliers_count += 1
                
            # QUBO solve
            sol_qubo, e_sol = solve_qubo_fast_sa(Q, seed=seed)
            if np.all(sol_qubo == 1):
                keep_all_is_gs_count += 1
                
            n_kept_q = int(np.sum(sol_qubo))
            tp_q = int(np.sum(sol_qubo[:18]))
            p_q = tp_q / n_kept_q if n_kept_q > 0 else 0.0
            r_q = tp_q / 18.0
            qubo_precs.append(p_q)
            qubo_recs.append(r_q)
            
            # MAGSAC++ solve
            flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
            _, mask = cv2.findHomography(s_xy, r_xy, flag, 3.0)
            if mask is not None:
                m_vec = mask.ravel()
                n_kept_m = int(np.sum(m_vec == 1))
                tp_m = int(np.sum(m_vec[:18] == 1))
                p_m = tp_m / n_kept_m if n_kept_m > 0 else 0.0
                r_m = tp_m / 18.0
            else:
                p_m, r_m = 0.0, 0.0
            magsac_precs.append(p_m)
            magsac_recs.append(r_m)
            
        rec = {
            "canvas": C,
            "outlier_model": model_label,
            "qubo_prec": np.mean(qubo_precs),
            "qubo_rec": np.mean(qubo_recs),
            "magsac_prec": np.mean(magsac_precs),
            "magsac_rec": np.mean(magsac_recs),
            "keep_all_e_lt_inliers": keep_all_better_than_inliers_count,
            "keep_all_is_solution": keep_all_is_gs_count,
        }
        results.append(rec)

print(f"Sweep finished in {time.time()-t_start:.2f}s\n")

print(f"{'Canvas':>6} | {'Outlier Model':>13} | {'QUBO Prec':>9} | {'QUBO Rec':>8} | {'MAGSAC Prec':>11} | {'MAGSAC Rec':>10} | {'E(all)<E(inl)':>13} | {'Keep-All Opt':>12}")
print("-" * 105)
for r in results:
    print(f"{r['canvas']:6d} | {r['outlier_model']:>13} | {r['qubo_prec']*100:8.1f}% | {r['qubo_rec']*100:7.1f}% | {r['magsac_prec']*100:10.1f}% | {r['magsac_rec']*100:9.1f}% | {r['keep_all_e_lt_inliers']:10d}/10 | {r['keep_all_is_solution']:9d}/10")

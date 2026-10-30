import sys
import os
import math
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import time
from quantum.selection_qaoa import build_cell_qubo, solve_cell_exact
from quantum.qaoa import QAOASolver

def run_control_study(K_values=[4, 8, 12], n_trials=100, seed_base=1000):
    k_map = {4: 2, 8: 3, 12: 5}
    results = {}
    
    for K in K_values:
        k = k_map[K]
        print(f"\n--- Running K={K}, k={k} ({n_trials} trials) ---")
        
        pgs_list = []
        qaoa_hit_list = []
        unif_rand_hit_list = []
        k_rand_hit_list = []
        grid_cap_hit_list = []
        
        gap_qaoa_list = []
        gap_unif_list = []
        gap_krand_list = []
        gap_gridcap_list = []
        
        t_exact_list = []
        t_qaoa_list = []
        
        for trial in range(n_trials):
            trial_seed = seed_base + trial * 37 + K * 101
            rng = np.random.default_rng(trial_seed)
            
            coords = rng.uniform(0.0, 100.0, size=(K, 2))
            conf = rng.uniform(0.4, 0.95, size=K)
            
            Q, _ = build_cell_qubo(coords, conf, k=k, cell_size=100.0)
            Q_sym = (Q + Q.T) / 2.0
            
            # Exact brute force
            t0 = time.time()
            exact_x, min_E, max_E, mean_E = solve_cell_exact(Q, k)
            t_exact_list.append(time.time() - t0)
            
            # QAOA
            t0 = time.time()
            solver = QAOASolver(p_layers=2, n_optimizer_restarts=3, n_shots=1024, seed=trial_seed)
            res_qaoa = solver.solve(Q)
            t_qaoa_list.append(time.time() - t0)
            
            pgs = res_qaoa["ground_state_probability"]
            pgs_list.append(pgs)
            
            e_qaoa = res_qaoa["energy"]
            is_qaoa_hit = bool(np.isclose(e_qaoa, min_E, atol=1e-5))
            qaoa_hit_list.append(is_qaoa_hit)
            gap_qaoa_list.append(max(0.0, e_qaoa - min_E))
            
            # Uniform random best-of-1024
            rng_samples = np.random.default_rng(trial_seed)
            unif_bits = rng_samples.integers(0, 2, size=(1024, K))
            unif_energies = np.sum((unif_bits @ Q_sym) * unif_bits, axis=1)
            e_unif = float(np.min(unif_energies))
            is_unif_hit = bool(np.isclose(e_unif, min_E, atol=1e-5))
            unif_rand_hit_list.append(is_unif_hit)
            gap_unif_list.append(max(0.0, e_unif - min_E))
            
            # Random weight-k best-of-1024
            k_bits = np.zeros((1024, K))
            for i in range(1024):
                idx = rng_samples.choice(K, size=k, replace=False)
                k_bits[i, idx] = 1.0
            k_energies = np.sum((k_bits @ Q_sym) * k_bits, axis=1)
            e_krand = float(np.min(k_energies))
            is_krand_hit = bool(np.isclose(e_krand, min_E, atol=1e-5))
            k_rand_hit_list.append(is_krand_hit)
            gap_krand_list.append(max(0.0, e_krand - min_E))
            
            # Classical grid_cap top-k (highest confidence top k)
            top_k_idx = np.argsort(-conf)[:k]
            x_topk = np.zeros(K)
            x_topk[top_k_idx] = 1.0
            e_gridcap = float(x_topk @ Q_sym @ x_topk)
            is_gridcap_hit = bool(np.isclose(e_gridcap, min_E, atol=1e-5))
            grid_cap_hit_list.append(is_gridcap_hit)
            gap_gridcap_list.append(max(0.0, e_gridcap - min_E))
            
            if (trial + 1) % 25 == 0:
                print(f"  Trial {trial+1}/{n_trials} complete...")
                
        p_unif = 1.0 / (2 ** K)
        c_k = math.comb(K, k)
        p_comb = 1.0 / c_k
        mean_pgs = float(np.mean(pgs_list))
        
        results[K] = {
            "K": K,
            "k": k,
            "mean_pgs": mean_pgs,
            "ratio_to_unif": mean_pgs / p_unif,
            "ratio_to_comb": mean_pgs / p_comb,
            "qaoa_hit_rate": float(np.mean(qaoa_hit_list)),
            "unif_rand_hit_rate": float(np.mean(unif_rand_hit_list)),
            "k_rand_hit_rate": float(np.mean(k_rand_hit_list)),
            "grid_cap_hit_rate": float(np.mean(grid_cap_hit_list)),
            "mean_gap_qaoa": float(np.mean(gap_qaoa_list)),
            "mean_gap_unif": float(np.mean(gap_unif_list)),
            "mean_gap_krand": float(np.mean(gap_krand_list)),
            "mean_gap_gridcap": float(np.mean(gap_gridcap_list)),
            "runtime_exact_ms": float(np.mean(t_exact_list)) * 1000,
            "runtime_qaoa_ms": float(np.mean(t_qaoa_list)) * 1000,
        }
        
    return results

if __name__ == "__main__":
    t0 = time.time()
    res = run_control_study(K_values=[4, 8, 12], n_trials=100)
    print(f"\nCompleted in {time.time()-t0:.2f}s\n")
    print(f"{'K':>2} | {'k':>2} | {'P(GS)':>8} | {'P(GS)/(1/2^K)':>14} | {'P(GS)/(1/C)':>12} | {'QAOA Hit':>9} | {'Unif Hit':>9} | {'k-Rand Hit':>10} | {'Top-k Hit':>9} | {'Gap QAOA':>9} | {'Gap Unif':>9} | {'Exact ms':>9} | {'QAOA ms':>8}")
    print("-" * 145)
    for K, r in res.items():
        print(f"{r['K']:2d} | {r['k']:2d} | {r['mean_pgs']*100:7.3f}% | {r['ratio_to_unif']:14.2f}x | {r['ratio_to_comb']:12.2f}x | {r['qaoa_hit_rate']*100:8.1f}% | {r['unif_rand_hit_rate']*100:8.1f}% | {r['k_rand_hit_rate']*100:9.1f}% | {r['grid_cap_hit_rate']*100:8.1f}% | {r['mean_gap_qaoa']:9.4f} | {r['mean_gap_unif']:9.4f} | {r['runtime_exact_ms']:8.3f}ms | {r['runtime_qaoa_ms']:7.1f}ms")

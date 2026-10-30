import sys
import os
import math
sys.path.insert(0, os.path.abspath("Voyage-modify1"))

import numpy as np
import time
from quantum.selection_qaoa import build_cell_qubo, solve_cell_exact
from quantum.qaoa import QAOASolver

def compute_k12_bootstrap():
    K = 12
    k = 5
    n_trials = 100
    seed_base = 1000
    
    qaoa_hits = []
    wk_hits = []
    unif_hits = []
    
    for trial in range(n_trials):
        trial_seed = seed_base + trial * 37 + K * 101
        rng = np.random.default_rng(trial_seed)
        coords = rng.uniform(0.0, 100.0, size=(K, 2))
        conf = rng.uniform(0.4, 0.95, size=K)
        
        Q, _ = build_cell_qubo(coords, conf, k=k, cell_size=100.0)
        best_sol, min_E, _, _ = solve_cell_exact(Q, k=k)
        
        solver = QAOASolver(p_layers=2, n_optimizer_restarts=3, n_shots=1024, seed=trial_seed)
        res_qaoa = solver.solve(Q)
        e_qaoa = res_qaoa["energy"]
        qaoa_hits.append(1.0 if np.isclose(e_qaoa, min_E, atol=1e-5) else 0.0)
        
        Q_sym = 0.5 * (Q + Q.T)
        rng_samples = np.random.default_rng(trial_seed)
        
        # Uniform
        unif_bits = rng_samples.integers(0, 2, size=(1024, K))
        unif_energies = np.sum((unif_bits @ Q_sym) * unif_bits, axis=1)
        unif_hits.append(1.0 if np.isclose(float(np.min(unif_energies)), min_E, atol=1e-5) else 0.0)
        
        # Weight-k
        k_bits = np.zeros((1024, K))
        for i in range(1024):
            idx = rng_samples.choice(K, size=k, replace=False)
            k_bits[i, idx] = 1.0
        k_energies = np.sum((k_bits @ Q_sym) * k_bits, axis=1)
        wk_hits.append(1.0 if np.isclose(float(np.min(k_energies)), min_E, atol=1e-5) else 0.0)
        
    qaoa_arr = np.array(qaoa_hits)
    wk_arr = np.array(wk_hits)
    unif_arr = np.array(unif_hits)
    
    diff_wk = qaoa_arr - wk_arr       # QAOA - weight_k
    diff_unif = qaoa_arr - unif_arr   # QAOA - uniform
    
    rng_b = np.random.default_rng(42)
    n_boot = 10000
    boot_wk = np.empty(n_boot)
    boot_unif = np.empty(n_boot)
    
    for b in range(n_boot):
        idx = rng_b.choice(n_trials, size=n_trials, replace=True)
        boot_wk[b] = np.mean(diff_wk[idx])
        boot_unif[b] = np.mean(diff_unif[idx])
        
    ci_wk = np.percentile(boot_wk, [2.5, 97.5])
    ci_unif = np.percentile(boot_unif, [2.5, 97.5])
    
    print(f"K=12 QAOA Hit Rate: {np.mean(qaoa_arr)*100:.1f}%")
    print(f"K=12 Weight-k Hit Rate: {np.mean(wk_arr)*100:.1f}%")
    print(f"K=12 Uniform Hit Rate: {np.mean(unif_arr)*100:.1f}%")
    print(f"Paired Diff (QAOA - Weight-k): {np.mean(diff_wk)*100:+.1f}% | 95% Bootstrap CI: [{ci_wk[0]*100:+.1f}%, {ci_wk[1]*100:+.1f}%]")
    print(f"Paired Diff (QAOA - Uniform):  {np.mean(diff_unif)*100:+.1f}% | 95% Bootstrap CI: [{ci_unif[0]*100:+.1f}%, {ci_unif[1]*100:+.1f}%]")

if __name__ == "__main__":
    compute_k12_bootstrap()

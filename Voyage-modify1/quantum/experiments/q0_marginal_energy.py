import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from quantum.service import QuantumService
from quantum.matcher_qubo import QUBOMatcher
from src.matching.base import MatchResult

seed = 42
rng = np.random.default_rng(seed)
N = 24
s_xy = rng.uniform(50, 450, size=(N, 2)).astype(np.float32)
r_xy = s_xy + np.array([12.0, -8.0], dtype=np.float32)
r_xy[-6:] += rng.normal(0, 30.0, size=(6, 2)).astype(np.float32)
conf = rng.uniform(0.4, 0.95, size=N).astype(np.float32)
raw = MatchResult(src_xy=s_xy, ref_xy=r_xy, confidence=conf)

matcher = QUBOMatcher({"solver": "sqa", "lambda_penalty": 5.0})
M = 24
Q = np.zeros((M, M), dtype=np.float64)
for i in range(M):
    Q[i, i] = -float(conf[i])

lambda_pen = float(matcher.lambda_penalty)
pair_w = float(matcher.pairwise_weight)
gsd_ratio = 1.0

for i in range(M):
    for j in range(i + 1, M):
        d_src_ij = float(np.linalg.norm(s_xy[i] - s_xy[j]))
        d_ref_ij = float(np.linalg.norm(r_xy[i] - r_xy[j]))
        if d_src_ij < 2.0 or d_ref_ij < 2.0:
            Q[i, j] += lambda_pen
        else:
            if d_src_ij > 5.0 and d_ref_ij > 5.0:
                ratio = d_src_ij / (d_ref_ij * gsd_ratio + 1e-6)
                if 0.70 <= ratio <= 1.40:
                    Q[i, j] -= pair_w * float(np.exp(-((ratio - 1.0) ** 2) / 0.10))
                else:
                    Q[i, j] += pair_w * 0.50

def energy_of_set(indices, Q_mat):
    x = np.zeros(24, dtype=float)
    x[indices] = 1.0
    return float(x @ Q_mat @ x)

S0 = list(range(18))
E_S0 = energy_of_set(S0, Q)

print(f"Base 18-inlier set energy E(S_0): {E_S0:.6f}\n")
print(f"{'Outlier idx':>11} | {'Unary (Q_mm)':>12} | {'Pair Rewards':>14} | {'Pair Penalties':>14} | {'Marginal Delta E':>16} | {'New Energy':>12}")
print("-" * 90)

for m in range(18, 24):
    unary = Q[m, m]
    rewards = 0.0
    penalties = 0.0
    for i in S0:
        idx1, idx2 = min(i, m), max(i, m)
        val = Q[idx1, idx2]
        if val < 0:
            rewards += val
        elif val > 0:
            penalties += val
    dE = unary + rewards + penalties
    E_new = energy_of_set(S0 + [m], Q)
    print(f"{m:11d} | {unary:12.6f} | {rewards:14.6f} | {penalties:14.6f} | {dE:16.6f} | {E_new:12.6f}")

# Also cumulative
print("\nCumulative addition of outliers to S_0:")
print(f"{'Added Outlier':>13} | {'Cumulative Delta E':>18} | {'Total Energy':>14}")
print("-" * 52)
cum_set = list(S0)
for m in range(18, 24):
    cum_set.append(m)
    E_cum = energy_of_set(cum_set, Q)
    print(f"{m:13d} | {E_cum - E_S0:18.6f} | {E_cum:14.6f}")

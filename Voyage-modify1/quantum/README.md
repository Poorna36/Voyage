# Simulated / Quantum-Inspired QUBO Correspondence Module

## Overview

This directory contains research implementations of **simulated and quantum-inspired correspondence solvers** for keypoint matching on lunar imagery (SIH 2026 PS-26166 context):

- **Simulated Quantum Annealing (SQA):** Path-Integral Monte Carlo (PIMC) simulation running on standard CPU.
- **QAOA Statevector Simulation:** Classical NumPy simulation of the Quantum Approximate Optimization Algorithm for gate-based quantum circuits ($M \le 16$ variables; runs for $M \le 16$, unreliable at $p=2$).
- **Classical Baselines:** Classical Simulated Annealing (SA) on QUBO, Hungarian bipartite matching algorithm, and standard SIFT + Lowe ratio test (0.75) + RANSAC homography.

> [!NOTE]
> All algorithms in this module run entirely on classical CPU using NumPy / OpenCV. No quantum hardware or cloud QPU access is utilized. Benchmark measurements show that quantum-inspired heuristics (SQA) do **not** provide speedup or accuracy improvements over classical baselines on this problem.

---

## Directory Structure

```
quantum/
├── __init__.py               # Package exports
├── qubo_builder.py           # QUBO / Ising Hamiltonian constructor (v1 & v2)
├── annealer.py               # Simulated Quantum Annealing (PIMC, CPU-only)
├── qaoa.py                   # QAOA statevector simulator (NumPy, M <= 16, CPU-only)
├── classical_baseline.py     # Hungarian, Classical SA, and SIFT+RANSAC baselines
├── matcher_qubo.py           # M4 — BaseMatcher-compliant QUBO matcher
├── benchmark.py              # Multi-condition 10-seed benchmarking harness
├── test_quantum_matcher.py   # Automated test suite (23 unit tests)
└── results/                  # Benchmark output JSON reports
    ├── benchmark.json        (legacy benchmark output)
    └── benchmark_hard.json   (multi-condition 10-seed raw measurements)
```

---

## Formulations: `qubo_v1` vs. `qubo_v2`

| Dimension | `qubo_v1` (Original Implementation) | `qubo_v2` (Corrected Formulation) |
|---|---|---|
| **Candidate Pre-filtering** | Top-$K$ affinity threshold only ($U \ge 0.10$). Purity: ~30% true pairs. | Mutual Nearest Neighbour (MNN) + Lowe ratio test ($d_1/d_2 < 0.80$). Purity: **78.9% true pairs**. |
| **Distance Invariance** | Scale-blind ratio $|d_{src}/d_{ref} - 1.0|$. Penalises true matches when images differ in scale. | Scale-corrected ratio $\|d_{ref}/(d_{src} \cdot s_{est}) - 1.0\|$ using keypoint scales or GSD ratio. Invariant across all scales. |
| **Angle Invariance** | Absolute displacement direction difference $\|\psi_{src} - \psi_{ref}\|$. Degrades severely at 45° and 90° rotations. | Relative orientation difference $(\psi_{ref} - \theta_{ref}) - (\psi_{src} - \theta_{src})$. **Strictly invariant to arbitrary 2D rotations**. |
| **One-to-One Penalty $\lambda$** | Fixed $\lambda = 5.0$. Dwarfed by max single-variable gain ($+49.0$), incentivizing duplicate match violations. | Dynamic data-driven $\lambda = \max(15.0, 1.25 \times \max_k G_k)$. **Guarantees zero constraint violations**. |
| **Match Decoding** | Appends all active variables above confidence threshold without deduplication. | Greedy 1-to-1 conflict resolution: sorts by confidence and rejects duplicate keypoint assignments. |

---

## Solvers

### 1. Simulated Quantum Annealing (`solver='sqa'`)
Path-Integral Monte Carlo (PIMC) simulation on CPU:
- $P=8\text{--}16$ Trotter slices (classical replicas)
- Transverse field schedule: $\Gamma(t): 3.0 \to 0.001$
- Temperature schedule: $T(t): 5.0 \to 0.01$ (geometric)
- Evaluates quantum tunnelling heuristics on CPU.

### 2. QAOA Statevector Simulation (`solver='qaoa'`)
Gate-level statevector simulation in NumPy:
- $p=2$ layers of alternating problem $U_P(\gamma)$ and mixer $U_M(\beta)$ unitaries.
- Runs for $M \le 16$, unreliable at $p=2$ (approximation ratio drops, ground-state probability decays to $0.0016$).
- Vectorized state manipulations over $2^M$ computational basis states.
- $\|Q\|_\infty$ normalization and mixed restarts for Nelder-Mead optimization.

### 3. Classical Simulated Annealing (`solver='classical_sa'`)
Single-spin Metropolis simulated annealing on the exact same QUBO matrix $Q$:
- Direct classical optimization baseline using the identical thermal schedule and read count.

> [!NOTE]
> The runtime difference between Classical SA (~0.08–0.19s) and SQA (~0.35–1.2s) partly reflects implementation differences: SQA runs a Python Path-Integral Monte Carlo loop with 8–16 Trotter replicas (requiring 8–16× more spin evaluations per sweep), whereas SA runs a single Metropolis chain. It should not be interpreted as an inherent algorithmic scaling difference.

### 4. Classical Hungarian Algorithm (`solver='classical'`)
Solves the linear assignment problem on negative unary costs $-U$:
- Provably guarantees strict one-to-one matching (0.0% constraint violations).
- Evaluated on full QUBO objective $x^T Q x$ and geometric accuracy.

### 5. Classical Feature Baselines (`sift_ratio_ransac`, `sift_noratio_ransac`)
Standard computer vision pipeline: SIFT detection & description + Lowe ratio test ($0.75$) + RANSAC homography estimation ($3.0$ px threshold), compared with and without Lowe ratio filtering.

---

## Empirical Benchmark Results

All metrics are evaluated across **10 independent seeds** per condition against known ground-truth homography $H_{true}$ ($3.0$ px inlier threshold).

### 1. Control & Illumination Conditions (10 Seeds)

#### Condition: `easy` (Control: Lunar regolith, rotation 8°, scale 1.15, tx=12, ty=-8)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | 50.1 ± 15.6 | 50.1 ± 15.6 | 100.0 ± 0.0% | 54.0 ± 6.4% | 0.21 ± 0.04 | 0.24 ± 0.11 | 0.0 ± 0.0% | 0/10 | 0.020s |
| `sift_noratio_ransac` | 53.8 ± 16.6 | 53.7 ± 16.5 | 99.8 ± 0.5% | 57.9 ± 6.9% | 0.22 ± 0.04 | 0.25 ± 0.11 | 0.0 ± 0.0% | 0/10 | 0.023s |
| `hungarian_v1` | 25.6 ± 1.9 | 15.0 ± 3.1 | 58.2 ± 9.2% | 16.9 ± 2.8% | 7.74 ± 16.75 | 0.54 ± 0.32 | 0.0 ± 0.0% | 0/10 | 0.080s |
| `hungarian_v2` | 14.4 ± 2.9 | 14.1 ± 2.8 | 98.1 ± 3.0% | 15.9 ± 2.7% | 0.23 ± 0.08 | 0.46 ± 0.28 | 0.0 ± 0.0% | 0/10 | 0.044s |
| `classical_sa_v1` | 29.2 ± 2.6 | 16.4 ± 2.5 | 56.3 ± 8.0% | 18.7 ± 3.4% | 2.97 ± 3.78 | 0.56 ± 0.38 | 4.6 ± 2.5% | 0/10 | 0.189s |
| `classical_sa_v2` | 17.3 ± 2.5 | 15.5 ± 2.0 | 90.1 ± 6.9% | 17.8 ± 3.7% | 0.26 ± 0.09 | 0.48 ± 0.31 | 0.0 ± 0.0% | 0/10 | 0.084s |
| `sqa_v1` | 29.2 ± 2.6 | 16.4 ± 2.5 | 56.3 ± 8.0% | 18.7 ± 3.4% | 2.97 ± 3.78 | 0.56 ± 0.38 | 4.6 ± 2.5% | 0/10 | 1.212s |
| `sqa_v2` | 17.3 ± 2.5 | 15.5 ± 2.0 | 90.1 ± 6.9% | 17.8 ± 3.7% | 0.26 ± 0.09 | 0.48 ± 0.31 | 0.0 ± 0.0% | 0/10 | 0.409s |
| `qubo_v1_ransac` | 16.4 ± 2.5 | 16.4 ± 2.5 | 100.0 ± 0.0% | 18.7 ± 3.4% | 0.24 ± 0.07 | 0.40 ± 0.28 | 4.6 ± 2.5% | 0/10 | 1.213s |
| `qubo_v2_ransac` | 15.5 ± 2.0 | 15.5 ± 2.0 | 100.0 ± 0.0% | 17.8 ± 3.7% | 0.23 ± 0.07 | 0.43 ± 0.32 | 0.0 ± 0.0% | 0/10 | 0.408s |

#### Condition: `illumination` (Linear spatial gradients + gamma change γ=0.7 vs γ=1.4)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | 13.5 ± 5.5 | 13.3 ± 5.2 | 99.0 ± 2.0% | 22.8 ± 6.4% | 0.39 ± 0.11 | 0.99 ± 0.96 | 0.0 ± 0.0% | 0/10 | 0.019s |
| `sift_noratio_ransac` | 21.5 ± 6.9 | 19.7 ± 9.0 | 88.4 ± 29.6% | 33.7 ± 13.5% | 10.35 ± 30.65 | 0.69 ± 0.41 | 0.0 ± 0.0% | 1/10 | 0.038s |
| `hungarian_v1` | 23.3 ± 2.2 | 8.8 ± 2.9 | 38.3 ± 14.1% | 14.7 ± 4.4% | 34.02 ± 26.68 | 1.26 ± 0.81 | 0.0 ± 0.0% | 1/10 | 0.063s |
| `hungarian_v2` | 7.9 ± 3.0 | 7.7 ± 2.8 | 97.4 ± 4.4% | 13.4 ± 4.5% | 0.28 ± 0.09 | 0.67 ± 0.45 | 0.0 ± 0.0% | 1/10 | 0.043s |
| `classical_sa_v1` | 26.9 ± 2.8 | 9.8 ± 2.7 | 37.1 ± 12.7% | 16.5 ± 4.5% | 14.47 ± 15.65 | 1.34 ± 0.92 | 3.3 ± 2.3% | 0/10 | 0.187s |
| `classical_sa_v2` | 10.2 ± 3.5 | 8.4 ± 3.1 | 83.2 ± 8.7% | 14.3 ± 4.9% | 0.54 ± 0.72 | 0.72 ± 0.47 | 0.0 ± 0.0% | 0/10 | 0.071s |
| `sqa_v1` | 26.9 ± 2.7 | 9.7 ± 2.6 | 36.8 ± 12.2% | 16.4 ± 4.2% | 16.14 ± 15.93 | 1.48 ± 1.13 | 3.2 ± 2.4% | 0/10 | 1.192s |
| `sqa_v2` | 10.2 ± 3.5 | 8.4 ± 3.1 | 83.2 ± 8.7% | 14.3 ± 4.9% | 0.54 ± 0.72 | 0.72 ± 0.47 | 0.0 ± 0.0% | 0/10 | 0.312s |
| `qubo_v1_ransac` | 9.8 ± 2.7 | 9.7 ± 2.6 | 99.0 ± 3.1% | 16.4 ± 4.2% | 0.26 ± 0.09 | 0.53 ± 0.23 | 3.2 ± 2.4% | 0/10 | 1.205s |
| `qubo_v2_ransac` | 8.4 ± 3.1 | 8.4 ± 3.1 | 100.0 ± 0.0% | 14.3 ± 4.9% | 0.25 ± 0.08 | 0.72 ± 0.47 | 0.0 ± 0.0% | 0/10 | 0.314s |

---

### 2. Geometry & Scale Conditions (10 Seeds)

#### Condition: `rotation_45` (45° in-plane rotation)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | 51.5 ± 8.2 | 51.5 ± 8.2 | 100.0 ± 0.0% | 61.6 ± 28.9% | 0.34 ± 0.05 | 0.27 ± 0.07 | 0.0 ± 0.0% | 0/10 | 0.021s |
| `sift_noratio_ransac` | 54.4 ± 8.7 | 54.3 ± 8.7 | 99.8 ± 0.4% | 65.3 ± 30.6% | 0.34 ± 0.05 | 0.28 ± 0.08 | 0.0 ± 0.0% | 0/10 | 0.026s |
| `hungarian_v1` | 24.3 ± 2.5 | 9.4 ± 3.8 | 39.0 ± 15.6% | 10.9 ± 7.9% | 51.68 ± 40.56 | 0.74 ± 0.48 | 0.0 ± 0.0% | 1/10 | 0.068s |
| `hungarian_v2` | 9.6 ± 3.8 | 8.5 ± 4.0 | 87.1 ± 10.7% | 10.8 ± 8.4% | 0.45 ± 0.20 | 2.87 ± 4.66 | 0.0 ± 0.0% | 0/10 | 0.045s |
| `classical_sa_v1` | 27.5 ± 3.5 | 10.5 ± 4.3 | 38.0 ± 13.9% | 12.3 ± 6.7% | 30.72 ± 26.57 | 0.67 ± 0.44 | 3.0 ± 1.7% | 2/10 | 0.187s |
| `classical_sa_v2` | 20.0 ± 3.6 | 17.8 ± 4.0 | **88.4 ± 6.4%** | 21.4 ± 10.5% | 0.39 ± 0.08 | 0.53 ± 0.29 | **0.0 ± 0.0%** | **0/10** | 0.086s |
| `sqa_v1` | 27.3 ± 3.6 | 10.4 ± 4.5 | **37.7 ± 14.8%** | 12.2 ± 6.9% | 33.43 ± 31.75 | 0.69 ± 0.47 | 2.4 ± 2.0% | **2/10** | 1.187s |
| `sqa_v2` | 20.0 ± 3.6 | 17.8 ± 4.0 | **88.4 ± 6.4%** | 21.4 ± 10.5% | 0.39 ± 0.08 | 0.53 ± 0.29 | **0.0 ± 0.0%** | **0/10** | 0.452s |
| `qubo_v1_ransac` | 11.0 ± 3.7 | 9.9 ± 5.4 | 79.2 ± 39.7% | 11.6 ± 7.7% | 20.83 ± 42.93 | 0.56 ± 0.25 | 2.4 ± 2.0% | 2/10 | 1.201s |
| `qubo_v2_ransac` | 17.9 ± 4.0 | 17.8 ± 4.0 | **99.4 ± 1.7%** | 21.4 ± 10.5% | 0.36 ± 0.06 | 0.47 ± 0.16 | **0.0 ± 0.0%** | **0/10** | 0.451s |

#### Condition: `scale_0.5` (Scale 0.5 downsampling)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | 17.5 ± 2.4 | 17.5 ± 2.4 | 100.0 ± 0.0% | 71.0 ± 5.6% | 0.27 ± 0.04 | 0.47 ± 0.18 | 0.0 ± 0.0% | 0/10 | 0.021s |
| `sift_noratio_ransac` | 18.0 ± 2.2 | 18.0 ± 2.2 | 100.0 ± 0.0% | 73.1 ± 6.0% | 0.28 ± 0.04 | 0.48 ± 0.17 | 0.0 ± 0.0% | 0/10 | 0.023s |
| `hungarian_v1` | 14.5 ± 1.4 | 8.8 ± 2.1 | 60.5 ± 11.6% | 35.8 ± 7.7% | 17.72 ± 27.65 | 0.70 ± 0.27 | 0.0 ± 0.0% | 0/10 | 0.048s |
| `hungarian_v2` | 7.9 ± 1.6 | 7.7 ± 1.6 | 97.5 ± 5.0% | 31.4 ± 5.9% | 0.30 ± 0.08 | 0.69 ± 0.29 | 0.0 ± 0.0% | 0/10 | 0.038s |
| `classical_sa_v1` | 16.5 ± 1.8 | 9.9 ± 1.7 | 60.6 ± 10.0% | 40.5 ± 7.4% | 5.37 ± 15.65 | 0.76 ± 0.31 | 3.4 ± 2.9% | 0/10 | 0.144s |
| `classical_sa_v2` | 9.6 ± 1.3 | 8.4 ± 1.3 | 87.7 ± 7.5% | 34.2 ± 5.1% | 0.35 ± 0.13 | 0.75 ± 0.28 | 0.0 ± 0.0% | 0/10 | 0.063s |
| `sqa_v1` | 16.5 ± 1.8 | 9.9 ± 1.7 | 60.6 ± 10.0% | 40.5 ± 7.4% | 5.37 ± 15.65 | 0.76 ± 0.31 | 3.4 ± 2.9% | 0/10 | 0.932s |
| `sqa_v2` | 9.6 ± 1.3 | 8.4 ± 1.3 | 87.7 ± 7.5% | 34.2 ± 5.1% | 0.35 ± 0.13 | 0.75 ± 0.28 | 0.0 ± 0.0% | 0/10 | 0.266s |
| `qubo_v1_ransac` | 9.9 ± 1.7 | 9.9 ± 1.7 | 100.0 ± 0.0% | 40.5 ± 7.4% | 0.29 ± 0.04 | 0.66 ± 0.28 | 3.4 ± 2.9% | 0/10 | 0.933s |
| `qubo_v2_ransac` | 8.4 ± 1.3 | 8.4 ± 1.3 | 100.0 ± 0.0% | 34.2 ± 5.1% | 0.30 ± 0.06 | 0.75 ± 0.28 | 0.0 ± 0.0% | 0/10 | 0.265s |

---

### 3. Challenging Lunar Conditions & Real Imagery (10 Seeds)

#### Condition: `noise_25` (Severe sensor noise, σ=25 grey levels)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | 20.5 ± 6.5 | 20.1 ± 6.3 | 98.4 ± 3.6% | 8.2 ± 1.9% | 0.76 ± 0.23 | 1.52 ± 0.90 | 0.0 ± 0.0% | 0/10 | 0.023s |
| `sift_noratio_ransac` | 20.7 ± 11.1 | 17.2 ± 13.6 | 68.1 ± 37.1% | 6.7 ± 5.1% | 39.88 ± 65.61 | 27.31 ± 48.28 | 0.0 ± 0.0% | 3/10 | 0.079s |
| `hungarian_v1` | 23.7 ± 2.6 | 2.8 ± 1.8 | 11.2 ± 6.4% | 1.1 ± 0.7% | 130.96 ± 19.92 | 59.26 ± 71.18 | 0.0 ± 0.0% | 7/10 | 0.070s |
| `hungarian_v2` | 3.7 ± 1.4 | 2.6 ± 1.6 | 71.8 ± 30.9% | 1.0 ± 0.6% | 25.41 ± 39.66 | 3.74 ± 2.10 | 0.0 ± 0.0% | 8/10 | 0.046s |
| `classical_sa_v1` | 27.8 ± 3.0 | 2.9 ± 1.8 | 10.0 ± 5.7% | 1.2 ± 0.7% | 106.22 ± 21.81 | 135.78 ± 186.02 | 2.3 ± 2.2% | 6/10 | 0.191s |
| `classical_sa_v2` | 7.6 ± 2.5 | 2.7 ± 1.6 | 33.6 ± 15.8% | 1.1 ± 0.6% | 67.55 ± 55.74 | 20.43 ± 13.56 | 0.0 ± 0.0% | 7/10 | 0.066s |
| `sqa_v1` | 27.4 ± 2.8 | 2.9 ± 1.8 | 10.2 ± 5.8% | 1.2 ± 0.7% | 103.27 ± 27.23 | 49.95 ± 46.29 | 1.6 ± 1.7% | 6/10 | 1.186s |
| `sqa_v2` | 7.6 ± 2.5 | 2.7 ± 1.6 | 33.6 ± 15.8% | 1.1 ± 0.6% | 67.55 ± 55.74 | 20.43 ± 13.56 | 0.0 ± 0.0% | 7/10 | 0.230s |
| `qubo_v1_ransac` | 5.2 ± 0.4 | 1.5 ± 1.8 | 27.3 ± 31.0% | 0.6 ± 0.7% | 52.13 ± 38.97 | 11.13 ± 0.00 | 1.6 ± 1.7% | 9/10 | 1.239s |
| `qubo_v2_ransac` | 4.3 ± 0.5 | 2.0 ± 1.5 | 44.0 ± 28.5% | 0.8 ± 0.6% | 59.50 ± 63.08 | 22.18 ± 16.33 | 0.0 ± 0.0% | 8/10 | 0.231s |

#### Condition: `repeated_structure` (Grid of ambiguous identical craters)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | 9.4 ± 3.2 | 8.5 ± 4.5 | 84.0 ± 33.2% | 7.4 ± 3.6% | 12.87 ± 25.54 | 0.27 ± 0.10 | 0.0 ± 0.0% | 2/10 | 0.031s |
| `sift_noratio_ransac` | 24.0 ± 5.9 | 23.9 ± 6.0 | 99.5 ± 1.6% | 20.8 ± 3.6% | 0.20 ± 0.05 | 0.22 ± 0.07 | 0.0 ± 0.0% | 0/10 | 0.072s |
| `hungarian_v1` | 6.9 ± 1.8 | 1.4 ± 1.0 | 21.7 ± 17.8% | 1.2 ± 0.9% | 86.09 ± 31.02 | inf | 0.0 ± 0.0% | 10/10 | 0.064s |
| `hungarian_v2` | 6.2 ± 2.2 | 2.5 ± 1.1 | 41.1 ± 13.3% | 2.3 ± 1.1% | 47.67 ± 29.46 | 636.02 ± 482.24 | 0.0 ± 0.0% | 7/10 | 0.042s |
| `classical_sa_v1` | 15.3 ± 2.1 | 3.1 ± 2.0 | 20.5 ± 13.9% | 2.7 ± 1.8% | 73.39 ± 22.54 | 15.06 ± 24.00 | 0.9 ± 1.4% | 6/10 | 0.189s |
| `classical_sa_v2` | 12.9 ± 3.2 | 3.9 ± 1.4 | 30.8 ± 9.6% | 3.5 ± 1.5% | 78.83 ± 12.10 | 480.92 ± 831.57 | 0.0 ± 0.0% | 6/10 | 0.071s |
| `sqa_v1` | 15.3 ± 2.1 | 3.1 ± 2.1 | 20.7 ± 14.4% | 2.7 ± 1.9% | 65.76 ± 10.49 | 206.15 ± 324.88 | 0.9 ± 1.4% | 6/10 | 1.220s |
| `sqa_v2` | 12.9 ± 3.2 | 3.9 ± 1.4 | 30.8 ± 9.6% | 3.5 ± 1.5% | 78.83 ± 12.10 | 480.92 ± 831.57 | 0.0 ± 0.0% | 6/10 | 0.309s |
| `qubo_v1_ransac` | 4.9 ± 0.9 | 1.5 ± 2.3 | 27.5 ± 37.8% | 1.3 ± 2.1% | 65.94 ± 39.46 | 0.28 ± 0.05 | 0.9 ± 1.4% | 8/10 | 1.245s |
| `qubo_v2_ransac` | 4.7 ± 0.9 | 2.4 ± 2.4 | 44.5 ± 38.7% | 2.2 ± 2.3% | 55.89 ± 46.22 | 0.81 ± 0.70 | 0.0 ± 0.0% | 7/10 | 0.318s |

#### Condition: `real_crops` (Actual Lunar Surface: Copernicus / Lunar Highlands crops)
| Method | Matches | Inliers (<3px) | Precision (%) | Recall (%) | Median Err (px) | H_RMSE (px) | Violations (%) | Failures | Runtime (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `sift_ratio_ransac` | **495.0 ± 71.1** | **495.0 ± 71.1** | **100.0 ± 0.0%** | **59.6 ± 2.0%** | **0.24 ± 0.02** | **0.10 ± 0.01** | **0.0 ± 0.0%** | **0/10** | **0.037s** |
| `sift_noratio_ransac` | 530.6 ± 76.1 | 530.5 ± 76.1 | 100.0 ± 0.1% | 63.9 ± 2.1% | 0.25 ± 0.02 | 0.10 ± 0.02 | 0.0 ± 0.0% | 0/10 | 0.040s |
| `hungarian_v1` | 19.2 ± 2.4 | 12.3 ± 2.9 | 63.6 ± 10.6% | 1.5 ± 0.4% | 2.48 ± 5.18 | 0.18 ± 0.05 | 0.0 ± 0.0% | 0/10 | 0.095s |
| `hungarian_v2` | 13.3 ± 3.1 | 12.3 ± 2.6 | 93.0 ± 4.9% | 1.5 ± 0.4% | 0.20 ± 0.04 | 0.18 ± 0.05 | 0.0 ± 0.0% | 0/10 | 0.071s |
| `classical_sa_v1` | 24.8 ± 2.8 | 12.5 ± 2.9 | 50.2 ± 9.5% | 1.5 ± 0.4% | 9.33 ± 13.28 | 0.23 ± 0.15 | 4.6 ± 3.1% | 0/10 | 0.214s |
| `classical_sa_v2` | 14.8 ± 3.1 | 12.5 ± 2.8 | 84.3 ± 6.0% | 1.5 ± 0.4% | 0.21 ± 0.05 | 0.18 ± 0.05 | 0.0 ± 0.0% | 0/10 | 0.106s |
| `sqa_v1` | 25.0 ± 2.7 | 12.5 ± 2.9 | 49.8 ± 9.5% | 1.5 ± 0.4% | 9.93 ± 14.44 | 0.23 ± 0.15 | 5.0 ± 2.8% | 0/10 | 1.231s |
| `sqa_v2` | 14.8 ± 3.1 | 12.5 ± 2.8 | 84.3 ± 6.0% | 1.5 ± 0.4% | 0.21 ± 0.05 | 0.18 ± 0.05 | 0.0 ± 0.0% | 0/10 | 0.390s |
| `qubo_v1_ransac` | 12.5 ± 2.9 | 12.5 ± 2.9 | 100.0 ± 0.0% | 1.5 ± 0.4% | 0.18 ± 0.03 | 0.23 ± 0.15 | 5.0 ± 2.8% | 0/10 | 1.221s |
| `qubo_v2_ransac` | 12.5 ± 2.8 | 12.5 ± 2.8 | 100.0 ± 0.0% | 1.5 ± 0.4% | 0.19 ± 0.03 | 0.19 ± 0.06 | 0.0 ± 0.0% | 0/10 | 0.392s |

---

### 4. QAOA Statevector Scaling ($M \le 16$, Dedicated Table)

Evaluated with statevector simulation ($p=2$ layers, 5 Nelder-Mead restarts, mixed small-$\gamma$ + uniform initialization):

| Qubits ($M$) | Layers ($p$) | Restarts | Approximation Ratio | Ground-State Probability $P(GS)$ | Expected Energy $\langle H_P \rangle$ | Ground-State Energy $E_{GS}$ | Runtime (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **4** | 2 | 5 | 0.8705 | 0.468927 (46.9%) | -1.640 | -1.884 | 0.077s | Functional |
| **6** | 2 | 5 | 0.2033 | 0.119053 (11.9%) | -0.691 | -3.398 | 0.110s | Unreliable |
| **8** | 2 | 5 | 0.6056 | 0.074955 (7.5%) | -7.008 | -11.572 | 0.174s | Moderately stable |
| **10** | 2 | 5 | 0.4757 | 0.028780 (2.9%) | -4.256 | -8.945 | 0.307s | Degraded |
| **12** | 2 | 5 | 0.3812 | 0.021660 (2.2%) | -6.034 | -15.830 | 0.823s | Degraded |
| **14** | 2 | 5 | 0.4066 | 0.023954 (2.4%) | -4.987 | -12.266 | 3.224s | Low fidelity |
| **16** | 2 | 5 | 0.5833 | **0.001625 (0.16%)** | -16.981 | -29.112 | **13.397s** | **Runs for $M \le 16$, unreliable at $p=2$** |

---

## Direct Findings on the Core Questions

### 1. Does the QUBO formulation beat SIFT + RANSAC on any condition tested?
**No.** Across all 12 tested conditions (including severe noise, extreme illumination gradients, scale changes, repeated ambiguous craters, and real Chandrayaan-2 lunar crops), **SIFT + RANSAC decisively outperformed QUBO-based matching in every single condition**:
- On real lunar crops (`real_crops`), SIFT+RANSAC found **495 inliers with 100% precision in 0.037s**, whereas SQA found only 12.5 inliers in 0.390s.
- Under severe noise (`noise_25`), SIFT+RANSAC retained 20.1 inliers with 98.4% precision (0 failures), whereas SQA/QUBO v2 achieved only 2.7 inliers (7/10 failures).
- Under ambiguous repeated crater structures (`repeated_structure`), SIFT+RANSAC achieved 99.5% precision with 0 failures, while QUBO matchers suffered 6–10 failures out of 10 seeds.

### 2. Does SQA beat classical Simulated Annealing on the same QUBO?
**No.** In every single condition and formulation:
- **Solution Quality:** SQA and Classical SA achieved statistically identical inlier counts, precision, and median error (e.g. on `easy` v2, both achieved exactly 15.5 ± 2.0 inliers and 90.1 ± 6.9% precision).
- **Speed:** Classical SA was consistently **3× to 6× faster** than SQA (0.084s vs 0.409s on `easy` v2; 0.189s vs 1.212s on `easy` v1).
- Simulated quantum tunnelling in Path-Integral Monte Carlo did not find better minima than single-chain thermal Metropolis annealing on these correspondence landscapes.

### 3. Under what conditions, if any, is quantum correspondence matching actually useful for Chandrayaan-2 images?
**None at present.** Classical feature-based pipelines (SIFT/RIFT + Lowe ratio test + RANSAC / MAGSAC++) operate directly in continuous pixel coordinate space, exploit projective geometry invariants, and run in 10–40ms. Formulating correspondence as an NP-hard binary quadratic optimization problem throws away geometric structure, introduces severe combinatorial bottleneck constraints, and requires orders of magnitude more computation without any accuracy benefit.

### 4. What would it take to make QUBO-based matching competitive with classical methods in this domain?
To make QUBO competitive:
1. **Physical QPU Hardware with Fast Anneal Times:** Execution would need microsecond-scale physical quantum annealers (e.g., D-Wave Advantage with >5,000 qubits) rather than CPU simulations.
2. **Dense Higher-Order Geometric Encodings (HUBO):** 3-point or 4-point geometric invariants (affine or projective invariants) require cubic or quartic binary terms ($x_i x_j x_k$), which cannot be mapped to 2-local QUBOs without hundreds of auxiliary slack variables that dilute the energy landscape.
3. **Hardware-Native Graph Embedding:** Mapping dense correspondence graph couplings onto physical Chimera/Pegasus QPU topologies currently incurs a severe variable-overhead penalty (embedding minor chains).
4. **Integration as a Coarse Pre-Filter Only:** QUBO could at best serve as a global discrete combinatorial pre-selector before continuous sub-pixel bundle adjustment, but classical algorithms (GMS, SuperGlue, LightGlue) already accomplish this in 5–15ms on standard hardware.

---

## Running the Benchmark & Tests

```bash
# Run full 10-seed multi-condition benchmark
python -m quantum.benchmark --conditions easy,noise_10,noise_25,illumination,partial_overlap,rotation_45,rotation_90,scale_0.5,scale_2.0,repeated_structure,combined,real_crops --n-seeds 10 --qaoa

# Run automated test suite
python -m unittest quantum.test_quantum_matcher -v
```

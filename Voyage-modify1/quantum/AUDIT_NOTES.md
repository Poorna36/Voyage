# Quantum Subsystem Audit Notes

This document tracks all audit findings, discrepancies between specifications (`docs/quantum/03_MODULE_SPECS.md`) and implementation, silent fallbacks, mislabeled algorithms, claim corrections, diagnostic findings, and deferred items across Q0–Q7.

---

## 1. Summary Status vs. Spec (`03_MODULE_SPECS.md`)

| Module | Spec'd Behavior (`03_MODULE_SPECS.md`) | Current State (Code) | Missing / Unfinished Work | Status |
|--------|----------------------------------------|----------------------|--------------------------|--------|
| **Q0 — QUBO Consistency Filter** | Takes upstream `MatchResult`, filters for geometric consistency and 1-to-1 uniqueness using QUBO Ising optimization. Classical twin: MAGSAC++/RANSAC inlier set. | `QUBOMatcher.filter_matches()` in [`matcher_qubo.py`](matcher_qubo.py) is complete. Builds inlined QUBO, dispatches to SQA/QAOA/SA, returns filtered `MatchResult`. Classical twin in `service.py:QuantumService._run_q0_filter` uses OpenCV `USAC_MAGSAC` / `RANSAC`. | **(a)** YAML config keys `unary_weight`, `pairwise_weight`, `max_candidates`, `sqa_sweeps`, `sqa_reads` not read by service runner. **(b) Formulation limit**: Hardcoded ratio window `[0.70, 1.40]` in `matcher_qubo.py:304` only partially removes gross outliers and is blind to pixel-level errors on typical image scales ($s / d_{mean} \lesssim 0.25$). Any window modification moves `benchmark_hard.json` numbers and is deferred to review. **(c)** Not registered in `src/matching/` pipeline registry. | **ACCEPTED LIMITATION** (classical twin **RESOLVED 2026-10-30**, fix **DEFERRED**) |
| **Q1 — QAOA Keypoint-Subset Selection** | Inside each 8×8 grid cell, choose best k=5 matches that are confident and spatially spread. Classical twins: `grid_cap` (top-k) and exact brute-force over 2^K bitstrings measuring approximation ratio. | [`selection_qaoa.py`](selection_qaoa.py): `select_qaoa()` covers cells, `select_cell_qaoa()` runs QAOA, `solve_cell_exact()` brute-forces 2^K. `shots` forwarded to `QAOASolver` and recorded. Additive cell metrics `cells_total`, `cells_trivial`, `cells_qaoa` added. Tolerant fixture comparison against `218c16e` verified. | **(a) HIGH**: No evidence QAOA beats cardinality-aware random sampling ($P(\text{GS})$ is $0.74\times$ of $1/\binom{K}{k}$ at $K=8, 12$; hit rate $63\%$ vs $71\%$; paired diff $-8.0\%$, 95% CI: $[-21.0\%, +4.0\%]$). Exact brute force is 50x faster at $K=12$ (9ms vs 451ms). **(b) HIGH**: Default problem ($N=60, \text{cap}=5$) distributes $\le 4$ candidates per cell, so QAOA is completely bypassed (`cells_trivial=37, cells_qaoa=0`). **(c)** Downstream benefit (inliers/RMSE after Q1 vs `grid_cap`) is untested and belongs to promotion criteria. **(d)** `max_qubits_per_cell` hardcoded to 12; `restarts` defaults to 3 vs YAML 5. | **FINISHED / OPEN ANALYSIS** (shots forwarding, metrics, baseline verification **RESOLVED 2026-10-30**) |
| **Q2 — Quantum-Kernel Classifier** | 8-qubit feature map (H, Rz, ring of ZZ). Q2a: router predicting best matcher from 13-feature MSM vector. Q2b: inlier scorer from 8 match features. Classical twin: RBF-SVM / LightGBM. | [`kernel_classifier.py`](kernel_classifier.py): 8-qubit and 7-qubit feature maps. Labeled `"SYNTHETIC SMOKE TEST - not evidence of real-data performance"` in code, output, and REST response. | **(a) Q2a Router DEFERRED**: Repo has 0 real PairRecords or `meta.json`; benchmark labels are 93.3% SIFT; 9 of 13 real MSM features are constant. Module docstring and service output state: `"router not trained: requires real PairRecords"`. **(b) Q2b Inlier Scorer (Negative Result)**: Evaluated with 5-fold 3-repeat GroupKFold across 12 conditions (2400 samples). Quantum-kernel SVM performs near chance (ROC-AUC $0.5618 \pm 0.0265$, PR-AUC $0.4610 \pm 0.0311$), severely lagging RBF-SVM (ROC-AUC $0.9785$), Logistic Regression ($0.9801$), HistGradientBoosting ($0.9880$), and single-feature Lowe ratio ($0.9406$). Paired diff vs RBF-SVM: $\Delta\text{ROC} = -0.4167$ (95% CI: $[-0.4314, -0.4017]$). | **Q2a DEFERRED / Q2b EVALUATED (NEGATIVE RESULT)** |
| **Q3 — Grover Consensus Search** | Amplitude amplification over 4-match subsets from n≤16 candidates. Oracle: O(√(N/M)) queries vs O(N/M) for uniform random. Classical twin: Random-sampling RANSAC. | [`grover_consensus.py`](grover_consensus.py): Enumerates C(n,4) subsets, simulates Grover iterations via statevector. | **(a)** Oracle builds full classical truth table first (`grover_consensus.py:144-157`), requiring $O(N)$ classical evaluations. **(b)** Config mismatch: YAML uses `max_candidate_matches: 16`; code uses `max_candidates`. **(c)** Classical twin is theoretical query count ratio only. | **OPEN** |
| **Q4 — QFT Phase Correlation** | Gate-level 2D QFT on statevector, amplitude-encoded (2^k × 2^k) tiles, 2D paraboloid sub-pixel fit. Classical twin: `numpy.fft.fft2(..., norm="ortho")`. | [`qft_phase_corr.py`](qft_phase_corr.py): `apply_1d_qft_statevector()` applies H, CPHASE, SWAP. | **(a)** Inverse transform at line 215 uses classical `np.fft.ifft2` rather than gate-level inverse QFT. **(b)** Config mismatch: YAML `tile_size: 32` vs service default `16`. **(c)** `zero_pad` not implemented. | **OPEN** |
| **Q5 — Quantum Image Encoding + Edge Detection** | FRQI (2n+1 qubits), NEQR (2n+8 qubits), amplitude encoding. QHED via cyclical permutation + Hadamard on position qubit. Classical twin: OpenCV/SciPy Sobel. | [`image_encoding.py`](image_encoding.py): `encode_frqi()`, `decode_frqi()`, `encode_neqr()`, `decode_neqr()`. | **(a)** `quantum_edge_detection()` uses classical central difference (`0.5 * abs(f[x+2] - f[x])`) instead of real QHED circuit. **(b)** YAML flags `encoding_type` and `edge_detection` ignored by service runner. **(c)** Amplitude encoding missing. | **OPEN** |
| **Q6 — Backend Abstraction** | Protocol `QuantumBackend` implemented by `numpy_statevector`, `qiskit_aer`, `dwave_neal`, `ibm_runtime`, `dwave_leap`. | [`backends.py`](backends.py): All 5 adapters exist with standard interface and registry. | **(a)** `QiskitAerBackend` falls back to Numpy directly. (Note: Qiskit is NOT INSTALLED; tagged as ASSUMED). **(b)** `IbmRuntimeBackend` and `DwaveLeapBackend` silently fall back to Numpy without reporting `simulated_fallback=True`. **(c)** YAML `backends:` block unread. | **OPEN** |
| **Q7 — Post-Quantum Signing + Verification** | SHA3-256 digest + Hybrid ML-DSA-65 (FIPS 204 / Dilithium) / Ed25519 digital signatures on all exported artifacts and manifests. Classical twin: plain SHA-256. | [`pqc.py`](pqc.py): Signs and verifies with Ed25519. Classical SHA-256 computed. | **(a)** Code is labeled `"ML-DSA-65+Ed25519"` but executes only 64-byte Ed25519 (`dilithium-py` commented out). **(b) Evaluation Complete**: Installed `cryptography 50.0.2` (OpenSSL 4.0.3) natively exposes final FIPS 204 `MLDSA65PrivateKey` (3309-byte signature, 1952-byte public key) with zero external C dependencies. Recommendation ready for approval. | **OPEN (EVALUATION READY)** |

---

## 2. Detailed Audit Log & Discrepancies by File

### Claim Corrections & Provenance
1. **Q1 Test (a) Baseline Provenance (Corrected 2026-10-30)**:
   The initial audit report claimed test (a) matched a saved pre-change baseline. In truth, test (a) initially only ran the post-change code and asserted `"shots" == 1024` without loading any pre-change record. Real baseline JSON fixtures were generated from a git worktree at `218c16e` (`tests/fixtures/q1_baseline_default_218c16e.json` and `q1_baseline_capped_218c16e.json`), and test (a) now loads and asserts tolerant equality (`rtol=1e-9` for floats, exact for ints/keys, excluding `"shots"` and additive `cells_*`). The temporary worktree was pruned and removed.
2. **Qiskit Installation Claim (Corrected 2026-10-30)**:
   The previous status claim that Qiskit is installed was incorrect. An explicit import check had **not** been run prior to making that claim; the assumption was based on seeing import guards (`try: import qiskit ... except ImportError`) pass without throwing unhandled exceptions. Running `python -c "import qiskit"` confirmed that Qiskit is **NOT INSTALLED** in this environment (`Python 3.14.7`).
3. **Problem 1 Sigma Label Clarification (Corrected 2026-10-30)**:
   Problem 1 in the Q0 diagnostic reports was generated by `rng.normal(0, 30.0, size=(6, 2))` in [`quantum/experiments/q0_marginal_energy.py:16`](quantum/experiments/q0_marginal_energy.py). The label "sigma=15" in the previous narrative summary was a typo; the true generating scale was $\sigma = 30.0\text{px}$, matching the $-164.596907$ and $-255.440524$ energies exactly.
4. **MSM Feature List Provenance (Corrected 2026-10-30)**:
   The feature list cited in the initial inventory (`mean_intensity_diff`, `variance_ratio`, `histogram_intersection`, etc.) was an ad-hoc hallucination not present in the codebase. The **real 13 MSM features** are defined verbatim in [`src/selector/features.py:65-79`](src/selector/features.py): `sensor_pair_enc`, `gsd_ratio`, `latitude_abs`, `delta_solar_azimuth`, `terrain_class_enc`, `crater_density`, `masked_fraction`, `overlap_fraction`, `src_texture_contrast`, `ref_texture_contrast`, `src_mean_gradient`, `ref_mean_gradient`, `tile_count`. On synthetic benchmark pairs, 9 of these 13 are degenerate constants.

---

## 3. Q0 Operating Envelope & Pair Term Call Graph

### Pair Term Builder Call Graph & Window Analysis

| Path / Caller | Function Building Pair Term | File & Line | Window / Kernel Code | Is `eps` a Parameter? |
|---|---|---|---|:---:|
| **Service Q0 Filter** (`QuantumService._run_q0_filter`) | Inlined loop in `QUBOMatcher.filter_matches()` | [`quantum/matcher_qubo.py:303-310`](quantum/matcher_qubo.py#L303-L310) | `if 0.70 <= ratio <= 1.40:`<br>`    Q[i, j] -= pair_w * float(np.exp(-((ratio - 1.0) ** 2) / 0.10))`<br>`else:`<br>`    Q[i, j] += pair_w * 0.50` | **NO** (hardcoded literals `0.70`, `1.40`, `0.10`, `0.50`) |
| **Benchmark `qubo_v1`** (`QuantumBenchmark._run_trial`) | `QUBOBuilder._fill_pairwise()` | [`quantum/qubo_builder.py:440-456`](quantum/qubo_builder.py#L440-L456) | `ratio_err = abs(d_src / d_ref - 1.0)`<br>`geo_score = np.exp(-(ratio_err / eps) ** 2) * np.exp(-(angle_err * 3.0) ** 2)`<br>`Q[k, l] -= w * geo_score` | **YES** (`dist_ratio_eps`, default 0.25) |
| **Benchmark `qubo_v2` / `sqa_v2` / `classical_sa_v2`** | `QUBOBuilder._fill_pairwise()` | [`quantum/qubo_builder.py:458-490`](quantum/qubo_builder.py#L458-L490) | `ratio_err = abs(d_ref / (d_src * s_est + 1e-9) - 1.0)`<br>`geo_score = np.exp(-(ratio_err / eps) ** 2) * ...`<br>`Q[k, l] -= w * geo_score` | **YES** (`dist_ratio_eps`, default 0.25) |

- **Recomputed Problem 1 Pair Counts**: On the exact $[0.70, 1.40]$ window the filter actually uses:
  - **Rewards ($Q_{ij} < 0$)**: **262 pairs**
  - **Penalties ($Q_{ij} > 0$)**: **14 pairs**
  - Total pairwise couplings: $262 + 14 = 276 = \binom{24}{2}$.
  - Under `eps=0.25` ($[0.75, 1.25]$), rewards would be 251 and penalties 25.

### Operating Envelope Sweep with $s / d_{mean}$ Column (10 seeds per cell)

Mean pairwise distance: Canvas 100px: $d_{mean} = 52.3\text{px}$; Canvas 200px: $d_{mean} = 104.7\text{px}$; Canvas 400px: $d_{mean} = 209.3\text{px}$; Canvas 800px: $d_{mean} = 418.7\text{px}$.

| Canvas | Outlier Model | $s / d_{mean}$ | QUBO Prec | QUBO Rec | MAGSAC++ Prec | MAGSAC++ Rec | Keep-All Global Opt? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100px | near ($s=5$)   | 0.096 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 100px | near ($s=10$)  | 0.191 | 76.3% | 100% | 97.8% | 100% | 8/10 seeds |
| 100px | near ($s=20$)  | 0.382 | 82.8% | 100% | 96.7% | 100% | 2/10 seeds |
| 100px | near ($s=30$)  | 0.573 | 86.4% | 100% | 96.1% | 100% | 0/10 seeds |
| 100px | near ($s=50$)  | 0.956 | 92.0% | 100% | 96.7% | 100% | 0/10 seeds |
| 100px | near ($s=100$) | 1.911 | 96.4% | 100% | 97.2% | 100% | 0/10 seeds |
| 100px | gross uniform  | —     | 89.3% | 100% | 98.9% | 100% | 0/10 seeds |
| 200px | near ($s=5$)   | 0.048 | 75.0% | 100% | 98.3% | 100% | 10/10 seeds |
| 200px | near ($s=10$)  | 0.096 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 200px | near ($s=20$)  | 0.191 | 76.3% | 100% | 96.7% | 100% | 8/10 seeds |
| 200px | near ($s=30$)  | 0.287 | 78.7% | 100% | 96.7% | 100% | 5/10 seeds |
| 200px | near ($s=50$)  | 0.478 | 85.6% | 100% | 97.2% | 100% | 0/10 seeds |
| 200px | near ($s=100$) | 0.956 | 91.5% | 100% | 97.2% | 100% | 0/10 seeds |
| 200px | gross uniform  | —     | 90.0% | 100% | 99.4% | 100% | 0/10 seeds |
| 400px | near ($s=5$)   | 0.024 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 400px | near ($s=10$)  | 0.048 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 400px | near ($s=20$)  | 0.096 | 75.0% | 100% | 98.3% | 100% | 10/10 seeds |
| 400px | near ($s=30$)  | 0.143 | 75.0% | 100% | 96.7% | 100% | 10/10 seeds |
| 400px | near ($s=50$)  | 0.239 | 77.0% | 100% | 98.3% | 100% | 7/10 seeds |
| 400px | near ($s=100$) | 0.478 | 85.6% | 100% | 97.8% | 100% | 0/10 seeds |
| 400px | gross uniform  | —     | 90.4% | 100% | 98.9% | 100% | 0/10 seeds |
| 800px | near ($s=5$)   | 0.012 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 800px | near ($s=10$)  | 0.024 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 800px | near ($s=20$)  | 0.048 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 800px | near ($s=30$)  | 0.072 | 75.0% | 100% | 98.9% | 100% | 10/10 seeds |
| 800px | near ($s=50$)  | 0.119 | 75.0% | 100% | 98.3% | 100% | 10/10 seeds |
| 800px | near ($s=100$) | 0.239 | 77.0% | 100% | 97.8% | 100% | 7/10 seeds |
| 800px | gross uniform  | —     | 90.2% | 100% | 98.9% | 100% | 0/10 seeds |

- **Transition Confirmation**: Keep-all global optimality fades deterministically with $s / d_{mean}$:
  - $s/d_{mean} \le 0.14$: 10/10 seeds keep-all is global optimum.
  - $s/d_{mean} \approx 0.19$: 8/10 seeds.
  - $s/d_{mean} \approx 0.24$: 7/10 seeds.
  - $s/d_{mean} \approx 0.29$: 5/10 seeds.
  - $s/d_{mean} \approx 0.38$: 2/10 seeds.
  - $s/d_{mean} \ge 0.48$: 0/10 seeds.
- **Definitive Finding**: On a real image with $d_{mean}$ in the hundreds of pixels ($d_{mean} \approx 300\text{px}$), a $3\text{px}$ registration error corresponds to $s / d_{mean} = 0.010$ (a $1\%$ distance ratio deviation). Because $[0.70, 1.40]$ permits a $\pm 30\text{--}40\%$ deviation, **Q0 is completely blind to pixel-level errors and awards maximal pairwise rewards to 3px outliers**.
- **Rewritten Q0 Conclusion**: **Q0 only partially removes gross outliers (~90% precision on gross uniform outliers vs ~99% for MAGSAC++) and is blind to pixel-level errors**.

---

## 4. Q1 Cardinality-Aware Baselines & Control Statistics

### Control Study: $K \in \{4, 8, 12\}$, 100 Seeded Random Cells Each

| Metric | $K=4, k=2$ | $K=8, k=3$ | $K=12, k=5$ |
|---|:---:|:---:|:---:|
| **QAOA $P(\text{GS})$ (Statevector)** | 16.941% | 1.319% | **0.093%** |
| **Uniform Random $P(\text{GS}) = 1/2^K$** | 6.250% | 0.391% | **0.024%** |
| **Ratio to $1/2^K$ (Unconstrained)** | $2.71\times$ | $3.38\times$ | **$3.82\times$** |
| **Subspace Random $1/\binom{K}{k}$ (Cardinality-Aware)** | 16.667% | 1.786% | **0.126%** |
| **Ratio to $1/\binom{K}{k}$** | **$1.02\times$** | **$0.74\times$** | **$0.74\times$** |
| **QAOA Best-of-1024 Hit Rate** | 100.0% | 100.0% | **63.0%** |
| **Weight-$k$ Random Best-of-1024 Hit Rate** | 100.0% | 100.0% | **71.0%** |
| **Uniform Random Best-of-1024 Hit Rate** | 100.0% | 93.0% | **27.0%** |
| **`grid_cap` Top-$k$ Hit Rate** | 36.0% | 6.0% | **0.0%** |
| **Mean Optimality Gap: QAOA** | 0.0000 | 0.0000 | **0.0534** |
| **Mean Optimality Gap: Weight-$k$ Random** | 0.0000 | 0.0000 | **0.0379** |
| **Mean Optimality Gap: Uniform Random** | 0.0000 | 0.0068 | **0.2161** |
| **Mean Optimality Gap: `grid_cap`** | **0.2990** | **0.8452** | **2.1726** |
| **Runtime: Exact Brute Force (ms)** | 0.043 ms | 0.508 ms | **9.001 ms** |
| **Runtime: Simulated QAOA (ms)** | 43.0 ms | 100.6 ms | **451.2 ms** |

### Paired Bootstrap 95% Confidence Intervals at $K=12$ (10,000 resamples)
- **QAOA vs. Cardinality-Aware Weight-$k$ Random**:
  $$\text{Paired Diff} = -8.0\% \quad \text{with 95\% CI: } [-21.0\%, +4.0\%]$$
  The CI spans zero: **there is no evidence that QAOA outperforms simple cardinality-aware random sampling**.
- **QAOA vs. Unconstrained Uniform Random**:
  $$\text{Paired Diff} = +36.0\% \quad \text{with 95\% CI: } [+23.0\%, +49.0\%]$$
  Uniform random is a straw man because feasible solutions have exactly $k$ ones.
- **Speed Comparison**: At $K=12$, exact brute-force search over all $\binom{12}{5} = 792$ combinations runs in **9.0 ms** and achieves a **100% guarantee**, whereas simulated QAOA requires **451.2 ms** (50x slower) to achieve a **63% hit rate**.

---

## 5. Seed & Determinism Follow-ups

- **`cv2.setRNGSeed` is a complete no-op for `USAC_MAGSAC`**: Across 10 different seeds on both $N=60$ and $N=100$ geometries, inlier masks mapped to original indices produced pairwise Jaccard similarities of exactly **1.000000**.
- **Row Permutation Sensitivity**: When outliers are dense ($N=100$, 75% outliers), row permutation alters the hypothesis generation sequence. Inlier counts vary between 22 and 23 (`[22, 22, 22, 23, 22, 22, 22, 22, 22, 23]`), producing a Mean Jaccard of **0.984541** (Min 0.956522).
- **`randomGeneratorState`**: `cv2.UsacParams(randomGeneratorState=...)` directly controls the internal C++ RNG. Inlier counts vary across states, producing Mean Jaccard of **0.542848** (Min 0.315789). **MAGSAC++ is an inherently stochastic baseline; variance over states must be reported in benchmark comparisons**.

---

## 6. Q2 Evaluation (Q2a Router & Q2b Inlier Scorer)

### Q2a Router Status: DEFERRED
- **Absence of Real Data**: The repository contains **0 real PairRecords** and **0 `meta.json` files**.
- **Severe Class Imbalance**: On synthetic benchmark pairs, only SIFT and RIFT2 run (LightGlue not installed); SIFT wins **93.3%** of scenes.
- **Feature Degeneracy**: 9 of the 13 real MSM features are 100% constant across synthetic scenes.
- **Status**: Router training is **DEFERRED**. Service output and docstring explicitly state: `"router not trained: requires real PairRecords"`.

### Q2b Inlier Scorer Benchmark Evaluation
- **Task**: Classify candidate correspondences before ratio test and before RANSAC as inlier/outlier based on 7 real match features with $n_{qubits} = 7$ (no constant padding).
- **Ground Truth**: Symmetric reprojection error under $H_{true} \le 3.0\text{px}$.
- **Dataset**: 2,400 correspondences across 12 conditions $\times$ 10 seeds (120 scenes).
- **Validation**: 5-fold, 3-repeat GroupKFold grouped by scene (zero scene leakage). Scaler fit on train folds only.

| Model | ROC-AUC | PR-AUC | Balanced Accuracy | Precision @ 95% Recall | Train Time | Predict Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.9880 ± 0.0038** | **0.9831 ± 0.0057** | **0.9469 ± 0.0109** | **0.9156 ± 0.0450** | 0.503s | 0.010s |
| **Logistic Regression**  | 0.9801 ± 0.0050 | 0.9706 ± 0.0091 | 0.9247 ± 0.0121 | 0.8616 ± 0.0410 | 0.005s | 0.000s |
| **RBF-SVM**              | 0.9785 ± 0.0068 | 0.9702 ± 0.0098 | 0.9397 ± 0.0123 | 0.8781 ± 0.0522 | 0.097s | 0.018s |
| **Lowe Ratio Baseline**  | 0.9406 ± 0.0160 | 0.9271 ± 0.0263 | 0.8807 ± 0.0203 | 0.6286 ± 0.0960 | 0.000s | 0.000s |
| **Quantum-Kernel SVM**   | **0.5618 ± 0.0265** | **0.4610 ± 0.0311** | **0.5344 ± 0.0191** | **0.3865 ± 0.0355** | 0.553s | 0.120s |

- **Paired Differences vs. RBF-SVM (Quantum - RBF)**:
  - $\Delta\text{ROC-AUC} = \mathbf{-0.4167}$ (95% Bootstrap CI: $[-0.4314, -0.4017]$)
  - $\Delta\text{PR-AUC} = \mathbf{-0.5092}$ (95% Bootstrap CI: $[-0.5251, -0.4951]$)
- **Definitive Finding**: The quantum kernel feature map collapses towards random chance ($ROC \approx 0.56$). Classical non-linear (HistGradientBoosting, RBF-SVM) and linear (LogisticRegression) models easily achieve $>0.97$ ROC-AUC. Single-feature Lowe-ratio thresholding achieves $0.94$ ROC-AUC. The quantum feature map offers no advantage on this task.

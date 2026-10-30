# Quantum Subsystem Audit Notes

This document tracks all audit findings, discrepancies between specifications (`docs/quantum/03_MODULE_SPECS.md`) and implementation, silent fallbacks, mislabeled algorithms, claim corrections, and deferred items across Q0–Q7.

---

## 1. Summary Status vs. Spec (`03_MODULE_SPECS.md`)

| Module | Spec'd Behavior (`03_MODULE_SPECS.md`) | Current State (Code) | Missing / Unfinished Work | Status |
|--------|----------------------------------------|----------------------|--------------------------|--------|
| **Q0 — QUBO Consistency Filter** | Takes any upstream `MatchResult`, filters for geometric consistency and 1-to-1 uniqueness using QUBO Ising optimization. Classical twin: MAGSAC++/RANSAC inlier set or classical SA on same Q. | `QUBOMatcher.filter_matches()` in [`matcher_qubo.py`](matcher_qubo.py) is complete and functional. Builds QUBO, dispatches to SQA/QAOA/SA, returns filtered `MatchResult`. Classical twin in `service.py:QuantumService._run_q0_filter` uses OpenCV `USAC_MAGSAC` / `RANSAC`. | **(a)** YAML config keys `unary_weight`, `pairwise_weight`, `max_candidates`, `sqa_sweeps`, `sqa_reads` are not read by the service runner (`service.py:QuantumService._run_q0_filter` lines 224-226). **(b)** Pairwise distance ratio kernel `[0.70, 1.40]` in `matcher_qubo.py:filter_matches` (lines 303-310) is too permissive for sparse points with near-miss noise ($s / \bar{d} \lesssim 0.25$), making keep-all the ground state on the default test problem. Any window change would alter `benchmark_hard.json` numbers and is deferred to review. **(c)** Not registered in `src/matching/` pipeline registry. | **OPEN** (reopened for diagnosis; classical twin **RESOLVED 2026-10-30**) |
| **Q1 — QAOA Keypoint-Subset Selection** | Inside each 8×8 grid cell, choose best k=5 matches that are confident and spatially spread. Classical twins: `grid_cap` (top-k) and exact brute-force over 2^K bitstrings measuring approximation ratio. | [`selection_qaoa.py`](selection_qaoa.py): `select_qaoa()` covers all cells, `select_cell_qaoa()` runs QAOA per cell, `solve_cell_exact()` does brute-force over 2^K. `shots` parameter is forwarded to `QAOASolver` and recorded. Additive cell metrics `cells_total`, `cells_trivial`, `cells_qaoa` added. Baseline fixtures from `218c16e` verified. | **(a) HIGH**: Default problem ($N=60, \text{cap}=5$) distributes $\le 4$ candidates per cell, so QAOA is completely bypassed (`cells_trivial=37, cells_qaoa=0`). Non-trivial cell execution requires $K > \text{cap}$. **(b)** `service.py:QuantumService._run_q1_selection` defaults `restarts=3` (YAML specifies `restarts: 5`). **(c)** `max_qubits_per_cell` hardcoded to 12 in `selection_qaoa.py:select_cell_qaoa` line 140 rather than read from YAML. **(d)** `sigma_fraction` hardcoded as `cell_size / 3.0` in `build_cell_qubo` (line 33). **(e)** Not hooked into `src/selection/` pipeline. | **OPEN** (shots forwarding and baseline verification **RESOLVED 2026-10-30**) |
| **Q2 — Quantum-Kernel Classifier** | 8-qubit feature map (H, Rz(x_i), ring of ZZ). Q2a: router predicting best matcher from 13-feature MSM vector. Q2b: inlier scorer from 8 match features. Classical twin: RBF-SVM / LightGBM. | [`kernel_classifier.py`](kernel_classifier.py): `simulate_quantum_feature_state()` implements feature map (H, Rz, ZZ ring). `QuantumKernelClassifier` wraps with SVM. `run_q2_benchmark()` dispatches tasks. Labeled `"SYNTHETIC SMOKE TEST - not evidence of real-data performance"` in code, output, and REST response. | **(a) Data-Source Mismatch**: Zero real PairRecords or `meta.json` exist in the repo (`data/pairs/` and `data/processed/` are empty). Benchmark uses synthetic Gaussian/uniform vectors. 7 of 13 MSM features are constant on synthetic pairs. **(b)** Matcher availability: `lightglue` is not installed; only SIFT and RIFT2 run. SIFT wins 93.3% of benchmark scenes (severe class imbalance). **(c)** YAML `c_reg`, `reps`, `subsample_max` hardcoded. **(d)** Not hooked into `src/selector/` or `src/matching/`. | **OPEN** |
| **Q3 — Grover Consensus Search** | Amplitude amplification over 4-match subsets from n≤16 candidates. Oracle: O(√(N/M)) queries vs O(N/M) for uniform random. Classical twin: Random-sampling RANSAC. | [`grover_consensus.py`](grover_consensus.py): Enumerates C(n,4) subsets, simulates Grover iterations via statevector. | **(a)** Oracle builds full classical truth table first (`grover_consensus.py:grover_consensus_search` lines 144-157), which is O(N) classical evaluation. Must be clearly documented as simulated oracle. **(b)** Parameter name mismatch: YAML specifies `max_candidate_matches: 16` and `subset_size: 4`; code uses `max_candidates` and hardcodes `subset_size=4`. **(c)** Classical twin is only a theoretical query count ratio; missing real timed random-sampling RANSAC. | **OPEN** |
| **Q4 — QFT Phase Correlation** | Gate-level 2D QFT on statevector, amplitude-encoded (2^k × 2^k) tiles, 2D paraboloid sub-pixel fit. Classical twin: `numpy.fft.fft2(..., norm="ortho")`. | [`qft_phase_corr.py`](qft_phase_corr.py): `apply_1d_qft_statevector()` applies H, CPHASE, SWAP. `qft_phase_correlation()` performs cross-correlation. | **(a)** Inverse transform at `qft_phase_corr.py:qft_phase_correlation` line 215 uses `np.fft.ifft2` on cross-power spectrum rather than gate-level inverse QFT sequence. **(b)** Config mismatch: YAML specifies `tile_size: 32`, but `service.py:QuantumService._run_q4_phase_corr` line 382 defaults to `tile_size: 16`. **(c)** `zero_pad` not implemented. **(d)** Not hooked into `src/refinement/`. | **OPEN** |
| **Q5 — Quantum Image Encoding + Edge Detection** | FRQI (2n+1 qubits), NEQR (2n+8 qubits), amplitude encoding. QHED via cyclical permutation + Hadamard on position qubit. Classical twin: OpenCV/SciPy Sobel. | [`image_encoding.py`](image_encoding.py): `encode_frqi()`, `decode_frqi()`, `encode_neqr()`, `decode_neqr()` implemented. | **(a)** `quantum_edge_detection()` (`image_encoding.py` lines 155-156) uses classical central-difference approximation (`0.5 * abs(f[x+2] - f[x])`) instead of real QHED circuit. **(b)** YAML config flags `encoding_type: "frqi"` and `edge_detection: true` are ignored by `service.py:QuantumService._run_q5_encoding` (lines 408-417), which always encodes both FRQI and NEQR and runs edge detection unconditionally. **(c)** Amplitude encoding functions (`encode_amplitude`, `decode_amplitude`) missing. | **OPEN** |
| **Q6 — Backend Abstraction** | Protocol `QuantumBackend` implemented by `numpy_statevector`, `qiskit_aer`, `dwave_neal`, `ibm_runtime`, `dwave_leap`. | [`backends.py`](backends.py): All 5 adapters exist with standard interface and registry. | **(a)** `QiskitAerBackend` (lines 234, 237, 240) bypasses Qiskit Aer by directly instantiating `NumpyStatevectorBackend().run_*`. (Note: Qiskit is NOT INSTALLED; tagged as ASSUMED). **(b)** `IbmRuntimeBackend` (lines 268-283) and `DwaveLeapBackend` (lines 316-335) silently fall back to Numpy internally even with tokens. Must report `unavailable` or return `simulated_fallback=True`. **(c)** YAML `backends:` block is completely unread by the service. | **OPEN** |
| **Q7 — Post-Quantum Signing + Verification** | SHA3-256 digest + Hybrid ML-DSA-65 (FIPS 204 / Dilithium) / Ed25519 digital signatures on all exported artifacts and manifests. Classical twin: plain SHA-256. | [`pqc.py`](pqc.py): Signs and verifies with Ed25519. Classical SHA-256 twin computed. | **(a) CRITICAL**: Code is labeled `"ML-DSA-65+Ed25519"` but executes only 64-byte Ed25519 (`dilithium-py` commented out). Post-quantum protection is absent. **(b)** YAML `sign_extensions: [".tif", ".csv", ".json", ".png"]` and `private_key_env: "VOYAGE_PQC_PRIVATE_KEY"` are hardcoded in code rather than loaded dynamically. | **OPEN** |

---

## 2. Detailed Audit Log & Discrepancies by File

### Claim Corrections
1. **Q1 Test (a) Baseline Provenance (Corrected 2026-10-30)**:
   The initial audit report claimed test (a) matched a saved pre-change baseline. In truth, test (a) initially only ran the post-change code and asserted `"shots" == 1024` without loading any pre-change record. This has now been corrected: real baseline JSON fixtures were generated from a git worktree at `218c16e` (`tests/fixtures/q1_baseline_default_218c16e.json` and `q1_baseline_capped_218c16e.json`), and test (a) now loads and asserts equality against them (excluding only the newly introduced `"shots"` and additive `cells_*` keys). It is proven to fail if optimization parameters (e.g., `p_layers`) deviate.
2. **Qiskit Installation Claim (Corrected 2026-10-30)**:
   The previous status claim that Qiskit is installed was incorrect. An explicit import check had **not** been run prior to making that claim; the assumption was based on seeing import guards (`try: import qiskit ... except ImportError`) pass without throwing unhandled exceptions. Running `python -c "import qiskit"` confirmed that Qiskit is **NOT INSTALLED** in this environment (`Python 3.14.7`).

### `quantum/backends.py`
- **Lines 234, 237, 240 (`QiskitAerBackend` direct Numpy instantiation)**:
  `QiskitAerBackend.run_qaoa`, `sample_qubo`, and `run_circuit` instantiate `NumpyStatevectorBackend().run_*` directly. The claim that this bypasses Aer "even when Aer is installed" is **ASSUMED** from code structure until Qiskit and Qiskit Aer can actually run on this machine.
- **Lines 268-283, 316-335 (`IbmRuntimeBackend`, `DwaveLeapBackend` Silent Fallbacks)**:
  `IbmRuntimeBackend.run_qaoa` and `DwaveLeapBackend.sample_qubo` return numpy statevector/annealer results rather than raising `RuntimeError` or reporting `simulated_fallback=True` when hardware tokens are absent.
- **Top-level `backends:` config unread**:
  The `backends:` section in `configs/quantum.yaml` is never read or loaded.

### `quantum/image_encoding.py`
- **Lines 155-156 (`quantum_edge_detection()` QHED Mislabel)**:
  `quantum_edge_detection()` computes `0.5 * np.abs(np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0))` (classical central difference) and labels it quantum edge detection. True QHED requires cyclical shift gate sequence on the position register followed by a Hadamard on the auxiliary/position qubit.
- **Lines 1-220 (Missing Amplitude Encoding)**:
  Module implements FRQI and NEQR, but lacks the amplitude encoding (`encode_amplitude`, `decode_amplitude`) specified in spec.

### `quantum/qft_phase_corr.py`
- **Line 215 (`qft_phase_correlation()` Hybrid QFT/Classical Inverse)**:
  Forward 2D QFT is computed with statevector quantum gate sequences, but inverse cross-power spectrum uses classical `np.fft.ifft2(cross_power)` rather than gate-level inverse QFT.
- **Tile size config mismatch**:
  `configs/quantum.yaml` sets `tile_size: 32` (10 qubits), but `service.py:QuantumService._run_q4_phase_corr` line 382 defaults to `tile_size: 16` (8 qubits).

### `quantum/grover_consensus.py`
- **Lines 144-157 (`grover_consensus_search()` Classical Truth Table Construction)**:
  Oracle builds an explicit truth table over all $\binom{N}{4}$ subsets classically before Grover simulation, requiring $O(N)$ classical evaluations.
- **Lines 198-210 (`theoretical_grover_speedup()` Theoretical-only Baseline)**:
  Classical twin compares only $O(N/M)$ formula against $O(\sqrt{N/M})$; no timed random-sampling RANSAC is executed.
- **Config key mismatch**:
  YAML config uses `max_candidate_matches: 16` and `subset_size: 4`; `grover_consensus.py` uses `max_candidates` and hardcodes `subset_size=4`.

### `quantum/kernel_classifier.py`
- **Lines 188-258 (`run_q2_benchmark()` Synthetic Data Smoke Test)**:
  **RESOLVED 2026-10-30**: Code, return dictionary, and REST responses are labeled `"SYNTHETIC SMOKE TEST - not evidence of real-data performance"`.

### `quantum/pqc.py`
- **Lines 27-28, 52-70 (`sign_artifact()` Ed25519 Mislabel as ML-DSA-65)**:
  Claims `"ML-DSA-65+Ed25519"` algorithm in manifests and records, but generates only 64-byte Ed25519 signature. FIPS 204 ML-DSA-65 is commented out and not executed.
- **Hardcoded config**:
  `sign_extensions` and `private_key_env` are hardcoded rather than loaded from `configs/quantum.yaml`.

### `quantum/service.py`
- **Lines 228-256 (`QuantumService._run_q0_filter()` Classical Twin)**:
  **RESOLVED 2026-10-30**: Updated to run OpenCV `USAC_MAGSAC` / `RANSAC` homography inlier estimation as classical twin with median-cut recorded side-by-side.
  - **OpenCV Seed Finding**: `cv2.setRNGSeed(seed)` has **no effect** on `cv2.USAC_MAGSAC` (Jaccard similarity across 10 seeds is 1.000000; USAC uses its own internal C++ generator controlled by `cv2.UsacParams.randomGeneratorState`). Row-order permutation alters hypothesis sampling. Documented in code comment at line 234.
- **Lines 289-326 (`QuantumService._run_q1_selection()` Shots Forwarding & Cell Metrics)**:
  **RESOLVED 2026-10-30**: `shots` parameter is forwarded to `select_qaoa` and `select_cell_qaoa`, reaching `QAOASolver` and recorded in `metrics`, `problem`, and `circuit`. Additive cell metrics `cells_total`, `cells_trivial`, `cells_qaoa` are computed and recorded.
- **Line 295 (`restarts` Default Mismatch)**:
  `restarts=params.get("restarts", 3)` defaults to 3 rather than the 5 specified in `configs/quantum.yaml`.

### `quantum/selection_qaoa.py`
- **Line 140 (`max_qubits_per_cell` Hardcoded)**:
  `select_cell_qaoa` caps candidates to 12 using a hardcoded literal rather than reading `max_qubits_per_cell` from configuration.
- **Line 33 (`sigma_fraction` Hardcoded)**:
  `build_cell_qubo` uses `sigma = cell_size / 3.0` rather than reading `sigma_fraction` (0.333333) from configuration.

---

## 3. Global Findings

1. **Config Merging in `QuantumService.run_module`**:
   `service.py:QuantumService.run_module` (lines 123-124) sets `p = params or {}` and `inp = inputs or {}`. It never merges module-specific configuration from `self.config["modules"][mid]` into `p`. Parameters defined in `configs/quantum.yaml` are ignored unless passed manually by the caller.
2. **Promotion Gate File-Existence Only & Q7 Exemption**:
   `service.py:is_module_promoted` (lines 53-56) only checks whether `results/quantum/promotion/<module_id>.json` exists on disk (`report_file.exists()`). It does not validate report contents or signatures. Furthermore, `service.py:QuantumService.run_module` line 127 and `api/routes/quantum.py` line 170 explicitly exempt `Q7_signing` from the promotion gate (`module != "Q7_signing"`).
3. **Absence of Quantum Calls in `src/`**:
   A full-codebase scan confirms there are **zero references or imports** to `quantum` inside `src/`. The quantum subsystem runs strictly in standalone scripts, test harnesses, or via `/api/quantum/*`.
4. **Hardware Adapters Fall Back to Numpy**:
   In `quantum/backends.py`, both `IbmRuntimeBackend` and `DwaveLeapBackend` fall back directly to `NumpyStatevectorBackend` and `SimulatedQuantumAnnealer` without raising an error or returning `simulated_fallback=True`.
5. **Runtime Discrepancy in `quantum/README.md`**:
   Table row for $M=16$ QAOA statevector simulation in `quantum/README.md` (line 206) lists wall-clock runtime as **13.397s**, whereas actual empirical execution measures **13.89s**.
6. **Classical SA vs. SQA Solvers (ASSUMED Status)**:
   - `ClassicalSimulatedAnnealer` (`quantum/classical_baseline.py` lines 249-327) and `SimulatedQuantumAnnealer` (`quantum/annealer.py` lines 47-223) are independent implementations: Classical SA uses a single classical spin configuration ($P=1$) with thermal Metropolis updates; SQA implements Path-Integral Monte Carlo with $P=16$ Trotter replicas and inter-slice transverse quantum tunnel coupling $J_t$.
   - The claim that *"SA and SQA both reach the ground state across all problems"* is **ASSUMED** pending comprehensive listing of per-instance energies from both solvers over all benchmark seeds.
7. **Q1 Default Problem QAOA Bypass (HIGH)**:
   In the default service invocation ($N=60, \text{cap}=5$ on an $8 \times 8$ grid), all 37 occupied cells have $\le 4$ candidates. Because $K \le k$ across all cells, QAOA is **100% bypassed** (`cells_trivial=37, cells_qaoa=0`). Evaluating QAOA requires test problems where local cell density exceeds the cap.

---

## 4. Q0 Operating Envelope & Diagnostic Finding

1. **Solver Succeeded**: SQA reached the exact QUBO global minimum energy ($-255.440524$), confirmed identical to full $2^{24} = 16,777,216$ exact brute-force enumeration.
2. **Formulation Weakness & Operating Envelope**:
   - The pairwise distance ratio window $[0.70, 1.40]$ in `matcher_qubo.py` is scale- and density-sensitive.
   - On sparse keypoints with near-miss noise ($s / \bar{d} \lesssim 0.25$, e.g. $\sigma \le 30\text{px}$ on $400\text{px}$ canvas), distance ratios stay within $[0.70, 1.40]$, awarding negative energy rewards to outliers (262 rewards vs 14 penalties on Problem 1), making keep-all-24 the true ground state.
   - Q0 filters successfully when outlier displacement is large relative to mean inter-point distance ($s / \bar{d} \gtrsim 0.35$, e.g. dense clusters, $\sigma \ge 50\text{px}$ on small canvas, or gross uniform outliers), where distance ratios escape $[0.70, 1.40]$ and trigger penalties.
   - Changing the ratio window in `qubo_builder.py` or `matcher_qubo.py` would shift established numbers across `benchmark_hard.json` and `README.md`; therefore, any formulation modification is deferred to the review phase.

---

## 5. Phase Scope: Deferred vs. In-Scope

### Deferred (Record Only, Do Not Change)
1. **YAML key wiring**: Binding unused config keys to functions when doing so would alter existing baseline benchmark numbers.
2. **Q0 ratio window tuning**: Modifying $[0.70, 1.40]$ window or distance formulas that would alter `benchmark_hard.json`.
3. **Pipeline integration (`src/` hooks)**: Hooking Q0–Q7 into `src/matching/`, `src/selection/`, `src/selector/`, `src/refinement/`. A unified shadow hook will be evaluated after Q7.

### In-Scope (Implemented Module by Module)
1. Every functional gap against `03_MODULE_SPECS.md`.
2. Every place where a function or algorithm claims more than the code actually executes.
3. Adding/extending test suites for each module, verified pass/fail verbatim.

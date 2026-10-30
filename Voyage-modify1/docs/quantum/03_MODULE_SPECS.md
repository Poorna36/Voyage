# 03 — Module Specifications (Q0–Q7)

## Q0 — QUBO correspondence matcher & consistency filter (L2)
- Q0b Consistency Filter takes any upstream `MatchResult` and filters matches for geometric consistency and 1-to-1 uniqueness using QUBO Ising optimization.
- Classical twin: MAGSAC++ / RANSAC inlier set or classical Simulated Annealing on identical $Q$.

## Q1 — QAOA keypoint-subset selection (L3)
- Inside each cell of an $8 \times 8$ grid, choose the best $k$ matches (default $k=5$) that are confident and spatially spread out.
- Classical twins: `grid_cap` (top-k) and exact brute-force over $2^K$ bitstrings to measure approximation ratio.

## Q2 — Quantum-kernel classifier (L1.5 router, L4 inlier scoring)
- 8-qubit feature map with Hadamard, $R_z(x_i)$, and ring of $ZZ$ interactions.
- Q2a: Router predicting best matcher from 13-feature MSM vector.
- Q2b: Inlier scorer classifying candidate matches as inlier/outlier from 8 match features.
- Classical twin: RBF-SVM / LightGBM.

## Q3 — Grover search over RANSAC hypotheses (L4)
- Amplitude amplification searching over 4-match subsets from $n \le 16$ candidate matches.
- Oracle queries: $O(\sqrt{N/M})$ for Grover vs $O(N/M)$ for uniform random sampling.
- Classical twin: Random-sampling RANSAC.

## Q4 — QFT phase correlation (L5)
- Explicit gate-level 2D Quantum Fourier Transform on statevector for amplitude-encoded tiles ($2^k \times 2^k$).
- 2D paraboloid fit for sub-pixel translation estimation.
- Classical twin: `numpy.fft.fft2(..., norm="ortho")` phase correlation.

## Q5 — Quantum image encoding and edge detection (Preprocessing demo)
- FRQI ($2n+1$ qubits), NEQR ($2n+8$ qubits), amplitude encoding.
- Quantum Hadamard Edge Detection (adjacent pixel difference via cyclical permutation + Hadamard).
- Classical twin: OpenCV / SciPy Sobel gradient magnitude filter.

## Q6 — Backend abstraction
- Protocol `QuantumBackend` implemented by `numpy_statevector` (default), `qiskit_aer`, `dwave_neal`, `ibm_runtime`, and `dwave_leap`.

## Q7 — Post-quantum signing and key exchange (L6, API)
- SHA3-256 digest + Hybrid ML-DSA-65 (FIPS 204) / Ed25519 digital signatures on all exported artifacts and manifests.
- Classical twin: Plain SHA-256 integrity checksum.

# 05 — Benchmark and Testing Protocol

## 1. Principles
1. **Identical Inputs**: The quantum module and its classical twin receive the exact same inputs and seed.
2. **Honest Accounting**: Gate counts, simulated statevector queries, and circuit depth are reported truthfully.
3. **Promotion Gate**: A module cannot transition from `shadow` to `active` without a passing report in `results/quantum/promotion/<module>.json`.
   - Q0: Non-inferiority to MAGSAC++ within 0.10 px RMSE and 0 one-to-one constraint violations.
   - Q1: Mean approximation ratio >= 0.95 on >= 90% of cells at p=2; pair runtime within 60s limit.
   - Q2: Non-inferiority in AUC/accuracy compared to classical SVM.
   - Q3: Oracle query count <= 1.5x theoretical bound and success rate >= 0.90.
   - Q4: Circuit vs FFT2 difference < 1e-9; integer shift recovery on >= 95% of noise-free cases.
   - Q5: Reconstruction error within quantization bound and Pearson correlation >= 0.85 vs Sobel.
   - Q7: 100% tamper detection rate on invalid byte flips.

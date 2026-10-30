# 04 — API and Dashboard

## Route group: `api/routes/quantum.py`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/quantum/status` | Modules, modes, backends and availability, limits, versions |
| POST | `/api/quantum/run` | Start one module run; returns `run_id` (HTTP 202) |
| GET | `/api/quantum/runs/{run_id}` | Status and `QuantumRunRecord` when finished |
| GET | `/api/quantum/runs` | History (filter by `pair_id`, `module`, `limit`) |
| GET | `/api/quantum/circuit/{module}` | Gate/qubit summary and a drawable gate list for a small example |
| GET | `/api/quantum/benchmarks/latest` | Latest benchmark summary per module and condition |
| POST | `/api/quantum/sign` | Sign an exported product (Q7) |
| POST | `/api/quantum/verify` | Verify a product against its signature (Q7) |

## Dashboard: "Quantum Lab"

Integrated directly into the Mission Control Workbench as a dedicated tab (`quantum`).
Includes:
- Honesty Banner
- Run Configuration Panel
- Side-by-side Quantum vs Classical Twin Comparison Table
- Interactive Circuit Visualizer
- Measurement Probability Histograms
- Optimization Energy Convergence Charts
- Post-Quantum Security & Verification Panel
- Module x Condition Benchmark Board

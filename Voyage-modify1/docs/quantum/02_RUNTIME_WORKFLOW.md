# 02 — Runtime Workflow

How one image pair moves through the pipeline with the quantum subsystem
attached. Two flows are covered: the **pipeline flow** (quantum modules attached to
L0–L7) and the **Quantum Lab flow** (a user runs one module from the dashboard).

## 1. Pipeline flow (hybrid, shadow-first)

```text
L0 Ingest + geometry
 └── L1 Preprocess ───────────────────────► Q5 (shadow demo on tiles)
      └── L1.5 MSM router (LightGBM) ─────► Q2a (shadow: kernel router)
           └── L2 Matchers ───────────────► Q0 (optional M4 matcher/filter)
                └── L3 ANMS + grid cap ───► Q1 (shadow: QAOA subset selection)
                     └── L4 MAGSAC++ ─────► Q3 (shadow: Grover), Q2b (inlier score)
                          └── L5 NCC/POC ─► Q4 (shadow: QFT phase correlation)
                               └── L6 Export ──► Q7 (sign every product)
                                    └── L7 Evaluation & records
```

## 2. Quantum Lab flow (dashboard user path)

1. Pick pair, module, backend, seed in dashboard.
2. `POST /api/quantum/run` validates pair and module, queues job, returns `202 Accepted` with `run_id`.
3. Background task executes both quantum module and classical twin on identical inputs.
4. Dashboard polls `GET /api/quantum/runs/{run_id}` until completed.
5. Renders side-by-side comparison table, gate statistics, and convergence charts.

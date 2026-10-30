# 01 — Quantum Subsystem Architecture

## 1. Module map

| ID | Name | Pipeline layer | Classical twin (already in repo) | Default mode |
|----|------|----------------|----------------------------------|--------------|
| Q0 | QUBO correspondence matcher (exists) | L2 matcher `M4` | SIFT + Lowe + RANSAC; Hungarian; classical SA | `off` in pipeline, `shadow` in lab |
| Q1 | QAOA keypoint-subset selection | L3 spatial selection | `grid_cap` / ANMS (`src/selection/`) | `shadow` |
| Q2 | Quantum-kernel classifier | L1.5 router and L4 inlier scoring | LightGBM MSM (`src/selector/`), RBF-SVM | `shadow` |
| Q3 | Grover search over RANSAC hypotheses | L4 geometric verification | MAGSAC++/DEGENSAC (`src/registration/`) | `shadow` |
| Q4 | QFT phase correlation | L5 sub-pixel refinement | FFT phase correlation (`src/refinement/local.py`) | `shadow` |
| Q5 | Quantum image encoding + edge detection | Preprocessing demo track | Sobel / Canny (OpenCV) | `shadow` |
| Q6 | Backend abstraction | cross-cutting | NumPy statevector (current) | `numpy_statevector` |
| Q7 | Post-quantum signing / key exchange | L6 export, API | SHA-256 manifests (none signed today) | `active` when library present |

## 2. Where quantum plugs into L0–L7

```text
L0  Data & Geometry ............ unchanged
L1  Preprocessing .............. unchanged            ── Q5 (shadow demo on tiles)
L1.5 Matcher Selection (MSM) ... LightGBM router      ── Q2 (shadow: kernel router)
L2  Correspondence ............. SIFT/RIFT2/LNIFT/LG/Crater ── Q0 (matcher M4)
L3  Spatial selection .......... ANMS + 8x8 grid cap  ── Q1 (shadow: QAOA per cell)
L4  Geometric verification ..... MAGSAC++/DEGENSAC    ── Q3 (shadow: Grover), Q2 (inlier score)
L5  Sub-pixel refinement ....... NCC/POC + paraboloid ── Q4 (shadow: QFT phase corr)
L6  Cartographic export ........ GeoTIFF + GCP CSV    ── Q7 (sign every product)
L7  Evaluation & arbitration ... leaderboard + log    ── quantum run records appended
```

Rule: a quantum module reads the same inputs the classical step reads and writes
to its own `quantum_*.json` artifact. It does not overwrite classical artifacts
(`matches_selected.json`, `geometry.json`, `matches_refined.json`) unless its
mode is `active`.

## 3. Folder layout

```text
quantum/
├── __init__.py                 exports
├── qubo_builder.py             existing
├── annealer.py                 existing
├── qaoa.py                     existing
├── classical_baseline.py       existing
├── matcher_qubo.py             existing + Q0b filter mode
├── benchmark.py                existing
├── backends.py                 Q6 backend registry + adapters
├── selection_qaoa.py           Q1 QAOA spatial selection
├── kernel_classifier.py        Q2 Quantum-kernel classifier
├── grover_consensus.py         Q3 Grover hypothesis search
├── qft_phase_corr.py           Q4 2D QFT phase correlation
├── image_encoding.py           Q5 FRQI/NEQR + QHED
├── pqc.py                      Q7 Post-quantum hybrid signing
├── records.py                  QuantumRunRecord dataclass + JSON writer
├── service.py                  facade used by the API & pipeline
└── results/                    run JSON records

api/routes/quantum.py           FastAPI route group
configs/quantum.yaml            Subsystem configuration
requirements-quantum.txt        Optional extras
sih-dashboard/src/components/quantum/   QuantumLab and tabs
sih-dashboard/src/services/quantumApi.ts typed client
tests/test_quantum_*.py         Unit and integration tests
docs/quantum/                   Documentation
```

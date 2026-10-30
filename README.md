# Voyage — Documentation

Cross-sensor lunar image registration for Chandrayaan-2 (SIH 2026, PS-26166).
Documentation set rewritten from a full review of branch `q_initial` (commit `218c16e`).

Everything in this set describes the repository as it is, not as it was planned.
Where the code and an older plan disagree, the code wins and the difference is
recorded in `10_KNOWN_ISSUES.md`.

## Evidence legend

| Mark | Meaning |
|------|---------|
| ✅ | Verified in this review: the code was run, or the code path was read end to end |
| 📄 | Taken from a docstring, config file or older document; not exercised in this review |
| ⚠️ | A defect or misleading behaviour was found (details in `10_KNOWN_ISSUES.md`) |

## Reading order

| # | File | Read it to learn |
|---|------|------------------|
| 00 | `00_PROJECT_OVERVIEW.md` | The problem, the sensors, what works today, glossary |
| 01 | `01_SYSTEM_ARCHITECTURE.md` | Layers L0–L7, repository map, the three runtime surfaces |
| 02 | `02_DATA_AND_ARTIFACTS.md` | Folders, manifest schema, files each stage writes |
| 03 | `03_PIPELINE_STAGES.md` | Stage S0–S9 runbook with commands, gates and failures |
| 04 | `04_MATCHERS_AND_ALGORITHMS.md` | Matchers M0–M4, selector, selection, verification, refinement |
| 05 | `05_BACKEND_API.md` | Every endpoint, validation rules, what is real and what is simulated |
| 06 | `06_DASHBOARD.md` | Frontend structure, user flows, offline behaviour |
| 07 | `07_CONFIGURATION.md` | Config files, key parameters, tunable values, conflicts |
| 08 | `08_TESTING_AND_VALIDATION.md` | Test suites and results, validation protocol, synthetic benchmark |
| 09 | `09_DEPLOYMENT_AND_OPERATIONS.md` | Local run, Docker, Railway, Vercel, environment variables |
| 10 | `10_KNOWN_ISSUES.md` | Verified defects, ranked, with fixes |
| Q | `quantum/README.md` | The quantum subsystem: honest state, design, implementation plan |

## One-page summary

- **What it is.** A pluggable, benchmark-first pipeline that finds point
  correspondences between Chandrayaan-2 images (OHRC, TMC-2, IIRS) and LRO
  reference imagery, verifies them geometrically, refines them to sub-pixel
  accuracy, and exports registered GeoTIFFs and ground-control points.
- **What is solid.** The scientific engine in `src/` (about 12,300 lines) and its
  tests. The full test run gives 20,454 passed and 696 skipped (see
  `08_TESTING_AND_VALIDATION.md` for why that headline overstates coverage).
- **What is not the engine.** The FastAPI server is a bridge for the dashboard.
  Its `/api/pipeline/run` endpoint does not run the pipeline (⚠️ see 05 and 10).
- **Quantum.** Implemented as simulations behind `/api/quantum/*` and a dashboard
  tab. Several claims and numbers in it are not yet backed by the code (⚠️);
  the quantum docs give the corrected design and an implementation plan.
- **Deployment.** The Docker image omits the `quantum/` package, so the API would
  fail to start (⚠️ reproduced).

## Conventions used across the docs

- Pixel coordinates are always `(col, row) = (x, y)`, 0-indexed, origin at the
  centre of the top-left pixel. Array access is `image[row, col]`.
- Geographic coordinates are `[longitude, latitude]` in decimal degrees.
- "Pair" means one source image (Chandrayaan-2) matched to one reference image (LRO/SELENE).
- Paths are relative to the project root, the folder that contains `api/`, `src/`
  and `configs/` (in the Git repository this is the `Voyage-modify1/` folder).

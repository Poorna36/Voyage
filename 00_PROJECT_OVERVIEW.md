# 00 — Project Overview

## 1. Problem

Smart India Hackathon 2026, problem statement PS-26166: automatically register
Chandrayaan-2 optical and hyperspectral images against global lunar reference
imagery, despite large differences in resolution, lighting and geometry.

| Challenge | Range stated in the project README 📄 |
|-----------|----------------------------------------|
| Illumination | solar azimuth differences up to 180°, solar incidence up to 85° |
| Scale gap | resolution mismatch up to 17x (OHRC 0.3 m/px against TMC-2 5 m/px) |
| Relief and curvature | polar regions and high relief break a single planar (homography) model |
| Sensor modality | optical (OHRC, TMC-2) and hyperspectral (IIRS) data need different handling |

## 2. Sensors and references

| Name | Role | Notes |
|------|------|-------|
| OHRC | Chandrayaan-2 high-resolution camera (source) | about 0.31 m/px in the project configs |
| TMC-2 | Chandrayaan-2 terrain mapping camera (source) | about 5 m/px; the TMC-2 branch is not validated against real data |
| IIRS | Chandrayaan-2 imaging infrared spectrometer (source) | about 80 m/px, 250 bands, 0.8–5.0 µm; handled by a separate module (`src/matching/iirs.py`) |
| LRO NAC | Reference, narrow-angle camera | about 0.5 m/px; queried through the NASA Lunar Orbital Data Explorer (ODE) REST API |
| LRO WAC | Reference, wide-angle camera | cropped from a mosaic with GDAL/rasterio |
| SELENE | Optional third reference | connectivity check exists; the Moon Trek retrieval path is the least exercised |

Reference acquisition follows a fallback chain: NAC (ODE) → WAC crop → SELENE → skip the pair.

## 3. What the system produces

For each pair that passes its gates:

1. Verified point correspondences and a geometric model (similarity, affine or homography, or tile-wise local models).
2. Sub-pixel refined matches.
3. A registered GeoTIFF warped onto the reference grid and a ground-control-point (GCP) table.
4. Quality-control images (checkerboard, residual heat map, match overlay).
5. Evaluation records, a per-matcher leaderboard, and an arbitration log.

## 4. Current status (branch `q_initial`)

| Area | State | Evidence |
|------|-------|----------|
| Scientific engine (`src/`: ingest, preprocessing, matchers, selection, verification, refinement, evaluation) | Implemented (about 12,300 lines); covered by unit, stress and integration tests | ✅ 20,454 tests passed, 696 skipped (Python 3.12, without torch/ultralytics installed) |
| Command-line pipeline (`scripts/`, stages S1–S9) | Implemented as scripts | 📄 not run end to end: the repository ships no raw or paired data (`data/raw`, `data/pairs`, `data/processed` are empty or git-ignored) |
| Matcher selector (MSM) | Trained model shipped (`models/msm_v1.pkl`) | ✅ metadata read: 52 training samples, 5-fold CV accuracy 0.8855, four classes; see ⚠️ in 04 |
| Synthetic benchmark | Implemented (anchors, transforms, scorecards) | ✅ unit tests pass; 📄 no stored benchmark results |
| FastAPI backend | Implemented, 6 route groups | ✅ imported and tested; ⚠️ `/api/pipeline/run` is a simulation |
| React dashboard | Implemented (about 13,900 lines of TS/TSX) | 📄 not built in this review (no `node_modules`) |
| Quantum subsystem | Implemented as CPU simulations with API, tests and a dashboard tab | ✅ 25 tests pass; ⚠️ several misleading claims (see `quantum/`) |
| Deployment (Docker, Railway, Vercel) | Files present | ⚠️ the Docker image omits `quantum/`, so the API cannot start (reproduced) |

## 5. Who and what is in the repository

| Item | Where |
|------|-------|
| Backend engine | `src/` |
| Command-line stages | `scripts/` and `generate_pristine_dataset.py` |
| API server | `api/` |
| Dashboard | `sih-dashboard/` |
| Quantum subsystem | `quantum/`, `api/routes/quantum.py`, `sih-dashboard/src/components/quantum/` |
| Configs | `configs/` |
| Models | `models/` (`msm_v1.pkl`, `msm_v1_stats.json`, `crater_yolov9.pt`) |
| Synthetic generator code | `src/synthetic/`, `synthetic/` |
| Tests | `tests/` |
| Older documents | `AUDIT.md`, `FLOWS.md`, `CHANGES.md`, `progress.md`, and the previous `docs/` set (superseded by this set) |

## 6. Honest limits of what is demonstrated

- No real registration accuracy number is stored in the repository. Reported
  figures in older documents come from planned targets or lookup tables.
- The API run endpoint returns table-based metrics, so a dashboard "run" does
  not demonstrate the engine (see `05_BACKEND_API.md`).
- The matcher selector is trained on 52 samples; treat its accuracy as indicative only.
- The crater detector checkpoint is not validated on OHRC or TMC imagery; its
  labels came from Hough circles, not from the Robbins catalogue (per the project's own audit notes).

## 7. Glossary

| Term | Meaning |
|------|---------|
| ANMS / SSC | Adaptive Non-Maximal Suppression via Square Covering: spreads keypoints evenly over an image |
| ASP / ISIS3 | Ames Stereo Pipeline / USGS planetary image software; used for camera models and SPICE geometry |
| DEGENSAC | RANSAC variant that rejects degenerate (planar) configurations |
| F2 checks | Mandatory bounds and one-to-one checks on every match set before geometric fitting |
| GCP | Ground control point: a matched point pair with geographic position |
| GSD | Ground sample distance: metres per pixel |
| LightGlue / SuperPoint | Learned keypoint detector and attention-based matcher (matcher M2) |
| MAGSAC++ | Robust model-fitting estimator used for outlier rejection |
| MSM | Matcher Selection Model: LightGBM classifier that routes each pair to a matcher |
| NCC / POC | Normalised cross-correlation / phase-only correlation (sub-pixel refinement) |
| PDS4 | NASA Planetary Data System format; Chandrayaan-2 labels are PDS4 XML |
| QAOA / SQA / QUBO | Quantum approximate optimisation / simulated quantum annealing / quadratic unconstrained binary optimisation |
| RIFT2 / LNIFT | Illumination-robust feature matchers based on phase congruency / Gabor-style descriptors |
| SLZ | Safe landing zone (hazard analysis shown in the dashboard) |
| SPICE | NASA geometry kernels used to compute spacecraft and sun geometry |

# Phased Implementation Roadmap
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document specifies the technical implementation phases, module responsibilities, artifact deliverables, and verification checkpoints for the pipeline.

---

## Phase 0: Environment Setup and Data Scaffolding

### Deliverables
- Establish Ames Stereo Pipeline (ASP >= 3.7.0) and ISIS3 conda environment.
- Configure ISISDATA root and fetch Chandrayaan-2 base camera kernels.
- Retrieve 40-day SPICE CK kernel window matching target observation epochs.
- Initialize project directory structure (`configs/`, `data/`, `src/`, `scripts/`, `results/`).
- Ingest pilot Chandrayaan-2 Level-2 products (OHRC, TMC-2) and corresponding LRO reference strips.

### Verification Checkpoint
- `stereo_gui --version` confirms ASP >= 3.7.0.
- Directory structure conforms to specification; initial pilot data verified in `data/raw/`.

---

## Phase 1: Data Ingestion and Geometry (`Layer 0`)

### Deliverables
- `src/ingest/label_parser.py`: Parses PDS4 XML metadata to extract four-corner footprint coordinates, solar angles, acquisition epoch, and spatial resolution.
- `src/ingest/spice_init.py`: Wraps USGS `isisimport` and `spiceinit`, handling SPICE kernel attachment and camera model initialization.
- `scripts/build_pairs.py`: Calculates pointing uncertainty bounding boxes ($k \cdot \sigma$), queries the Lunar ODE REST API for overlapping LRO NAC frames, executes local WAC crops, and generates `manifest.jsonl`.

### Key Contracts
- Input: `data/raw/*.zip` (preserving original ISRO filenames).
- Output: `data/calibrated/*.cub`, `data/metadata/products.jsonl`, and `data/pairs/manifest.jsonl`.
- Tests: Unit tests T01 (metadata extraction) and T02 (bounding box padding calculation).

---

## Phase 2: Radiometric Preprocessing and Normalization (`Layer 1`)

### Deliverables
- `src/preprocessing/masks.py`: Evaluates solar incidence geometry and local intensity variance to generate binary validity masks (`valid_mask.png`).
- `src/preprocessing/normalize.py`: Applies 2nd/98th percentile dynamic range clipping and statistical moment transfer.
- `src/preprocessing/branches.py`: Tailors preprocessing according to sensor type (CLAHE + PCA for OHRC-NAC; histogram matching for TMC-WAC; minimal clipping for learned matchers).
- `src/preprocessing/resample.py`: Pyramid-resamples the coarser image to match GSD scales, applying solar-adaptive interpolation (bilinear under low sun, bicubic under high sun).
- `src/preprocessing/tiling.py`: Partitions large footprints into overlapping processing tiles ($512 \times 512\text{ px}$ with 64 px overlap).
- `scripts/preprocess.py`: CLI orchestration emitting preprocessed images and `meta.json`.

### Verification Checkpoint
- Masked pixel fraction falls within [5%, 30%] on nominal pairs (Test T03).
- Normalized source patch mean and variance match reference statistics within 5% (Test T04).

---

## Phase 3: Correspondence Matching Engine (`Layer 2`)

### Deliverables
- `src/matching/base.py`: Abstract base class defining standard matching interfaces (`detect`, `describe`, `match`).
- `src/matching/sift.py`: Classical M0 baseline using Difference-of-Gaussians and Lowe ratio filtering (0.75).
- `src/matching/rift.py`: M1 matcher implementing RIFT2 phase congruency and multi-octave log-Gabor scale-space search.
- `src/matching/lightglue.py`: M2 learned matcher integrating SuperPoint keypoint extraction and LightGlue transformer attention layers with automatic CPU fallback.
- `src/matching/crater.py`: M3 crater-geometry matcher integrating YOLOv9 crater detection and CNSF topological graph matching with Hough circle CPU fallback.
- `src/matching/registry.py`: Dynamic matcher factory and execution runner.

### Verification Checkpoint
- SIFT generates >= 50 valid candidates on textured scenes (Test T06).
- LightGlue enforces in-domain coordinate bounds and one-to-one mapping (Test T07).
- Crater matcher enforces quantitative density gating ($\ge \tau_c$).

---

## Phase 4: Uniform Spatial Selection (`Layer 3`)

### Deliverables
- `src/selection/anms.py`: Implements Suppression via Square Covering (SSC) for pre-description keypoint pruning on M0/M1.
- `src/selection/spatial.py`: Implements $8 \times 8$ spatial grid density budgeting, per-cell match capping (max 5), aggregate budget bisection (target 250 matches), and one-to-one conflict resolution.

### Verification Checkpoint
- Post-selection spatial coverage >= 0.60 across valid cells (Test T08).
- Output matches exported to `results/<pair_id>/<matcher>/matches_selected.json`.

---

## Phase 5: Geometric Verification and Sub-Pixel Refinement (`Layer 4` and `Layer 5`)

### Deliverables
- `src/registration/checks.py`: Enforces geometric sanity rules and coordinate boundaries.
- `src/registration/ladder.py`: Evaluates DEGENSAC / MAGSAC++ over the hierarchical model ladder (Similarity -> Affine -> Homography).
- `src/registration/tilewise.py`: Local tile-wise affine/homography fitting with Gaussian distance blending for polar ($|\text{lat}| > 55^\circ$) or rugged terrain.
- `src/registration/declustering.py`: Enforces GSD-scaled minimum spatial separation and $3\sigma$ Z-score outlier filtering.
- `src/refinement/local.py`: Windowed local normalized cross-correlation and phase correlation with Tukey window apodization, 3-level Gaussian pyramids, 2D paraboloid peak interpolation, and multimodal peak rejection.

### Verification Checkpoint
- DEGENSAC recovers known homographies within 0.1 px on synthetic tests (Test T09).
- Sub-pixel refinement achieves positive accuracy gain on >= 60% of test pairs (Test T11).

---

## Phase 5.5: Matcher Selection Model (`Layer 1.5` / `Stage 4.5`)

### Deliverables
- `src/selector/features.py`: Extracts 13-dimensional scene and sensor feature vectors from `PairRecord` and L1 `meta.json`.
- `src/selector/model.py`: Encapsulates LightGBM multi-class meta-model inference, rule-based hard gates, and dual-threshold confidence routing ($\tau_{\text{high}}=0.65, \tau_{\text{low}}=0.40$).
- `scripts/train_msm.py`: Training script with strictly disjoint $10^\circ \times 10^\circ$ geographic cell cross-validation (F15).
- `src/evaluation/msm_eval.py`: Evaluates selector accuracy and verifies the 8 Acceptance Criteria (AC1 through AC8).

### Verification Checkpoint
- Feature extraction completes in under 100 ms with verified determinism (Test T13).
- Model satisfies AC1 (accuracy >= 70%) and AC5 (runtime reduction >= 50%) on held-out test splits.

---

## Phase 6: Cartographic Product Generation (`Layer 6`)

### Deliverables
- `scripts/register.py`: Warps source imagery onto the reference cartographic grid using estimated geometric models.
- GeoTIFF Exporter: Emits georeferenced 16-bit GeoTIFFs preserving reference map projections.
- GCP Manifest Generator: Outputs GDAL-compatible Ground Control Points and tabular CSVs.
- Diagnostic Graphics: Generates 64 px checkerboard overlays, residual displacement vectors, and Gaussian residual heatmaps.

### Verification Checkpoint
- Warped products confirm valid coverage across >= 90% of overlapping footprints.
- GeoTIFF products verify cleanly in standard GDAL/QGIS toolchains.

---

## Phase 7: Ground Truth and Benchmark Annotation

### Deliverables
- Checkpoint Catalog: Establishes manual and synthetic ground-truth control points across >= 30 stratified test pairs.
- Schema Compliance: Implements `eval` (held-out evaluation), `fit` (numerical conditioning), and `qc` (inter-annotator variance) partitions.
- Precision Baseline: Evaluates duplicate annotations in `qc` to compute $\text{RMSE}_{\text{interann}}$.

### Verification Checkpoint
- At least 20 evaluation checkpoints per test pair.
- Inter-annotator precision establishes the experimental baseline for all algorithm claims.

---

## Phase 8: System Evaluation and Winner Arbitration (`Layer 7`)

### Deliverables
- `src/evaluation/metrics.py`: Computes RMSE, $\text{pct\_lt\_1px}$, $\text{pct\_lt\_0p5px}$, MedAE, inlier counts, inlier ratios, spatial coverage, and execution times.
- `src/evaluation/leakage_audit.py`: Validates complete geographic cell independence between training and test manifests.
- `src/evaluation/arbitration.py`: Implements production winner arbitration and logging (`results/arbitration.log`).
- `src/evaluation/aggregate.py`: Compiles stratified competitive leaderboards (`results/leaderboard.csv`).

### Verification Checkpoint
- Leakage audit passes with exit code 0 across all manifest splits.
- Multi-strata reporting explicitly displays polar, high-latitude, and partial-overlap performance.

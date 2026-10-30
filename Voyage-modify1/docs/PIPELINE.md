# Pipeline Execution Runbook
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document specifies the operational execution runbook, stage sequence, CLI commands, quality gates, and failure recovery protocols for the image registration pipeline.

All intermediate artifacts are namespaced by `pair_id` (e.g. `ohr_20200827T0030__nac_M123456789`) and candidate matcher (`sift`, `rift2`, `lnift`, `lightglue`, `crater`).

---

## 1. Pipeline Execution Flow

```text
data/raw/ (Uncompressed Level-2 ISRO zips; original filenames preserved)
      |
      | S1 Ingest           scripts/ingest.py
      v
data/calibrated/*.cub  +  data/metadata/products.jsonl                 [L0]
      |
      | S2 Pair Building    scripts/build_pairs.py
      v
data/pairs/manifest.jsonl  (+ reference crops in data/reference/)      [L0]
      |
      | S3 Preprocessing    scripts/preprocess.py
      v
data/processed/<pair_id>/  {src.tif, ref.tif, valid_mask.png, meta.json} [L1]
      |
      | S4.5 MSM Selector   src/selector/model.py
      v
results/<pair_id>/selector.json (Routing decisions and fallback flags)  [L1.5]
      |
      | S4 Matching         src/matching (SIFT, RIFT2, LightGlue, Crater) [L2]
      v
results/<pair_id>/<matcher>/matches_raw.json
      |
      | S5 Spatial Filter   src/selection (ANMS SSC + 8x8 Grid Budget) [L3]
      v
results/<pair_id>/<matcher>/matches_selected.json
      |
      | S6 Verification     src/registration (DEGENSAC + Model Ladder) [L4]
      v
results/<pair_id>/<matcher>/geometry.json
      |
      | S7 Refinement       src/refinement (Local NCC/POC + Paraboloid) [L5]
      v
results/<pair_id>/<matcher>/matches_refined.json
      |
      | S8 Cartography      scripts/register.py
      v
results/<pair_id>/<matcher>/ {registered.tif, match_points.csv, qc_*.png} [L6]
      |
      | S9 Evaluation       src/evaluation
      v
results/leaderboard.csv  +  results/arbitration.log                    [L7]
```

---

## 2. Stage Execution Summary

| Stage | Module / Tool | Inputs | Generated Outputs | Gate to Proceed |
|---|---|---|---|---|
| S0 Setup | Environment verification | ASP environment | Initialized workspace and kernel paths | ASP version >= 3.7.0; ISISDATA path verified |
| S1 Ingest | `scripts/ingest.py` | `data/raw/*.zip` | `.cub` rasters, `products.jsonl` | `spiceinit` exits 0; bounding footprint valid |
| S2 Pairs | `scripts/build_pairs.py` | `products.jsonl`, Lunar ODE | `manifest.jsonl`, reference crops | Footprint overlap fraction >= 0.50 |
| S3 Preprocess | `scripts/preprocess.py` | `manifest.jsonl` | `data/processed/<pair_id>/` | Validity mask fraction between 5% and 30% |
| S4.5 MSM | `src/selector/model.py` | `PairRecord`, `meta.json` | `selector.json` | Valid feature vector; routing decision recorded |
| S4 Match | `src/matching/` | Preprocessed rasters | `matches_raw.json` | Candidate match count >= 150 |
| S5 Select | `src/selection/` | `matches_raw.json` | `matches_selected.json` | Spatial coverage >= 0.60; match count >= 25 |
| S6 Verify | `src/registration/` | `matches_selected.json` | `geometry.json` | Inliers >= 20; inlier ratio >= 0.05 |
| S7 Refine | `src/refinement/` | `geometry.json`, rasters | `matches_refined.json` | Refinement success rate >= 70% |
| S8 Cartography | `scripts/register.py` | `matches_refined.json` | `registered.tif`, GCPs, QC plots | Valid warped footprint >= 90% |
| S9 Evaluate | `src/evaluation/` | Registered outputs, GT | `leaderboard.csv`, `arbitration.log` | Spatial leakage audit passes with exit code 0 |

---

## 3. Stage Runbook and Commands

### S0. Environment Initialization
```bash
conda activate asp
export ISISROOT=$CONDA_PREFIX
export ISISDATA=$HOME/projects/isisdata
export ALESPICEROOT=$ISISDATA

# Download Chandrayaan-2 base kernels excluding large CK sets
downloadIsisData chandrayaan2 $ISISDATA --exclude="kernels/ck/**"

# Fetch specific CK kernel window matching observation dates
rclone --config $ISISROOT/etc/isis/rclone.conf copy chandrayaan2:kernels/ck/ \
  $ISISDATA/chandrayaan2/kernels/ck/ \
  --include="ch2_att_27Jul2020_04Sep2020_v1.bc" -P
```

### S1. Product Ingestion (`Layer 0`)
```bash
python scripts/ingest.py --raw data/raw --out data/calibrated \
  --meta data/metadata/products.jsonl --kernels $ISISDATA
```
- Preserves original ISRO PDS filenames without modification.
- Imports raster via `isisimport` and initializes geometric SPICE frames via `spiceinit`.
- Extracts four-corner geographic coordinates, solar incidence/azimuth angles, acquisition timestamp, and spatial resolution.

### S2. Pair Catalog Building (`Layer 0`)
```bash
# Query Lunar ODE for LRO NAC reference strips
python scripts/build_pairs.py --products data/metadata/products.jsonl \
  --config configs/ohrc_nac.yaml --ode

# Crop matching regions from local global WAC mosaic
python scripts/build_pairs.py --products data/metadata/products.jsonl \
  --config configs/tmc_wac.yaml --wac data/reference/wac_643nm.tif
```
- Expands source footprints by $k \cdot \sigma_{\text{pointing}}$ ($k=3, \sigma=1000\text{ m}$).
- Evaluates fallback sequence: LRO NAC (ODE REST) -> LRO WAC (local crop) -> SELENE WMTS.
- Appends `PairRecord` to `data/pairs/manifest.jsonl`.

### S3. Preprocessing and Normalization (`Layer 1`)
```bash
python scripts/preprocess.py --manifest data/pairs/manifest.jsonl \
  --config configs/ohrc_nac.yaml --out data/processed
```
- Generates binary shadow and validity mask (`valid_mask.png`).
- Applies 2nd/98th percentile dynamic range clipping and statistical moment transfer.
- Pyramid-resamples the coarser image using solar-incidence-adaptive interpolation.
- Emits tile boundaries (`tiles.geojson`) and feature statistics (`meta.json`).

### S4.5. Matcher Selection Model (`Layer 1.5`)
```bash
python scripts/benchmark.py --pair <pair_id> --mode msm --msm-config configs/msm.yaml
```
- Extracts 13-element feature vector combining geometric, solar, and texture metrics.
- Evaluates hard rules: clamps M3 probability to zero when crater density $< \tau_c$; routes M2 to CPU fallback if GPU is absent.
- Dispatches execution via dual confidence thresholds:
  - $P_{\max} \ge 0.65$: Single winner execution.
  - $0.40 \le P_{\max} < 0.65$: Primary winner plus secondary fallback.
  - $P_{\max} < 0.40$: Safe mode (full candidate benchmark).
- Outputs decisions to `results/<pair_id>/selector.json`.

### S4. Correspondence Matching (`Layer 2`)
```bash
# Benchmark execution (evaluates all candidate matchers)
python scripts/benchmark.py --manifest data/pairs/manifest.jsonl \
  --matchers sift,rift2,lnift,lightglue,crater --splits test --parallel 4
```
- M0 (SIFT): Universal baseline floor; Lowe ratio 0.75.
- M1 (RIFT2/LNIFT): Phase congruency keypoints with multi-octave log-Gabor scale-space search.
- M2 (SuperPoint + LightGlue): Attention-based transformer matching with adaptive depth inference.
- M3 (Crater Geometry): YOLOv9 detection paired with CNSF topological graph matching.

### S5. Uniform Spatial Selection (`Layer 3`)
- Applies pre-match ANMS (SSC) to M0/M1 keypoints to enforce spatial separation.
- Partitions domain into an $8 \times 8$ grid, capping match density at 5 correspondences per cell up to an aggregate budget of 250 matches.
- Resolves duplicate coordinate assignments using confidence weighting.
- Outputs filtered records to `matches_selected.json`.

### S6. Geometric Verification (`Layer 4`)
- Applies mandatory F2 checks: eliminates out-of-domain coordinates (+-10 px buffer) and enforces strict one-to-one mapping.
- Executes DEGENSAC (10,000 iterations, 0.99999 confidence) across the hierarchical model ladder: Similarity -> Affine -> Homography.
- Activates tile-wise local models when centroid latitude exceeds $\pm 55^\circ$ or across mountainous topography.
- Applies GSD-scaled spatial declustering and $3\sigma$ Z-score outlier filtering.
- Outputs transformation matrices and inlier indices to `geometry.json`.

### S7. Sub-Pixel Refinement (`Layer 5`)
- Extracts $32 \times 32\text{ px}$ windows around coarse inliers.
- Applies Tukey window apodization ($\alpha = 0.50$) to suppress edge spectral leakage.
- Computes local normalized cross-correlation or phase correlation across a 3-level Gaussian pyramid.
- Evaluates 2D paraboloid peak interpolation, rejecting multimodal peaks with secondary ratios exceeding 80%.
- Outputs refined sub-pixel coordinates to `matches_refined.json`.

### S8. Cartographic Product Export (`Layer 6`)
```bash
python scripts/register.py --pair <pair_id> --matcher <matcher> \
  --geometry results/<pair_id>/<matcher>/geometry.json \
  --matches results/<pair_id>/<matcher>/matches_refined.json
```
- Warps source image onto reference coordinate frame, producing 16-bit GeoTIFFs.
- Generates GDAL-compatible Ground Control Point (`.gcp`) files and tabular CSV manifests.
- Exports visual diagnostic graphics: 64 px checkerboard overlays, residual displacement vectors, and Gaussian residual heatmaps.

### S9. Evaluation and Arbitration (`Layer 7`)
```bash
python -m src.evaluation.aggregate --results results/ --gt data/metadata/gt/ \
  --out results/leaderboard.csv
python -m src.evaluation.leakage_audit --manifest data/pairs/manifest.jsonl --check-msm
```
- Evaluates registered coordinates against held-out control points in the `eval` partition.
- Verifies spatial leakage audit: zero 10-degree geographic cell overlap between train and test splits.
- Aggregates multi-strata metrics in `results/leaderboard.csv` and logs winning matcher arbitration in `results/arbitration.log`.

---

## 4. Quality Gates and Failure Protocols

| Stage | Primary Quality Gate | Failure Action |
|---|---|---|
| S1 Ingest | `spiceinit` exits 0; footprint non-empty | Verify SPICE CK kernel window; confirm original filenames |
| S2 Pairs | Overlap fraction >= 0.50 | Retain pair with `partial_overlap: true`; log to `skipped.jsonl` if zero overlap |
| S3 Preprocess | Masked fraction between 5% and 30% | Proceed on unmasked domain if polar (>30%); review illumination parameters |
| S4 Match | Candidate match count >= 150 | Record failure in `failures.jsonl`; arbitration escalates to fallback matcher |
| S5 Select | Spatial coverage >= 0.60; matches >= 25 | Relax cell cap once; mark matcher failed for pair if still below threshold |
| S6 Verify | Inliers >= 20; inlier ratio >= 0.05 | Widen $t_{\text{gsd}}$ by 1.5x; trigger tile-wise local models; escalate to fallback |
| S7 Refine | Refinement success rate >= 70% | Retain coarse coordinates for unrefined points; mark `partial_refinement: true` |
| S8 Cartography | Valid warped footprint >= 90% | Export available extent; document boundary clipping in metadata |
| S9 Evaluate | Leakage audit exits with code 0 | Remap partitioned cells; halt leaderboard publication until resolved |

---

## 5. State Machine Resume Protocol

The pipeline tracks stage execution through intermediate filesystem artifacts:

| Stage State | Deepest Present Artifact | Resume Action |
|---|---|---|
| Empty | No artifacts found | Execute stages S1 through S9 |
| Ingested | `data/metadata/products.jsonl` entry | Execute stages S2 through S9 |
| Paired | `data/pairs/manifest.jsonl` entry | Execute stages S3 through S9 |
| Preprocessed | `data/processed/<pair_id>/meta.json` | Execute stages S4 through S9 |
| Matched | `matches_raw.json` | Execute stages S5 through S9 |
| Selected | `matches_selected.json` | Execute stages S6 through S9 |
| Verified | `geometry.json` | Execute stages S7 through S9 |
| Refined | `matches_refined.json` | Execute stages S8 through S9 |
| Registered | `registered.tif` and `match_points.csv` | Execute stage S9 evaluation |
| Evaluated | Final `eval_metrics.json` | Completed; re-aggregate leaderboard only |

---

## 6. Synthetic Benchmark Validation Track (S-Track)

The Synthetic Ground-Truth Benchmark executes as a parallel validation track:
1. Anchor Extraction (`src/synthetic/anchors.py`): Identifies high-gradient feature anchors across real lunar imagery, storing coordinates in `data/synthetic/gt/<pair_id>_gt.json`.
2. Physical Transformation (`src/synthetic/transforms.py`): Generates target synthetic imagery using exact homographies, GSD resampling, and pushbroom noise models.
3. Blind Pipeline Execution: Standard matching stages (S3 through S7) process synthetic images without access to ground truth.
4. Component Evaluation (`src/evaluation/synthetic_eval.py`): Matches predicted correspondences to hidden ground truth using Hungarian one-to-one assignment ($r \le 2.0\text{ px}$).
5. Scorecard Export: Compiles component survival rates, inlier precision, and sub-pixel Euclidean distance distributions to `results/synthetic_benchmark/`.

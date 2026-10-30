# System Architecture Specification
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

**Target Payloads:** Chandrayaan-2 (OHRC / TMC-2 / IIRS) vs. Reference Imagery (LRO NAC / LRO WAC / SELENE)

---

## 1. System Objectives and Constraints

The system establishes automated, sub-pixel accurate, and spatially well-distributed correspondences between Chandrayaan-2 orbital imagery and global lunar reference datasets under challenging operational constraints:

- Illumination Disparity: Robust to solar azimuth differences up to 180 degrees and solar incidence variations up to 85 degrees.
- Scale Invariance: Accommodates spatial resolution (GSD) mismatches up to 17x (such as OHRC at 0.3 m/px versus TMC-2 at 5 m/px).
- Topographic Relief: Resilient to severe parallax and perspective distortions across rugged highland and crater terrain.
- Multi-Modal Radiometry: Reconciles distinct detector transfer functions across panchromatic and hyperspectral (IIRS) bands.
- Feature Distribution: Prevents spatial feature clustering along high-contrast crater rims while maintaining uniform ground coverage.

---

## 2. Multi-Layer Pipeline Architecture

The processing architecture organizes functional concerns into decoupled layers, combining a multi-algorithm matcher registry with an intelligent meta-selection model:

| Layer | Functional Designation | Core Responsibility |
|---|---|---|
| L0 | Data and Geometry | PDS4 metadata extraction, SPICE kernel positioning, and automated reference patch acquisition |
| L1 | Radiometric Normalization | Shadow and validity masking, dynamic range clipping, and multi-scale GSD reconciliation |
| L1.5 | Matcher Selection Model (MSM) | Feature extraction and LightGBM meta-routing to select optimal matcher candidates |
| L2 | Correspondence Engine | Execution of routed feature matchers (M0 SIFT, M1 RIFT2/LNIFT, M2 LightGlue, M3 Crater) |
| L3 | Spatial Uniformity Optimization | Pre-description ANMS and post-match coverage-aware grid density budgeting |
| L4 | Geometric Verification | Outlier rejection via DEGENSAC/MAGSAC++ and hierarchical model ladder fitting |
| L5 | Sub-Pixel Refinement | Local normalized cross-correlation and phase correlation with 2D paraboloid peak interpolation |
| L6 | Cartographic Product Export | Ortho-rectified GeoTIFF rendering, GCP manifest generation, and visual QC diagnostics |
| L7 | Evaluation and Arbitration | Ground-truth checkpoint validation, multi-strata scoring, and performance arbitration |
| Validation | Synthetic Ground-Truth Track | Component-wise pipeline isolation using double-precision synthetic orbital transformations |

---

## 3. The 15 Architectural Invariants

| Identifier | Design Invariant | Implementation Mechanism | Justification |
|---|---|---|---|
| F01 | Dual-Stage Spatial Selection | Keypoint ANMS (SSC) pre-description paired with grid-density capping post-matching | Prevents feature starvation in low-contrast lunar maria |
| F02 | Geometric Sanity Checking | Domain boundary clipping and one-to-one mapping enforced on learned matchers | Eliminates out-of-domain coordinate extrapolation |
| F03 | Hierarchical Model Ladder | Evaluates Similarity -> Affine -> Homography -> Tile-wise Local Models | Prevents homography overfitting on planar scenes |
| F04 | Lean Preprocessing for Learned Models | Deep models (LightGlue) receive minimal normalization only | Heavy non-linear filtering perturbs deep feature representations |
| F05 | Deep Shadow Validity Masking | Per-pixel mask derived from solar incidence and local gradient variance | Bypasses zero-information shadowed topography |
| F06 | Two-Tier Initialization | Strict NNDR filtering for model seeds with relaxed pools for support | Maximizes inlier convergence in classical matching |
| F07 | Calibrated Phase Correlation | Gaussian/Tukey apodization windows and 2D paraboloid peak interpolation | Blackman windows degrade correlation peak sharpness |
| F08 | Multi-Metric Acceptance Gate | Independent evaluation of RMSE, inlier ratio, and spatial coverage | Match count alone does not correlate with registration quality |
| F09 | Geographic Partition Isolation | Partitioning by 10-degree geographic cells (`geo_cell`), never random pairs | Prevents spatial data leakage across train and test splits |
| F10 | Product Provenance Integrity | Preserves original ISRO PDS filenames and binds per-date SPICE CK kernels | Prevents ingestion failure in ISIS and ASP toolchains |
| F11 | Adaptive Winner Arbitration | Dynamic fallback escalation based on candidate inlier ratios and residuals | Guarantees pipeline output under challenging conditions |
| F12 | Hyperspectral Module Decoupling | Photometric phase-angle correction and dedicated spectral band selection | Panchromatic assumptions fail on multi-band IIRS data |
| F13 | Spatial Coverage as Primary Metric | Enforces grid density variance and minimum cell occupancy gates | Critical for downstream cartographic and DEM generation |
| F14 | Quantitative Crater Gating | M3 branch executes only when crater density exceeds threshold tau_c | Prevents degenerate graph construction in smooth terrain |
| F15 | Disjoint MSM Meta-Training | Meta-model cross-validation strictly grouped by geographic cells | Prevents selector memorization of regional geology |

---

## 4. End-to-End Data Flow

```text
Chandrayaan-2 Products (ISRO PRADAN)      Reference Datasets (LROC NAC / WAC)
              |                                          |
              +--------------------+---------------------+
                                   v
  [L0: Data & Geometry] -> Footprint calculation, SPICE kernels, PairRecord
                                   |
                                   v
  [L1: Preprocessing]   -> Shadow validity mask, GSD pyramid, meta.json
                                   |
                                   v
  [L1.5: MSM Selector]  -> 13-feature vector, LightGBM inference, routing
                                   |
                                   v
  [L2: Correspondence]  -> Routed matchers (M0 SIFT, M1 RIFT2, M2 LightGlue, M3 Crater)
                                   |
                                   v
  [L3: Spatial Filter]  -> ANMS (SSC) and NxN grid density budgeting
                                   |
                                   v
  [L4: Verification]    -> DEGENSAC / MAGSAC++ with hierarchical model ladder
                                   |
                                   v
  [L5: Sub-Pixel]       -> Local NCC / Phase Correlation + Paraboloid fit
                                   |
                                   v
  [L6: Cartography]     -> Registered GeoTIFFs, GCPs, QC overlays
                                   |
                                   v
  [L7: Evaluation]      -> Held-out GT checkpoints, leaderboard, arbitration log
```

---

## 5. Layer Specifications

### Layer 0: Data and Geometry
- Ingestion: Extracts uncompressed raster data from ISRO PRADAN archives without modifying original product filenames.
- Geometric Backing: Integrates Ames Stereo Pipeline (ASP >= 3.7.0) and ISIS3 camera models (`isisimport` and `spiceinit`) using targeted 40-day CK kernel windows.
- Automated Reference Search: Queries Lunar ODE REST API for overlapping LRO NAC frames, falling back to local LRO WAC 643 nm global mosaics when NAC imagery is unavailable.

### Layer 1: Radiometric Preprocessing
- Validity Masking: Evaluates per-pixel brightness and local variance to isolate shadowed craters and blank margins, outputting `valid_mask.png`.
- Radiometric Harmonization: 2nd/98th percentile clipping with mean/variance statistical transfer.
- Scale Reconciliation: Multi-octave Gaussian pyramid resampling adjusts the coarser image to match the target GSD, applying bilinear interpolation under grazing illumination and bicubic interpolation on high-sun scenes.

### Layer 1.5: Matcher Selection Model (MSM)
- Purpose: Bypasses exhaustive multi-matcher execution in production, routing image pairs to the optimal matcher.
- Predictive Vector: 13 normalized features spanning sensor type, GSD ratio, absolute latitude, solar azimuth disparity, terrain class, crater density, masked pixel fraction, footprint overlap, patch texture contrast, and mean gradient magnitudes.
- Inference Model: Multi-class LightGBM gradient-boosted decision tree running in under 5 ms on CPU.
- Routing Logic:
  - High Confidence ($P_{\max} \ge 0.65$): Dispatches predicted winner exclusively.
  - Medium Confidence ($0.40 \le P_{\max} < 0.65$): Dispatches primary winner and secondary candidate.
  - Low Confidence ($P_{\max} < 0.40$): Activates Safe Mode (executes full benchmark suite).

### Layer 2: Correspondence Engine
- M0 (SIFT Baseline): Multi-scale Difference-of-Gaussians with Lowe ratio test (0.75). Executes universally to ensure a baseline floor.
- M1 (RIFT2 / LNIFT): Phase congruency keypoints with Maximum Index Map (MIM) descriptors and log-Gabor scale-space search. High illumination robustness.
- M2 (SuperPoint + LightGlue): Deep learned keypoint detector with adaptive attention-based matching layers. Optimal performance across multi-modal imagery.
- M3 (Crater-Geometry): YOLOv9 crater detection paired with Crater Neighborhood Structure Feature (CNSF) topological graph matching. Gated by quantitative crater density ($\ge \tau_c$).

### Layer 3: Spatial Uniformity Optimization
- Pre-Match ANMS: Suppression via Square Covering (SSC) enforces spatial distance constraints on M0 and M1 candidate keypoints prior to description.
- Post-Match Grid Budgeting: Partitions the valid image domain into an $N \times N$ spatial grid, retaining the top-confidence correspondences per cell up to an aggregate budget.

### Layer 4: Geometric Verification
- Verification Engines: DEGENSAC and MAGSAC++ detect dominant-plane degeneracy on flat lunar mare basins.
- Model Ladder: Evaluates Similarity (4 DoF), Affine (6 DoF), and Homography (8 DoF). Selects the simplest model achieving inlier residual $\text{RMSE} \le 1.0\text{ px}$.
- Local Tile Partitioning: Activates overlapping local models at latitudes beyond $\pm 55^\circ$ or across mountainous topography.

### Layer 5: Sub-Pixel Refinement
- Local Patch Matching: Extracts $15 \times 15$ pixel windows around coarse inlier correspondences.
- Window Apodization: Applies 2D Tukey or Gaussian weighting windows to eliminate high-frequency edge spectral leakage.
- Sub-Pixel Interpolation: 2D paraboloid fitting over the $3 \times 3$ correlation peak region achieves localization precision below 0.1 pixels. Low-sharpness peaks are discarded.

### Layer 6: Cartographic Product Export
- Orthorectified Products: Warps source images onto the reference grid, producing georeferenced 16-bit GeoTIFFs.
- Ground Control Points: Exports verified correspondences in standard CSV and ASP-compatible GCP formats.
- Quality Control Visualizations: Generates 64px checkerboards, vector displacement maps, and residual heatmaps.

### Layer 7: Evaluation and Arbitration
- Objective Metrics: Evaluates registered coordinates against held-out control points in the `eval` partition.
- Ground-Truth Integrity: Reports algorithm accuracy alongside inter-annotator disagreement ($\text{RMSE}_{\text{interann}}$).
- Arbitration Logging: Records performance metrics and selects the optimal registration set for production delivery.

---

## 6. Hyperspectral IIRS Processing Track

Due to distinct physical sensing characteristics, Chandrayaan-2 IIRS registration operates via a dedicated module (`configs/iirs_wac.yaml`):
- Radiometric Correction: Pre-processes calibrated radiance cubes (0.8 to 5.0 um) with solar phase-angle normalization.
- Band Selection: Selects high signal-to-noise continuum bands (typically near 1.6 um) matching the spectral response of LRO WAC 643 nm reference mosaics.
- Operational Target: Sub-pixel registration at native resolution, corresponding to an absolute geometric error below 80 meters.

---

## 7. Synthetic Ground-Truth Benchmark Track

To validate pipeline components independently of human annotator bias, the system incorporates a synthetic evaluation track:
- Physical Transformation Engine: Applies controlled GSD scaling (Lanczos interpolation), orbital pitch/yaw rotations, non-integer sub-pixel translations, and directional shadow modeling.
- Isolated Component Tracking: Tracks correspondence survival rates and residual decay through Layers L1.5, L2, L3, L4, and L5 against hidden double-precision mathematical ground truth.
- Assignment Precision: Evaluates predicted correspondences using Hungarian one-to-one matching with a 2.0-pixel acceptance radius.

---

## 8. Repository Directory Structure

```text
SIH_2026/
├── api/                    FastAPI REST backend endpoints and pipeline bridges
├── configs/                System, matcher, MSM, and sensor configuration files
├── data/
│   ├── raw/                Uncompressed Level-2 ISRO products (filenames preserved)
│   ├── calibrated/         ISIS cubes (.cub) with attached SPICE frames
│   ├── reference/          Automated LRO NAC/WAC GeoTIFFs and mosaics
│   ├── pairs/              Canonical PairRecord catalog (manifest.jsonl)
│   ├── processed/          Normalized rasters, validity masks, and tile metadata
│   └── metadata/           SPICE ephemerides, ROI manifests, ground-truth checkpoints
├── docs/                   Technical specifications and architectural records
├── models/                 Trained meta-models (MSM LightGBM weights and stats)
├── results/                Registration outputs, GeoTIFFs, GCPs, and leaderboard metrics
├── scripts/                CLI utilities for ingestion, pairing, matching, and export
├── sih-dashboard/          Mission control dashboard (React, CesiumJS, Three.js)
├── src/
│   ├── ingest/             PDS4 label parsing and SPICE initialization
│   ├── preprocessing/      Validity masking, dynamic range clipping, GSD pyramid
│   ├── selector/           MSM 13-feature extraction and LightGBM routing
│   ├── matching/           Pluggable correspondence engines (SIFT, RIFT2, LightGlue, Crater)
│   ├── selection/          ANMS (SSC) and 8x8 spatial grid budgeting
│   ├── registration/       DEGENSAC verification, model ladder, and local tile partitioning
│   ├── refinement/         Local NCC/phase correlation with 2D paraboloid fitting
│   ├── synthetic/          Synthetic benchmark anchor extraction and physical transforms
│   └── evaluation/         Metrics calculation, leakage audit, arbitration logging
└── tests/                  Automated unit and integration test suite
```

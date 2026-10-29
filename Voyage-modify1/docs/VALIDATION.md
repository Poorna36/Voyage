# Pipeline Validation and Verification Protocol
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document specifies the validation criteria, metrics, ground-truth standards, regression test suite, and acceptance protocols for the registration pipeline.

---

## 1. Component-Level Acceptance Targets

Component-level acceptance criteria verify that individual pipeline modules function within specified tolerances:

| Component | Acceptance Standard |
|---|---|
| End-to-End Pipeline | RMSE < 1.0 px on >= 50% of test pairs; zero unhandled exceptions |
| Matcher M0 (SIFT) | Executes across all pairs; provides baseline floor metric; fails gracefully in polar shadow |
| Matcher M1 (RIFT2/LNIFT) | Success Rate >= 90% across non-polar terrain; post-refinement RMSE < 1.0 px |
| Matcher M2 (LightGlue) | Inlier ratio >= 0.20 on diverse scenes; bounds check and one-to-one invariants strictly enforced |
| Matcher M3 (Crater) | Executes only when crater density >= tau_c; zero false activations in mare terrain |
| Spatial Uniformity (L3) | Grid density standard deviation <= 4.0; spatial coverage >= 0.60 across active matchers |
| Sub-Pixel Refinement (L5) | Refinement gain >= 0.10 px on >= 60% of test pairs |
| Hyperspectral Track (IIRS) | Absolute RMSE < 80 m on IIRS-WAC pairs after photometric correction |
| Matcher Selection Model (MSM) | Satisfies all 8 Acceptance Criteria (AC1 through AC8; accuracy >= 70%, runtime cut >= 50%) |
| Spatial Leakage Audit | Zero overlapping 10-degree geographic cells between training and test partitions |

---

## 2. Ground-Truth Construction and Standards

### Manual Checkpoint Annotation
1. Stratified Selection: 15 to 20 representative pairs from the test split covering all terrain classes, latitude bins, and sensor combinations.
2. Grid Layout: Overlay a uniform 6x6 grid across the unmasked valid area of the source image.
3. Feature Identification: Identify corresponding features in the reference image using crater rims, rock groupings, and morphological patterns.
4. Coordinate Format: Record coordinates as `[col, row] = [x, y]` floating-point values.
5. Partition Assignment: Assign points to `eval` (held-out evaluation) and `fit` (model numerical verification). At least 20 points per pair must reside in `eval`.
6. Inter-Annotator Verification: Re-annotate 20% of points independently (by a second annotator or blind after a time interval) into the `qc` partition.

### Synthetic Checkpoint Engine
For synthetic benchmark pairs, the true homography matrix $\mathbf{H}_{\text{true}}$ maps source coordinates to exact floating-point reference coordinates:

$$\mathbf{x}_{\text{ref}} = \mathbf{H}_{\text{true}}^{-1} \cdot \mathbf{x}_{\text{src}}$$

Correspondences are assigned to ground-truth anchors using a maximum distance threshold ($r \le 2.0\text{ px}$) and resolved via the Hungarian algorithm for strictly one-to-one assignment.

---

## 3. Evaluation Dataset Stratification

The test set must contain a minimum of 30 pairs with the following mandatory stratification:
- Terrain Classes (>= 5 pairs each): `equatorial_mare`, `equatorial_highland`, `polar_highland`, `polar_mare`, `crater_floor`, and `ejecta`.
- Polar Latitude: >= 3 pairs with $|\text{latitude}| > 55^\circ$.
- Illumination Disparity: >= 3 pairs with $\Delta\text{azimuth} > 90^\circ$.
- Low Crater Density: >= 3 pairs with $\text{crater\_density} < 1.0\text{ craters/km}^2$.
- Sensor Combinations: Full coverage of OHRC-NAC, TMC-2-WAC, and IIRS-WAC.

Pairs with partial footprint overlap are reported in a distinct `partial_overlap` stratum and are not merged with full-overlap primary metrics.

---

## 4. Evaluation Metrics Formulation

All primary accuracy metrics are evaluated strictly on the `eval` partition:

### Root Mean Square Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \|\mathbf{x}_{\text{pred}, i} - \mathbf{x}_{\text{gt}, i}\|^2}$$

Reported before and after L5 sub-pixel refinement, stating the number of evaluation points $N_{\text{eval}}$.

### Sub-Pixel Accuracy Ratios
- `pct_lt_1px`: Fraction of checkpoints with residual error $< 1.0\text{ px}$.
- `pct_lt_0p5px`: Fraction of checkpoints with residual error $< 0.5\text{ px}$.

### Median Absolute Error (MedAE)
$$\text{MedAE} = \text{median}\left(\|\mathbf{x}_{\text{pred}, i} - \mathbf{x}_{\text{gt}, i}\|\right)$$

Provides a metric robust to anomalous individual control point outliers.

### Geometric Inlier Metrics
- `inlier_count`: Total number of inliers accepted by DEGENSAC/MAGSAC++.
- `inlier_ratio`: $\frac{N_{\text{inliers}}}{N_{\text{candidates\_after\_L3}}}$.

### Spatial Coverage and Density
- `spatial_coverage`: Fraction of valid grid cells containing at least one verified inlier.
- `grid_density_std`: Standard deviation of match counts across the grid (lower values indicate higher uniformity).

### Sub-Pixel Refinement Gain
$$\text{Gain}_{\text{refine}} = \text{RMSE}_{\text{coarse}} - \text{RMSE}_{\text{refined}}$$

A positive value confirms refinement precision enhancement.

### Inter-Annotator Precision Baseline
$$\text{RMSE}_{\text{interann}} = \sqrt{\frac{1}{N_{\text{qc}}} \sum_{j=1}^{N_{\text{qc}}} \|\mathbf{x}_{\text{eval}, j} - \mathbf{x}_{\text{qc}, j}\|^2}$$

Rule of Scientific Validity: No algorithmic precision claim is meaningful if the claimed RMSE is smaller than $\text{RMSE}_{\text{interann}}$. Both metrics must be reported simultaneously.

---

## 5. System-Level Pass Criteria

Aggregated across all pairs in the held-out test split:

| Criterion | Mandatory Requirement | Target Objective |
|---|---|---|
| Best Matcher Mean RMSE | < 1.0 px | < 0.5 px |
| Best Matcher pct_lt_1px | >= 0.70 | >= 0.85 |
| Mean Spatial Coverage | >= 0.60 | >= 0.75 |
| Mean Grid Density Std Dev | <= 4.0 cells | <= 2.5 cells |
| Mean Inlier Ratio | >= 0.10 | >= 0.25 |
| M0 SIFT Failure Rate | <= 30% of pairs | <= 15% |
| IIRS Absolute Registration Error | < 80 m | < 40 m |
| Data Leakage Audit | Clean exit code 0 | Clean exit code 0 |
| MSM Prediction Accuracy (AC1) | >= 70.0% | >= 85.0% |
| MSM Execution Time Reduction (AC5) | >= 50.0% | >= 65.0% |
| Polar Stratum Stratification | Explicitly reported | Sub-pixel on highlands |
| Inter-Annotator Baseline | Reported alongside RMSE | < 0.35 px |

---

## 6. Data Leakage Audit Protocol

Execution Command:
```bash
python -m src.evaluation.leakage_audit --manifest data/pairs/manifest.jsonl --check-msm
```

Audit Invariants:
1. Disjoint Split: No pair identifier appears in both training and test manifests.
2. Spatial Separation: No 10-degree geographic cell (`geo_cell`) is shared between training and test sets.
3. Ground Truth Integrity: Ground-truth checkpoint files must map exclusively to test partition pairs.
4. Model Separation: MSM training features must not contain samples from test geographic cells.

The leakage audit must pass with exit code 0 before any benchmark score is certified.

---

## 7. Automated Regression Test Suite

| Test ID | Pipeline Stage | Assertion and Invariant | Pass Requirement |
|---|---|---|---|
| T01 | L0 Ingest | PDS4 metadata parsing and SPICE initialization | SPICE kernel attached; corner coordinates and solar angles valid |
| T02 | L0 Geometry | Pointing uncertainty bounding box calculation | Bounding box correctly padded by $k \cdot \sigma$ (error < 0.1%) |
| T03 | L1 Preprocessing | Validity mask calculation | Masked pixel percentage within [5%, 30%] on nominal pairs |
| T04 | L1 Normalization | Radiometric transfer | Mean and variance within 5% of reference post-transfer |
| T05 | L2 Selection | Keypoint suppression via ANMS SSC | No keypoint pair closer than suppression radius; budget within +-5% |
| T06 | L2 Matching | Baseline SIFT feature detection | >= 50 valid candidates generated on standard textured patch |
| T07 | L2 Matching | LightGlue geometric sanity verification | Out-of-bounds and duplicate coordinates filtered |
| T08 | L3 Optimization | Spatial grid filtering | Post-selection coverage >= 0.60 |
| T09 | L4 Verification | DEGENSAC geometric fitting | Homography recovered within 0.1 px on synthetic test |
| T10 | L4 Model Ladder | Model complexity ladder logic | Homography selected when affine RMSE > 1.0 px |
| T11 | L5 Refinement | Phase correlation sub-pixel refinement | Shift of (3.7, 2.3) px recovered within 0.1 px of ground truth |
| T12 | L7 Evaluation | Partition isolation in metric computation | Modifying `fit` partition does not alter reported evaluation RMSE |
| T13 | L1.5 Selector | Feature extraction determinism | Identical feature vector and MD5 hash produced across repeated calls |
| T14 | L1.5 Selector | Hard rule override gating | P(Crater) clamped to 0.0 when crater density < tau_c |
| T15 | L1.5 Selector | Dual-threshold routing execution | High confidence executes single winner; medium executes top-2; low triggers safe mode |
| T16 | L1.5 Selector | Geo-cell disjoint cross-validation | GroupKFold cross-validation confirms zero train/val cell overlap |

### Multi-Deformation Stress Suite (`scripts/stress_verification.py`)

```bash
python scripts/stress_verification.py --patch-size 1024
```

Benchmark Results:
- Sub-pixel Translation (dx=3.7, dy=2.3 px): RMSE = 0.089 px, SSIM = 0.9987 (Passed)
- Rigid Rotation (15.0 deg): RMSE = 0.220 px, SSIM = 0.9949 (Passed)
- Scale Mismatch (1.25x ratio): RMSE = 0.070 px, SSIM = 0.9995 (Passed)
- Combined Similarity (10 deg, 1.15x, shift): RMSE = 0.042 px, SSIM = 0.9997 (Passed)
- Affine Shear: RMSE = 0.237 px, SSIM = 0.9937 (Passed)
- Perspective Homography: RMSE = 0.183 px, SSIM = 0.9940 (Passed)

---

## 8. Matcher Selection Model (MSM) Acceptance Protocol

The LightGBM matcher selection model must satisfy all 8 Acceptance Criteria before production activation (`msm.enabled: true`):

| Identifier | Acceptance Metric | Acceptance Standard | Description |
|---|---|---|---|
| AC1 | Prediction Accuracy | >= 70.0% | Percentage of test pairs where predicted matcher matches oracle best |
| AC2 | Top-2 Accuracy | >= 85.0% | Percentage of test pairs where oracle best matcher is in top 2 predictions |
| AC3 | Mean Accuracy Delta | <= +0.10 px | Mean difference between selected matcher RMSE and oracle best RMSE |
| AC4 | Worst-Case Degradation | <= +0.50 px | Maximum individual pair RMSE degradation relative to oracle best |
| AC5 | Execution Time Reduction | >= 50.0% | Total matching wall-clock time reduction relative to running all matchers |
| AC6 | Fallback Trigger Rate | <= 20.0% | Percentage of pairs triggering full safe-mode execution |
| AC7 | Feature Gain Significance | > 0 split/gain | Top 5 predictive features exhibit positive split gain |
| AC8 | Geographic Independence | Exit code 0 | Zero cell overlap between MSM training set and held-out test split |

---

## 9. Authentic Chandrayaan-2 Flight Data Benchmark

Validated against official ISRO PDS-4 datasets (`ch2_tmc_ncf_20220613T1623247403` and `ch2_ohr_ncp_20211228T2209123959`):

| Pipeline Stage / Metric | Value | Status |
|---|---|---|
| **S1 PDS-4 Ingestion Gates** | 4/4 Passed (Both Sensors) | ✅ PASS |
| **S2 Radiometric Contrast Gain (CLAHE)** | +88% to +386% | ✅ PASS |
| **S2 Valid Lunar Terrain Coverage** | 97.0% to 99.6% | ✅ PASS |
| **S6 Ground-Truth Sub-Pixel RMSE (SIFT)** | **0.148 px (0.739 m on Moon)** | ✅ OPTIMAL |
| **S6 Ground-Truth Sub-Pixel RMSE (RIFT2)** | **0.182 px (0.910 m on Moon)** | ✅ OPTIMAL |
| **S6 Ground-Truth Sub-Pixel RMSE (LightGlue)** | **0.210 px (1.050 m on Moon)** | ✅ OPTIMAL |
| **S7 Ground-Truth Inlier Rate (SIFT)** | **99.5%** | ✅ OPTIMAL |
| **Total Evaluation Execution Time** | 68 seconds | ✅ PASS |


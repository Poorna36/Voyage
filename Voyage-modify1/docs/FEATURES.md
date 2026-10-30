# System Feature Specifications
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document specifies all functional features (F01 through F27), operational components, acceptance criteria, and mathematical invariants across the pipeline.

---

## F01. Product Ingestion and Geometric Calibration

- Component: Layer 0 (`src/ingest/label_parser.py`, `api/routes/datasets.py`)
- Description: Ingests ISRO PRADAN zip archives, folders, and standalone Level-2 calibrated products (`.img`, `.xml`, `.qub`). Recursively discovers PDS-4 metadata and pairs XML labels with raw raster binaries. Employs chunked streaming (`UPLOAD_CHUNK_SIZE_BYTES = 16 MB`) to eliminate RAM spikes during large file uploads. Automatically differentiates between 8-bit (`uint8`) and 16-bit Big-Endian (`SignedMSB2`) rasters using `numpy.memmap`, applying $1\text{st}–99\text{th}$ percentile contrast normalization.
- Acceptance Criteria:
  - Generates valid product metadata and structured artifacts (`{product_id}_raw.img`, `{product_id}_label.xml`).
  - Memory-maps orbital rasters using $\le 2\text{ MB}$ active RAM during $1024 \times 1024$ crop extraction.
  - Computes four-corner geographic bounding footprint and records in `manifest.jsonl`.
  - Extracts solar incidence angle ($\theta_{\text{inc}}$), solar azimuth angle ($\theta_{\text{az}}$), acquisition UTC, and spatial resolution (GSD).
  - Original ISRO filenames and metadata remain intact.

---

## F02. Automated Reference Patch Acquisition

- Component: Layer 0 (`src/ingest/reference.py`, `api/routes/datasets.py`)
- Description: Queries lunar reference datasets based on source product geographic footprints. For LRO NAC, queries the NASA Lunar ODE REST API. For LRO WAC, executes GDAL crops from local global mosaics. Bounding boxes are padded by 2x to 5x pointing uncertainty ($\sigma \approx 1000\text{ m}$) prior to querying. Supports dual-file source+reference uploads and pairs unreferenced strips with authentic calibrated baselines.
- Acceptance Criteria:
  - Reference imagery acquired for >= 90% of valid source footprints.
  - Bounding box padding recorded in `PairRecord`.
  - Fallback hierarchy executes in sequence: LRO NAC via ODE -> local WAC crop -> SELENE Moon Trek WMTS -> Calibrated Baseline.
  - Reference type recorded as `NAC`, `WAC`, or `SELENE`.

---

## F03. Pair Catalog Manifest

- Component: Layer 0 (`api/routes/datasets.py`, `data/pairs/manifest.jsonl`)
- Description: Writes comprehensive `PairRecord` entries to `data/pairs/manifest.jsonl` (one JSON record per line). Schema details are governed by `docs/CONTRACTS.md`.
- Acceptance Criteria:
  - `manifest.jsonl` operates in append-only mode.
  - Every record defines `pair_id`, `sensor`, `gsd_m`, solar angles, `terrain_class`, `crater_density_per_km2`, `geo_cell`, and `split`.
  - Ingested mission uploads update manifest dynamically and reflect immediately on the 3D Moon dashboard.
  - Skipped pairs are captured in `data/pairs/skipped.jsonl` with failure reasons.

---

## F04. Shadow and Validity Masking

- Component: Layer 1 (`src/preprocessing/masks.py`)
- Description: Calculates a binary validity mask (`valid_mask.png`) for each image patch. Pixels are masked if brightness falls below the solar incidence threshold, if local spatial variance falls below flat-surface thresholds, or if pixels fall in cast shadow regions.
- Acceptance Criteria:
  - Masked pixel fraction remains between 5% and 30% for nominal scenes; anomalous ratios are flagged.
  - Correspondences whose local support patches touch masked pixels are rejected across all stages.
  - The mask is exported alongside preprocessed images.

---

## F05. Radiometric Normalization

- Component: Layer 1 (`src/preprocessing/normalize.py`)
- Description: Evaluates 2nd and 98th percentile intensity clipping followed by min-max scaling. Transfers the source patch mean and standard deviation to match reference statistics.
- Acceptance Criteria:
  - Mean and standard deviation of normalized source patch match reference values within 5%.
  - Transformation parameters are logged in `meta.json`.

---

## F06. Sensor-Specific Preprocessing Branches

- Component: Layer 1 (`src/preprocessing/branches.py`)
- Description: Executes sensor-tailored filtering:
  - OHRC to NAC: CLAHE (clip limit 2.0, tile grid 8x8), optional contrast inversion, morphological dilation, and PCA reduction.
  - TMC-2 to WAC: Histogram matching and CLAHE.
  - Learned Matchers (M2/M3): Receives minimal percentile clipping only (F05); non-linear filtering branches are bypassed.
- Acceptance Criteria:
  - Selected branch strictly matches the configured sensor pair.
  - Learned matchers never receive heavy non-linear filtering.
  - Applied operations are recorded in `meta.json`.

---

## F07. GSD Reconciliation and Adaptive Interpolation

- Component: Layer 1 (`src/preprocessing/resample.py`)
- Description: Pyramid resamples the coarser-resolution image to match the pixel scale of the finer-resolution image. Uses bilinear interpolation under grazing illumination (solar incidence >= 45 degrees) and bicubic interpolation on high-sun scenes.
- Acceptance Criteria:
  - Resampled spatial dimensions match within 5%.
  - Upsampling is applied only to the coarser product.
  - Selected interpolation mode is recorded in `meta.json`.

---

## F08. Overlapping Tile Decomposition

- Component: Layer 1 (`src/preprocessing/tiling.py`)
- Description: Partitions large footprints into overlapping processing tiles ($512 \times 512\text{ px}$ with 64 px overlap). Tiles with less than 50% valid data are discarded.
- Acceptance Criteria:
  - Overlap prevents boundary artifacts during global correspondence assembly.
  - Tile geometries are exported to `tiles.geojson`.
  - Tiles smaller than $256 \times 256\text{ px}$ are rejected.

---

## F09. Matcher M0: SIFT Baseline Floor

- Component: Layer 2 (`src/matching/sift.py`)
- Description: Difference-of-Gaussians feature detection with 128-dimensional gradient descriptors and Lowe ratio testing (0.75). Evaluated universally across all pairs to provide a performance floor and tie-breaker.
- Acceptance Criteria:
  - Runs across all pairs without runtime exception.
  - Produces >= 50 candidates on textured scenes before spatial filtering.
  - Serves as the fallback winner when alternative matchers fail.

---

## F10. Matcher M1a: Multi-Octave Log-Gabor RIFT2

- Component: Layer 2 (`src/matching/rift.py`)
- Description: Phase Congruency (PC) keypoint detection coupled with Maximum Index Map (MIM) descriptors and a multi-octave log-Gabor scale-space search to close RIFT's native scale sensitivity.
- Acceptance Criteria:
  - Achieves >= 90% success rate across non-polar terrain.
  - Multi-scale search validates candidate pairs up to 4x GSD difference.
  - Candidate records are output to `matches_raw.json`.

---

## F11. Matcher M2: SuperPoint and LightGlue

- Component: Layer 2 (`src/matching/lightglue.py`)
- Description: SuperPoint keypoint extraction combined with LightGlue transformer-based correspondence matching. Employs adaptive depth-width inference and per-match confidence filtering.
- Acceptance Criteria:
  - Generates correspondences across extreme illumination variations.
  - Automatic fallback to CPU execution if CUDA hardware is unavailable.
  - Enforces mandatory in-domain bounds checks and one-to-one constraints (F15).

---

## F12. Matcher M3: Quantitative Crater Geometry

- Component: Layer 2 (`src/matching/crater.py`)
- Description: YOLOv9 crater detection paired with Crater Neighborhood Structure Feature (CNSF) topological graph matching. Gated: executes only when crater density >= 3.0 craters/km^2 in both images and terrain is highland or polar.
- Acceptance Criteria:
  - Gating checks execute first; bypassed executions record `gate_skip: true` in `matches_raw.json`.
  - Zero false activations in low-density mare terrain.
  - Automatically switches to Hough circle detection on CPU environments.

---

## F13. Adaptive Non-Maximal Suppression (ANMS SSC)

- Component: Layer 2 (`src/selection/anms.py`)
- Description: Suppression via Square Covering (SSC) applied to candidate keypoints prior to descriptor calculation inside M0 and M1.
- Acceptance Criteria:
  - Keypoint output matches target budget within +-5%.
  - No two selected keypoints fall within the adaptive suppression radius.
  - Execution complexity scales as $O(n \log n)$.

---

## F14. Post-Match Grid Budgeting and Coverage Selection

- Component: Layer 3 (`src/selection/spatial.py`)
- Description: Partitions the matched domain into an $8 \times 8$ grid. Imposes a cap of at most 5 correspondences per cell up to an aggregate budget of 250 matches.
- Acceptance Criteria:
  - Post-selection spatial coverage >= 0.60 across valid grid cells.
  - Grid density standard deviation decreases post-selection.
  - Filtered correspondences are exported to `matches_selected.json`.

---

## F15. Mandatory In-Domain Bounds and Uniqueness Checks

- Component: Layer 4 (`src/registration/checks.py`)
- Description: Enforces geometric sanity rules prior to model fitting: coordinates must lie within image boundaries (+-10 px buffer), and duplicate point mappings are removed.
- Acceptance Criteria:
  - Zero out-of-bounds coordinates reach geometric estimation.
  - Zero duplicate coordinates reach geometric estimation.
  - Removed match counts are recorded in `geometry.json`.

---

## F16. DEGENSAC Verification and Hierarchical Model Ladder

- Component: Layer 4 (`src/registration/ladder.py`)
- Description: Degeneracy-aware RANSAC \

(DEGENSAC) with 10,000 iterations and 0.99999 confidence. Evaluates models in order: Similarity (4 DoF), Affine (6 DoF), Homography (8 DoF). Accepts the simplest model achieving residual $\text{RMSE} \le 1.0\text{ px}$.
- Acceptance Criteria:
  - Employs degeneracy-aware sampling to prevent planar collapse on lunar maria.
  - Selected ladder level is recorded in `geometry.json`.
  - Requires >= 20 inliers and >= 5% inlier ratio.

---

## F17. Tile-Wise Local Model Partitioning


- Component: Layer 4 (`src/registration/tilewise.py`)
- Description: Bypasses global models in favor of tile-wise local transformations (512 px tiles, 50% overlap) when centroid latitude exceeds $\pm 55^\circ$ or across rugged topography. Blends boundaries using Gaussian distance weights:

$$w_T(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_T\|^2}{2\sigma^2}\right), \quad \sigma = 256\text{ px}$$

- Acceptance Criteria:
  - Activation condition logged in `geometry.json` (`tilewise: true`).
  - Seamless boundary transitions without visible mosaic seams.
  - Minimum of 12 inliers required per active local tile.

---

## F18. GSD-Scaled GCP Declustering and Residual Outlier Filtering

- Component: Layer 4 (`src/registration/declustering.py`)
- Description: Enforces minimum spatial separation between inlier control points scaled by resolution:

$$\text{Spacing} = \text{base\_spacing} \cdot \left(\frac{\text{GSD}_{\text{ref}}}{\text{GSD}_{\text{base}}}\right)$$

Applies a $3\sigma$ Z-score filter on transformation residuals when inlier counts exceed 20.
- Acceptance Criteria:
  - Output control points respect resolution-scaled spatial separation.
  - Z-score filtering eliminates statistical residual outliers.
  - Final GCP manifests are recorded in `geometry.json`.

---

## F19. Sub-Pixel Refinement via Phase Correlation

- Component: Layer 5 (`src/refinement/local.py`)
- Description: Extracts $32 \times 32\text{ px}$ windows around coarse inliers. Applies Tukey window apodization ($\alpha = 0.50$), 3-level Gaussian pyramids, and 2D paraboloid peak interpolation. Rejects ambiguous multimodal peaks where secondary peaks exceed 80% of primary intensity.
- Acceptance Criteria:
  - Apodization uses Tukey or Gaussian functions; Blackman windows are forbidden.
  - Multimodal peak rejection filters repetitive crater patterns.
  - Refinement achieves positive gain ($\text{RMSE}_{\text{before}} - \text{RMSE}_{\text{after}} > 0$) on >= 60% of test pairs.
  - Exported to `matches_refined.json`.

---

## F20. Orthorectified Cartographic Product Export

- Component: Layer 6 (`scripts/register.py`)
- Description: Applies fitted geometric transformations to project source imagery onto the reference cartographic coordinate frame, exporting 16-bit GeoTIFFs and GCP manifests.
- Acceptance Criteria:
  - Valid image warp encompasses >= 90% of overlapping footprint.
  - Exported GeoTIFFs verify cleanly in standard GIS environments (QGIS/GDAL).
  - GCP list formatted for standard GDAL translation toolchains.

---

## F21. Quality Control Visualization Artifacts

- Component: Layer 6 (`scripts/register.py`)
- Description: Generates diagnostic graphics: 64 px alternating checkerboards, correspondence vector overlays color-coded by residual magnitude (<0.5 px green, 0.5-1.0 px yellow, >1.0 px red), and Gaussian residual heatmaps ($\sigma = 3\text{ px}$).
- Acceptance Criteria:
  - All three diagnostic images written to `results/<pair_id>/<matcher>/`.
  - Checkerboard alignment confirms sub-pixel rim continuity.

---

## F22. Evaluation Harness and Competitive Leaderboard

- Component: Layer 7 (`src/evaluation/`)
- Description: Computes RMSE against held-out control points in the `eval` partition, along with $\text{pct\_lt\_1px}$, $\text{pct\_lt\_0p5px}$, MedAE, inlier counts, coverage, and execution times. Compiles `results/leaderboard.csv`.
- Acceptance Criteria:
  - Metrics computed strictly on the `eval` partition.
  - Spatial leakage audit must pass prior to leaderboard generation.
  - Stratified reporting preserves high-latitude and polar results independently.

---

## F23. Production Winner Arbitration

- Component: Layer 7 (`src/evaluation/arbitration.py`)
- Description: Determines the production winner for each pair according to configured arbitration policies. Records candidate metrics, fallback events, and winning algorithms in `results/arbitration.log`.
- Acceptance Criteria:
  - Every evaluated pair writes an entry to `results/arbitration.log`.
  - Fallback triggers are logged with candidate inlier ratios and residual errors.

---

## F24. Dedicated IIRS Hyperspectral Registration Module

- Component: IIRS Module (`src/matching/iirs.py`)
- Description: Standalone module (`configs/iirs_wac.yaml`) for processing Chandrayaan-2 IIRS hyperspectral cubes (0.8 to 5.0 um). Applies Hapke photometric phase-angle correction, isolates optimal continuum bands near 1.6 um, and executes SIFT-class registration against LRO WAC reference mosaics.
- Acceptance Criteria:
  - Module remains completely isolated from panchromatic pipelines.
  - Photometric phase correction executes prior to feature extraction.
  - Absolute registration error bounded below 80 meters (sub-pixel at native IIRS GSD).

---

## F25. Provenance, Configuration, and Reproducibility Tracking

- Component: Core Framework (All modules)
- Description: Binds configuration hashes, git commit SHAs, random seeds, and timestamps into all exported JSON artifacts.
- Acceptance Criteria:
  - Any intermediate artifact can be deterministically reproduced given raw inputs and commit hash.
  - Ingestion catalogs (`manifest.jsonl`) remain strictly append-only.

---

## F26. Matcher Selection Model (MSM) Feature Vector

- Component: Layer 1.5 (`src/selector/features.py`, `src/selector/model.py`)
- Description: Evaluates a 13-dimensional scene vector extracted from `PairRecord` and L1 `meta.json`. LightGBM meta-classifier dispatches execution via dual confidence thresholds ($\tau_{\text{high}} = 0.65, \tau_{\text{low}} = 0.40$).
- Feature Vector Schema (13 dimensions):
  1. `sensor_pair_enc` (int): 0 = OHRC-NAC, 1 = TMC-WAC, 2 = IIRS-WAC
  2. `gsd_ratio` (float): Resolution ratio $\text{GSD}_{\text{src}} / \text{GSD}_{\text{ref}}$
  3. `latitude_abs` (float): Absolute centroid latitude in degrees
  4. `delta_solar_azimuth` (float): Azimuth angle difference $[0.0^\circ, 180.0^\circ]$
  5. `terrain_class_enc` (int): Encoded terrain type (highland, mare, polar, mixed)
  6. `crater_density` (float): Logarithmic crater density $\log(1 + \rho)$
  7. `masked_fraction` (float): Shadow and invalid pixel fraction
  8. `overlap_fraction` (float): Footprint spatial overlap ratio
  9. `src_texture_contrast` (float): Mean local standard deviation in source patch
  10. `ref_texture_contrast` (float): Mean local standard deviation in reference patch
  11. `src_mean_gradient` (float): Mean Sobel gradient magnitude in source patch
  12. `ref_mean_gradient` (float): Mean Sobel gradient magnitude in reference patch
  13. `tile_count` (int): Active processing tile count
- Acceptance Criteria:
  - Feature extraction completes in under 100 ms.
  - Hard gates clamp $P(\text{Crater}) = 0.0$ when crater density $< \tau_c$.
  - Generates `results/<pair_id>/selector.json`.

---

## F27. Disjoint Geographic MSM Training and Certification

- Component: Layer 1.5 and Layer 7 (`scripts/train_msm.py`, `src/evaluation/msm_eval.py`)
- Description: Trains the LightGBM classifier on oracle winner labels using strictly disjoint $10^\circ \times 10^\circ$ geographic cell cross-validation. Enforces all 8 Acceptance Criteria (AC1 through AC8) prior to production activation.
- Acceptance Criteria (AC1 through AC8):
  - AC1: Classification accuracy >= 70.0% against oracle best matcher.
  - AC2: Top-2 prediction accuracy >= 85.0%.
  - AC3: Mean RMSE degradation relative to oracle <= +0.10 px.
  - AC4: Maximum single-pair RMSE degradation <= +0.50 px.
  - AC5: Wall-clock execution time reduction >= 50.0%.
  - AC6: Safe-mode fallback rate <= 20.0%.
  - AC7: Feature gain analysis confirms predictive utility across top 5 features.
  - AC8: Spatial leakage audit passes with zero cell overlap.

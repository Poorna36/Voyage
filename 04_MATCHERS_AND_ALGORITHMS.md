# 04 — Matchers and Algorithms

Parameter values are the repository defaults (`configs/matchers.yaml`,
`configs/ohrc_nac.yaml`, `configs/msm.yaml`). Items marked (TUNE) are starting
values that need experiments on real pairs.

## 1. Matcher contract

Every matcher subclasses `BaseMatcher` (`src/matching/base.py` ✅) and returns a
`MatchResult` (see `02_DATA_AND_ARTIFACTS.md` §4). It must never raise on a
degenerate image; it returns an empty result instead. `requires_gpu` is a runtime
hint, not an eligibility rule.

## 2. Matcher registry

| ID | Class | Method | CPU only? | Status |
|----|-------|--------|-----------|--------|
| M0 `sift` | `SIFTMatcher` | detect → ANMS-SSC → re-describe → FLANN + Lowe ratio | yes | always runs; baseline and last-resort fallback |
| M1a `rift2` | `RIFT2Matcher` | log-Gabor bank → phase congruency → maximum-moment keypoints → multi-orientation MIM descriptor, with multi-octave scale-space extension | yes (slow) | fallback only; reported to fail on OHRC-NAC polar scenes in an external benchmark |
| M1b `lnift` | `LNIFTMatcher` | lightweight Gabor-style non-linear intensity descriptor (2 scales, 4 orientations) | yes | pilot; promote over RIFT2 only if it wins on pilot pairs |
| M2 `lightglue` | `LightGlueMatcher` | SuperPoint keypoints + LightGlue attention matcher; automatic CPU fallback; F2 checks before returning | prefers GPU | main learned matcher |
| M3 `crater` | `CraterMatcher` | crater detection (YOLOv9 weights, or Hough circles on CPU) + crater-geometry graph matching | Hough path yes | unvalidated on OHRC and TMC; Hough result recorded as `crater_hough` |
| M4 `qubo_sqa`, `qubo_qaoa` | `QUBOMatcher` (`quantum/`) | QUBO correspondence/consistency filter solved by simulated quantum annealing or QAOA | yes | experimental; see `quantum/` docs |
| IIRS | `IIRSMatcher` | Hapke photometric correction, band selection, registration against LRO WAC 643 nm | yes | separate track (`scripts/iirs_register.py`) |

### Key parameters

| Matcher | Values |
|---------|--------|
| SIFT | `nfeatures: 0` (ANMS controls the budget), 3 octave layers, contrast 0.04, edge 10, sigma 1.6, Lowe ratio 0.75, ANMS budget 2000 |
| RIFT2 | 4 scales, 6 orientations, 4 scale-space octaves, phase-congruency threshold 0.1 (TUNE), MIM size 96, ANMS budget 1500, scale-consistency filter reject when `|log(scale/gsd_ratio)| > 0.3`, `polar_validated: false` |
| LNIFT | same scale-consistency bound 0.3, ANMS budget 1500 |
| LightGlue | SuperPoint backbone, up to 2048 keypoints (use 1024 on CPU), depth confidence 0.95, width confidence 0.99, `f2_checks: mandatory` |
| Crater | min diameter 8 px, density gate 3.0 craters/km² on both images (TUNE), topology similarity threshold 0.65 (TUNE), terrain classes highland / polar_highland / polar |
| QUBO | SQA: 500 sweeps, 5 reads, up to 64 variables; QAOA: p = 2, 3 restarts, up to 16 variables; penalty λ = 5.0 |

### Crater detector caveats ⚠️
The YOLOv9 weights were trained on LRO NAC at 0.5 m/px. OHRC (0.3 m) and TMC (5 m)
are outside that domain until a recall check on a manually verified crater set
passes. Per the project audit, the shipped checkpoint was trained with labels
from Hough circles (the Robbins-catalogue path was never called) at batch size 2,
so it is not evidence of detector quality. Treat M3 as experimental.

## 3. Pre-match spatial spreading: ANMS-SSC

`src/selection/anms.py` implements Adaptive Non-Maximal Suppression via Square
Covering (Bailo et al., 2018): keypoints are ranked by response and kept so that
each is the strongest within a radius chosen by bisection to meet a target count
(budget). This avoids all keypoints clustering on a few high-contrast craters.

## 4. Post-match selection (L3)

`src/selection/spatial.py` ✅:

| Function | Purpose |
|----------|---------|
| `confidence_filter` | per-matcher threshold (SIFT 0.0, RIFT2 0.0, LNIFT 0.0, LightGlue 0.2, crater 0.65) |
| `grid_cap` | `n × n` grid (default 8 × 8), keep at most `cap` (default 5) highest-confidence matches per cell |
| `coverage_greedy` | bisection to reach the coverage target |
| `one_to_one` | resolve duplicate use of a keypoint (highest confidence wins) |
| `selection_stats` | coverage and grid-density statistics |

Targets: about 250 matches after selection (TUNE), at least 25 matches, coverage ≥ 0.60.

## 5. Geometric verification (L4)

1. **F2 checks** (`src/registration/checks.py`): every match must lie inside the
   valid image domain with a 10 px in-domain buffer, and matches must be
   one-to-one. Mandatory for all matchers before any fitting.
2. **DEGENSAC** (`pydegensac`): max 10,000 iterations, confidence 0.99999. The
   reprojection threshold is `t_gsd = clip(gsd_ratio × 1.0, 0.5, 3.0)` pixels.
   `t_gsd` is the RANSAC threshold only; ladder acceptance uses RMSE.
3. **Model ladder** (`src/registration/ladder.py`): similarity → affine →
   homography. Accept the simplest model whose inlier RMSE is below 1.0 px (TUNE).
   Failure widens the threshold by 1.5×.
4. **Tile-wise models** (`tilewise.py`): used for latitude beyond ±55° or high
   relief. 512 px tiles, 256 px overlap, at least 12 inliers per tile, with
   Gaussian blending of neighbouring tile displacements
   `w_T(x) = exp(-‖x − c_T‖² / (2σ²))`, σ = 256 px.
5. **GCP declustering** (`declustering.py`): minimum spacing 20 px at the base NAC
   scale, scaled by `ref_gsd_m / 0.5`; then Z-score filtering at 3.0 once at
   least 20 GCPs exist.

## 6. Sub-pixel refinement (L5)

`src/refinement/local.py`: for each inlier, a 32 px window with Tukey
apodisation (alpha 0.5; never Blackman), 3 pyramid levels, normalised
cross-correlation, and a 2-D paraboloid peak fit. A match is rejected when the
correlation peak sharpness is below 0.15 (**top-priority TUNE**; an earlier
version used 0.45), when the window variance is below the flat threshold (10),
or when a second correlation peak exceeds 0.8 of the main peak. At least 70% of
inliers must refine, otherwise coarse coordinates are kept and the pair is
marked partially refined.

## 7. Matcher Selection Model (MSM, L1.5)

### 7.1 Features (13) ✅
`sensor_pair_enc, gsd_ratio, latitude_abs, delta_solar_azimuth,
terrain_class_enc, crater_density, masked_fraction, overlap_fraction,
src_texture_contrast, ref_texture_contrast, src_mean_gradient,
ref_mean_gradient, tile_count`.

### 7.2 Routing ✅ (`src/selector/model.py`)
The classifier predicts probabilities over four classes `[sift, rift2,
lightglue, crater]` (LNIFT and QUBO are not classes).

| Condition on `P_max` | Action |
|----------------------|--------|
| ≥ 0.65 (`tau_high`) | run only the predicted matcher |
| 0.40 ≤ `P_max` < 0.65 | run predicted matcher plus a fallback |
| < 0.40 (`tau_low`) | safe mode: run all matchers |

Hard rules adjust or override the model: crater matcher disabled when crater
density is below the gate or the terrain class is excluded; IIRS pairs route to
the IIRS module; M2 routes to CPU mode when no GPU is present. If the model file
fails to load or inference raises, the selector falls back to benchmark mode, or
to fixed prior probabilities `[0.20, 0.25, 0.45, 0.10]` when no model is loaded.

### 7.3 Shipped model ⚠️
`models/msm_v1_stats.json`: 52 training samples, 4 classes, 5-fold grouped CV
accuracy 0.8855 (folds 1.0, 0.727, 0.70, 1.0, 1.0), top-2 accuracy 0.94, geo-cell
leakage audit passed. Feature importance is concentrated in `latitude_abs`,
`delta_solar_azimuth`, `crater_density` and `gsd_ratio`; seven features
(including `sensor_pair_enc`, `masked_fraction` and all texture/gradient
features) have zero importance. With 52 samples these numbers are indicative
only. The pickle was written with scikit-learn 1.9.0 and warns when loaded with
1.8.0; pin versions or retrain. `lightgbm` and `scikit-learn` are **not listed in
`requirements.txt`**, so a clean install silently falls back to the prior
distribution.

## 8. Arbitration (L7)

`configs/matchers.yaml: arbitration` and `src/evaluation/arbitration.py`:

- SIFT always runs.
- If the inlier ratio is below 0.05 the pair falls back to SIFT.
- Tie-break order: `crater > lightglue > lnift > rift2 > sift`, applied when
  RMSE differs by less than the ground-truth inter-annotator RMSE and inlier-ratio
  difference is below 0.05.
- One entry per pair is written to `results/arbitration.log` with the reason.

## 9. Evaluation metrics (L7) ✅

`src/evaluation/metrics.py`, computed only on `eval` checkpoints: RMSE,
percentage below 1 px and below 0.5 px, median absolute error, refinement gain,
spatial coverage, grid-density standard deviation, precision / recall / matching
score, and the ground-truth inter-annotator RMSE (noise floor).
`quality.py` adds SSIM, NCC, checkerboard and residual-heat-map generation.

## 10. IIRS track
IIRS cubes (`.qub`, about 80 m/px, 250 bands) are read, Hapke-corrected, a
registration band is selected, and the result is matched against LRO WAC 643 nm
by `IIRSMatcher`. It is exempt from the MSM and runs through
`scripts/iirs_register.py`.

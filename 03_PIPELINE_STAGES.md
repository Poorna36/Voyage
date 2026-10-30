# 03 — Pipeline Stages (S0–S9)

Runbook for the command-line engine. Gate values are the defaults in
`configs/*.yaml` and the previous pipeline specification 📄; scripts and modules
named here exist in the repository ✅.

## 1. Overview

| Stage | Layer | Script | Core module | Gate to proceed | If the gate fails |
|-------|-------|--------|-------------|-----------------|-------------------|
| S0 Setup | – | `scripts/setup_and_download.py` | – | ASP ≥ 3.7.0 and `ISISDATA` set (full SPICE path only) | fix environment |
| S1 Ingest | L0 | `scripts/ingest.py` | `src/ingest/label_parser.py` | `spiceinit` exits 0, footprint non-empty | check SPICE kernels and filenames |
| S2 Pairs | L0 | `scripts/build_pairs.py` | `src/ingest/reference.py` | overlap fraction ≥ 0.50 | keep with `partial_overlap: true`; zero overlap → `skipped.jsonl` |
| S3 Preprocess | L1 | `scripts/preprocess.py` | `src/preprocessing/*` | masked fraction 5–30% | polar (> 30%): continue on unmasked domain |
| S4.5 Selector | L1.5 | (inside `benchmark.py --mode msm`) | `src/selector/*` | valid feature vector | fall back to benchmark mode |
| S4 Match | L2 | `scripts/benchmark.py` | `src/matching/*` | ≥ 150 candidate matches | log failure; arbitration escalates to a fallback matcher |
| S5 Select | L3 | `scripts/benchmark.py` | `src/selection/*` | coverage ≥ 0.60 and ≥ 25 matches | relax cell cap once, else mark matcher failed for the pair |
| S6 Verify | L4 | `scripts/run_s6_s7.py` | `src/registration/*` | ≥ 20 inliers, inlier ratio ≥ 0.05 | widen threshold ×1.5; tile-wise models; escalate |
| S7 Refine | L5 | `scripts/run_s6_s7.py` | `src/refinement/local.py` | ≥ 70% of inliers refined | keep coarse coordinates; `partial_refinement: true` |
| S8 Export | L6 | `scripts/register.py` | – | valid warped footprint ≥ 90% | export available extent; note clipping |
| S9 Evaluate | L7 | `scripts/evaluate_pairs.py` | `src/evaluation/*` | leakage audit exits 0 | remap cells; no leaderboard until fixed |

Every gate failure is written to `results/failures.jsonl` by `src/failures.py`.

## 2. Stage details

### S1 — Product ingestion
Reads ISRO archives from `data/raw/`, runs `isisimport` and `spiceinit` (when ASP
is installed), parses PDS4 XML into a `ProductMeta` and appends one record per
product to `data/metadata/products.jsonl`.
```bash
python scripts/ingest.py --meta data/metadata/products.jsonl --kernels $ISISDATA
```
Supported labels: OHRC and IIRS (Chandrayaan-2), TMC (Chandrayaan-1 format).
Sample labels are in `data/phase1_spec/`.

### S2 — Pair building
For each product, acquires a reference by the fallback chain **NAC (ODE REST) →
WAC crop → SELENE → skip**, pads the bounding box, computes overlap, geo-cell,
terrain class and crater density, assigns the train/test split by geo-cell and
writes `manifest.jsonl`.
```bash
python scripts/build_pairs.py          # reads products.jsonl and queries Lunar ODE
python scripts/fetch_lroc_nac.py nac --min-lat -6.5 --max-lat -5.5 --min-lon 1.0 --max-lon 2.0 --download-img --limit 2
```

### S3 — Preprocessing
Per pair: shadow/invalid mask (dark pixels, local variance, incidence threshold,
validated against a 5–30% mask fraction), percentile clip (2–98) and statistical
transfer, sensor-pair branch (`ohrc_to_nac`: CLAHE → optional inversion →
morphology/PCA; `tmc_to_wac`; minimal), GSD reconciliation (**only the coarser
image is resampled**; bilinear when incidence ≥ 45°, bicubic otherwise), then
512 px tiles with 64 px overlap.
```bash
python scripts/preprocess.py
```

### S4.5 — Matcher selection (optional, production mode)
`MatcherSelector.predict()` turns the 13-feature vector into class probabilities
and a routing decision (see `04_MATCHERS_AND_ALGORITHMS.md` §6). Disabled by
default (`configs/msm.yaml: enabled: false`), in which case every enabled matcher runs.
```bash
python scripts/train_msm.py --manifest data/pairs/manifest.jsonl --results-dir results --processed-dir data/processed
python scripts/benchmark.py --pair <pair_id> --mode msm --msm-config configs/msm.yaml
```

### S4 and S5 — Matching and spatial selection
`scripts/benchmark.py` loops over the matcher registry for each pair, applies a
per-matcher confidence filter, an 8×8 grid cap (5 per cell), one-to-one
resolution and a coverage gate, and writes `matches_raw.json`,
`matches_selected.json`, `selection_stats.json`.
```bash
python scripts/benchmark.py --config configs/ohrc_nac.yaml --splits test \
    --matchers sift,rift2,lightglue,crater --parallel 4 [--resume] [--force]
```

### S6 and S7 — Verification and refinement
`scripts/run_s6_s7.py` runs the F2 checks, then the DEGENSAC model ladder
(similarity → affine → homography, stopping at the first model with inlier RMSE
below the threshold), tile-wise local models at latitude beyond ±55°, GCP
declustering and Z-score filtering (S6); then per-inlier NCC/POC refinement with
a paraboloid peak fit (S7).
```bash
python scripts/run_s6_s7.py --manifest data/pairs/manifest.jsonl --results results --data-dir data --matchers sift,lightglue
```

### S8 — Cartographic export
Warps the source onto the reference grid and writes the products.
```bash
python scripts/register.py --pair <pair_id> --matcher lightglue
```
Outputs: `registered.tif` (GeoTIFF, bicubic), `match_points.csv` and `.gcp`,
`qc_checkerboard.png` (64 px tiles), `qc_matches.png`, `qc_residuals.png`.

### S9 — Evaluation, arbitration, validation
```bash
python scripts/evaluate_pairs.py --manifest data/pairs/manifest.jsonl --results results
python -m src.evaluation.leakage_audit        # exit code 4 on leakage
```
`aggregate.py` builds `leaderboard.csv` by matcher × sensor pair × stratum;
`arbitration.py` writes the winning matcher per pair; `system_validation.py`
evaluates the system-level pass/fail criteria (11 in its docstring).

## 3. Other entry points

| Purpose | Script |
|---------|--------|
| Full comparative benchmark and leaderboard | `scripts/run_full_benchmark.py` |
| Real-data (OHRC + TMC-2 granules) evaluation | `scripts/run_real_data_eval.py` |
| Controlled-deformation stress suite | `scripts/stress_verification.py --patch-size 1024` |
| End-to-end system check (API, PDS4 parsing, upload, routing) | `scripts/verify_system_end_to_end.py` |
| Synthetic benchmark: generate, run, evaluate | `scripts/generate_synthetic_benchmark.py`, `scripts/run_synthetic_benchmark.py`, `scripts/eval_synthetic.py` |
| IIRS hyperspectral registration | `scripts/iirs_register.py` |
| Ground-truth tooling | `scripts/gt_annotator.py`, `scripts/gt_phase7.py`, `scripts/synthetic_gt_check.py` |
| Demo data for the dashboard | `generate_pristine_dataset.py`, `scripts/populate_processed_pairs.py` |
| Coordinate audit | `scripts/audit_coordinates.py` |

## 4. Resume protocol

Re-running skips completed work using the deepest artifact present:

| Deepest artifact | Resume from |
|------------------|-------------|
| none | S1 |
| `products.jsonl` entry | S2 |
| `manifest.jsonl` entry | S3 |
| `data/processed/<pair_id>/meta.json` | S4 |
| `matches_raw.json` | S5 |
| `matches_selected.json` | S6 |
| `geometry.json` | S7 |
| `matches_refined.json` | S8 |
| `registered.tif` + `match_points.csv` | S9 |

`--force` recomputes; `--resume` honours the table.

## 5. Failure handling summary

| Situation | Behaviour |
|-----------|-----------|
| Matcher returns < 150 candidates | failure logged; fallback matcher (SIFT always available) |
| Coverage or match-count gate fails | one relaxed pass, then matcher marked failed for that pair |
| No model passes the ladder | widen threshold ×1.5, then tile-wise models, then escalate |
| CPU-only, polar pair, low crater density | no validated primary matcher; result flagged |
| GPU absent | LightGlue runs on CPU (eligibility is unaffected, only runtime) |
| Crater detector not validated for the sensor | Hough-circle fallback, recorded as `crater_hough` |
| Leakage audit fails | exit code 4; leaderboard not published |

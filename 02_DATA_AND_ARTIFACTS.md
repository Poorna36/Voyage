# 02 — Data and Artifacts

## 1. Folder layout

| Path | Contents | In Git? |
|------|----------|---------|
| `data/raw/` | Chandrayaan-2 product archives (`.zip`, `.xml` + `.img`/`.qub`) | no (placeholder only) |
| `data/calibrated/` | ISIS cubes and calibrated products | no (placeholder only) |
| `data/reference/nac/`, `data/reference/wac/` | LRO NAC strips and WAC crops fetched from ODE | no (placeholders only) |
| `data/pairs/` | `manifest.jsonl`, `skipped.jsonl` | no (git-ignored) |
| `data/processed/<pair_id>/` | Preprocessed pair, masks, metadata | no (git-ignored) |
| `data/metadata/` | `roi_manifest.csv`, `products_real_summary.json`, `products.jsonl` (written by ingest) | partly (`roi_manifest.csv` tracked) |
| `data/metadata/gt/` | Ground-truth checkpoint files `<pair_id>_gt.json` | no (git-ignored) |
| `data/synthetic/gt/` | Hidden ground truth for the synthetic benchmark | placeholder |
| `data/phase1_spec/` | Sample PDS4 labels (OHRC, TMC, IIRS) and a format specification | yes |
| `data/sample_images/` | Two demo images | yes |
| `results/` | Per-pair outputs, `failures.jsonl`, `arbitration.log`, `leaderboard.csv`, `quantum/` | no (git-ignored) |
| `models/` | `msm_v1.pkl`, `msm_v1_stats.json`, `crater_yolov9.pt` (about 19 MB) | yes |

Consequence: a fresh clone contains **no image pairs**. The dashboard and API work
from curated presets and generated placeholders until real data is fetched with
`scripts/setup_and_download.py`, `scripts/fetch_lroc_nac.py` or `generate_pristine_dataset.py`.

## 2. Pair record (`data/pairs/manifest.jsonl`, one JSON object per line)

Written by `scripts/build_pairs.py` (stage S2).

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "src": {
    "product_id": "ch2_ohr_nrp_…", "cub_path": "data/calibrated/….cub",
    "gsd_m": 0.31, "solar_incidence_deg": 42.3, "solar_azimuth_deg": 178.5,
    "sensor": "OHRC", "utc": "2020-08-27T00:30:10.749Z",
    "footprint_ll": [[lon, lat], "… four corners"], "footprint_shape": [2048, 512]
  },
  "ref": {
    "product_id": "M123456789", "path": "data/reference/nac/M123456789_crop.tif",
    "gsd_m": 0.50, "type": "NAC", "footprint_ll": [["…"]]
  },
  "overlap_fraction": 0.83, "partial_overlap": false,
  "delta_azimuth_deg": 5.2, "latitude_center_deg": -6.15,
  "terrain_class": "polar_highland", "crater_density_per_km2": 4.7,
  "geo_cell": "-10_0", "split": "test",
  "gt_path": "data/metadata/gt/<pair_id>_gt.json", "created_at": "…"
}
```

Rules:
- `ref.type` is one of `NAC`, `WAC`, `SELENE`.
- `crater_density_per_km2` is craters per square kilometre. A unitless value is invalid.
- `geo_cell` is `"<lat_bin>_<lon_bin>"` on a 10° grid and is the unit of train/test separation.
- `gt_path` must be empty or absent for `split: train` (ground truth exists only for test pairs).

## 3. Artifacts by stage

| Stage | Output | Location |
|-------|--------|----------|
| S1 ingest | `products.jsonl` (one `ProductMeta` per product) | `data/metadata/` |
| S2 pairs | `manifest.jsonl`, `skipped.jsonl` | `data/pairs/` |
| S3 preprocess | `src.tif`, `ref.tif`, `valid_mask.png`, `meta.json`, tile index | `data/processed/<pair_id>/` |
| S4.5 selector | `selector.json` (routing decision and probabilities) | `results/<pair_id>/` |
| S4 matching | `matches_raw.json` (with provenance) | `results/<pair_id>/<matcher>/` |
| S5 selection | `matches_selected.json`, `selection_stats.json` | `results/<pair_id>/<matcher>/` |
| S6 verification | `geometry.json` (model type, matrix, inliers, RMSE, threshold used) | `results/<pair_id>/<matcher>/` |
| S7 refinement | `matches_refined.json` | `results/<pair_id>/<matcher>/` |
| S8 export | `registered.tif`, `match_points.csv`, `match_points.gcp`, `qc_checkerboard.png`, `qc_matches.png`, `qc_residuals.png` | `results/<pair_id>/<matcher>/` (`--out-dir`) |
| S9 evaluation | `pair_results/*.json`, `leaderboard.csv`, `arbitration.log` | `results/` |
| Any stage | `failures.jsonl` (append-only) | `results/` |
| Quantum (optional) | `results/quantum/<run_id>/…` run records | `results/quantum/` |

Paths marked 📄 come from script docstrings and constants; check the script's
`--out`/`--results` flags if you relocate output.

## 4. `MatchResult` (all matchers)

Defined in `src/matching/base.py` ✅.

| Field | Type | Meaning |
|-------|------|---------|
| `src_xy` | `(N, 2) float32` | `(col, row)` in the source image |
| `ref_xy` | `(N, 2) float32` | `(col, row)` in the reference image |
| `confidence` | `(N,) float32` | per-match confidence in `[0, 1]` |
| `scale` | `(N,) float32` or none | source/reference keypoint size ratio |
| `angle_deg` | `(N,) float32` or none | rotation of the match |
| `runtime_s` | float | wall time of the `match()` call |
| `matcher_params` | dict | at least `matcher_id`; used for provenance |

Shape and length assertions run in `__post_init__`; a `(row, col)` array fails fast.

## 5. Ground truth

- Format: `data/metadata/gt/<pair_id>_gt.json`, checkpoints with a `partition`
  of `fit`, `qc` or `eval`.
- Metrics are computed only on `eval` points (`src/evaluation/metrics.py`);
  adding a `fit` or `qc` point must not change a reported RMSE.
- `scripts/gt_annotator.py` generates and validates checkpoints (semi-automated);
  `gt_interannotator_rmse` measures the noise floor of manual labels.
- Config: 6x6 grid, at least 30 points per pair, 20% re-annotated for QC (`ohrc_nac.yaml`).

## 6. Splits and leakage

- Split key is `geo_cell`. Train and test cells must be disjoint.
- `src/evaluation/leakage_audit.py` checks the manifest, the leaderboard CSV and
  the MSM training data; any violation exits with code 4 and blocks publication
  of a leaderboard.
- The project audit found and fixed a leak where train pairs carried `gt_path`
  and cell `-20_0` appeared in both splits (`AUDIT.md`, ISSUE-07).

## 7. Synthetic benchmark data

`src/synthetic/anchors.py` extracts high-gradient anchor points; `transforms.py`
applies exact geometric and radiometric changes and transforms the anchors with
floating-point precision. Parameters (`TransformParams`) cover affine/projective
shifts, illumination gamma, MTF blur, pushbroom noise and shadow extension.
The generated target image plus the hidden transformed anchors form pairs with
exact ground truth; see `08_TESTING_AND_VALIDATION.md` for the scorecards.

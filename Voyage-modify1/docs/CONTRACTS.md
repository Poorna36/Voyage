# Pipeline Data Contracts and Schemas
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document specifies the data contracts, JSON schemas, and file formats exchanged across all pipeline stages.

---

## 1. Coordinate System Convention

- Pixel coordinates are strictly formatted as `[col, row] = [x, y]` using 0-indexed floating point values.
- Origin `(0.0, 0.0)` is located at the center of the top-left pixel.
- Array indexing in software maps as `image[row, col] -> coordinate[col, row]`.
- Geographic coordinates are expressed as decimal degrees `[longitude, latitude]`.

---

## 2. PairRecord Schema (`data/pairs/manifest.jsonl`)

One JSON record per line. Defines pair metadata, orbital geometry, and split allocation.

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "src": {
    "product_id": "ch2_ohr_nrp_20200827T0030107497_d_img_d18",
    "cub_path": "data/calibrated/ch2_ohr_nrp_20200827T0030107497_d_img_d18.cub",
    "gsd_m": 0.31,
    "solar_incidence_deg": 42.3,
    "solar_azimuth_deg": 178.5,
    "sensor": "OHRC",
    "utc": "2020-08-27T00:30:10.749Z",
    "footprint_ll": [[1.2, -6.1], [1.3, -6.1], [1.3, -6.2], [1.2, -6.2]],
    "footprint_shape": [2048, 512]
  },
  "ref": {
    "product_id": "M123456789",
    "path": "data/reference/nac/M123456789_crop.tif",
    "gsd_m": 0.50,
    "type": "NAC",
    "footprint_ll": [[1.2, -6.1], [1.3, -6.1], [1.3, -6.2], [1.2, -6.2]]
  },
  "overlap_fraction": 0.83,
  "partial_overlap": false,
  "delta_azimuth_deg": 5.2,
  "latitude_center_deg": -6.15,
  "terrain_class": "polar_highland",
  "crater_density_per_km2": 4.7,
  "geo_cell": "-10_0",
  "split": "test",
  "gt_path": "data/metadata/gt/ohr_20200827T003010__nac_M123456789_gt.json",
  "created_at": "2026-08-27T12:00:00Z"
}
```

Field Requirements:
- `pair_id`: Unique identifier combining source and reference tokens.
- `ref.type`: Permitted values are `NAC`, `WAC`, and `SELENE`.
- `crater_density_per_km2`: Physical density in craters per square kilometer. Unitless values are invalid.
- `geo_cell`: Spatial cell identifier `"<lat_bin>_<lon_bin>"` (10-degree grid) used for disjoint train/test partitioning.

---

## 3. MatchRecord Schema

Exported by L2 matching, L3 selection, and L5 refinement:
- `results/<pair_id>/<matcher>/matches_raw.json`
- `results/<pair_id>/<matcher>/matches_selected.json`
- `results/<pair_id>/<matcher>/matches_refined.json`

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "matcher": "lightglue",
  "stage": "refined",
  "matches": [
    {
      "id": 0,
      "src_xy": [512.42, 256.81],
      "ref_xy": [720.14, 410.53],
      "confidence": 0.97,
      "scale": 1.02,
      "angle_deg": 3.1,
      "gate_skip": false,
      "detector_validated": true,
      "refined_delta": [0.18, -0.06],
      "refine_sharpness": 0.84,
      "second_peak_ratio": 0.31,
      "refine_success": true,
      "is_inlier": true,
      "tile_id": "3_5"
    }
  ],
  "stats": {
    "candidate_count": 1240,
    "selected_count": 200,
    "inlier_count": 157,
    "inlier_ratio": 0.785,
    "coverage": 0.72,
    "grid_density_std": 2.3,
    "runtime_s": 4.8
  },
  "matcher_params_hash": "a1b2c3d4",
  "config_hash": "e5f6g7h8",
  "code_commit": "9i0j1k2l",
  "created_at": "2026-08-27T12:05:00Z"
}
```

Stage Lifecycle:
- `matches_raw.json`: Raw correspondences from L2. `is_inlier` and `refined_delta` are omitted.
- `matches_selected.json`: Correspondences filtered by L3 ANMS and grid budget.
- `matches_refined.json`: Final correspondences with L5 sub-pixel offsets, sharpness scores, and inlier flags.

---

## 4. TransformResult Schema (`results/<pair_id>/<matcher>/geometry.json`)

Documents geometric verification and transformation estimation from L4.

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "matcher": "lightglue",
  "model_type": "homography",
  "matrix_3x3": [
    [0.998, -0.012, 45.2],
    [0.012,  0.998, 12.8],
    [0.00001, -0.00002, 1.0]
  ],
  "inlier_count": 157,
  "inlier_ratio": 0.785,
  "inlier_rmse_px": 0.38,
  "condition_number": 14.2,
  "degenerated_detected": false,
  "ladder_history": [
    {"model": "similarity", "inlier_count": 92, "rmse_px": 1.45, "accepted": false},
    {"model": "affine",     "inlier_count": 138, "rmse_px": 0.72, "accepted": false},
    {"model": "homography", "inlier_count": 157, "rmse_px": 0.38, "accepted": true}
  ],
  "tile_models": null,
  "used_tilewise_fallback": false
}
```

---

## 5. Matcher Selection Model (MSM) Schemas

### Feature Vector Specification (`13-Dimensional`)

Extracted at L1.5 by `src/selector/features.py` from `PairRecord` and L1 `meta.json`:

| Index | Feature Key | Type | Description |
|---|---|---|---|
| 0 | `sensor_pair_enc` | Integer | One-hot/ordinal encoding: 0 = OHRC-NAC, 1 = TMC-WAC, 2 = IIRS-WAC |
| 1 | `gsd_ratio` | Float | Ratio of source GSD to reference GSD ($\text{GSD}_{\text{src}} / \text{GSD}_{\text{ref}}$) |
| 2 | `latitude_abs` | Float | Absolute centroid latitude in degrees $|lat| \in [0.0, 90.0]$ |
| 3 | `delta_solar_azimuth` | Float | Absolute difference in solar azimuth $|\Delta az| \in [0.0, 180.0]$ |
| 4 | `terrain_class_enc` | Integer | Encoded terrain class: 0 = highland, 1 = maria, 2 = polar, 3 = mixed |
| 5 | `crater_density` | Float | Logarithm of crater density: $\log(1 + \rho_{\text{crater}})$ |
| 6 | `masked_fraction` | Float | Fraction of total image pixels masked by shadow/validity filters |
| 7 | `overlap_fraction` | Float | Fractional overlap between source and reference footprints $(0.0, 1.0]$ |
| 8 | `src_texture_contrast` | Float | Mean local intensity standard deviation (8x8 window) in source patch |
| 9 | `ref_texture_contrast` | Float | Mean local intensity standard deviation (8x8 window) in reference patch |
| 10 | `src_mean_gradient` | Float | Mean Sobel gradient magnitude in source patch |
| 11 | `ref_mean_gradient` | Float | Mean Sobel gradient magnitude in reference patch |
| 12 | `tile_count` | Integer | Number of non-discarded processing tiles |

### Selector Result Schema (`results/<pair_id>/selector.json`)

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "selected_matcher": "lightglue",
  "confidence": 0.82,
  "fallback_matcher": "sift",
  "all_probs": {
    "sift": 0.12,
    "rift2": 0.04,
    "lightglue": 0.82,
    "crater": 0.02
  },
  "matchers_to_run": ["lightglue"],
  "routing_reason": "high_confidence_prediction",
  "hard_gate_applied": false,
  "is_safe_mode": false,
  "feature_vector": [0, 0.62, 6.15, 66.6, 1, 1.74, 0.12, 0.83, 18.4, 21.1, 14.2, 16.5, 16],
  "model_version": "msm_v1_lgbm",
  "model_hash": "c4d3e2f1",
  "created_at": "2026-08-27T12:02:00Z"
}
```

---

## 6. Evaluation Metrics Schema (`results/<pair_id>/eval_metrics.json`)

Aggregated by L7 evaluation on held-out ground truth:

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "matcher": "lightglue",
  "eval_n_checkpoints": 25,
  "rmse_before_l5_px": 0.48,
  "rmse_after_l5_px": 0.34,
  "refinement_gain_px": 0.14,
  "pct_lt_1px": 1.00,
  "pct_lt_0p5px": 0.88,
  "med_ae_px": 0.28,
  "inlier_count": 157,
  "inlier_ratio": 0.785,
  "spatial_coverage": 0.72,
  "grid_density_std": 2.3,
  "runtime_s": 4.8,
  "gt_interannotator_rmse_px": 0.42,
  "is_statistically_interpretable": true,
  "pass_system_criteria": true
}
```

---

## 7. Ground Truth Checkpoint Schema (`data/metadata/gt/<pair_id>_gt.json`)

```json
{
  "pair_id": "ohr_20200827T003010__nac_M123456789",
  "annotator": "manual_grid_6x6",
  "n_checkpoints": 36,
  "qc_reannotated_pct": 0.20,
  "checkpoints": [
    {
      "id": 0,
      "src_xy": [512.4, 256.8],
      "ref_xy": [720.1, 410.5],
      "partition": "eval"
    },
    {
      "id": 1,
      "src_xy": [680.0, 256.8],
      "ref_xy": [889.3, 408.2],
      "partition": "fit"
    },
    {
      "id": 0,
      "src_xy": [512.4, 256.8],
      "ref_xy": [720.3, 410.7],
      "partition": "qc"
    }
  ]
}
```

---

## 8. Logging Contracts

### Arbitration Log (`results/arbitration.log`)

Appended at the conclusion of evaluation for each pair. Captures competitive matcher rankings:

```text
2026-08-27T12:10:00Z | ohr_20200827T003010__nac_M123456789 | WINNER: lightglue | RMSE: 0.34px | INLIERS: 157 (78.5%) | COV: 0.72 | RUNTIME: 4.8s | MSM_ROUTED: true | CANDIDATES: [sift: 0.71px, lightglue: 0.34px]
```

### Ingestion Failure Log (`data/pairs/skipped.jsonl`)

Records pairs bypassed during ingestion or preprocessing:

```json
{"pair_id": "ohr_20200828T112233__nac_M999999999", "reason": "overlap_fraction_below_minimum", "overlap_fraction": 0.18, "timestamp": "2026-08-27T12:00:00Z"}
```

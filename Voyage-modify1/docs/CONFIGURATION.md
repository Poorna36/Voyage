# System Configuration Specification
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document details all system configuration schemas, parameters, operational thresholds, and tuning guidelines.

Parameters marked with `(TUNE)` require empirical calibration on training/pilot pairs before final production deployment.

---

## 1. Global System Defaults (`configs/default.yaml`)

```yaml
global:
  seed: 42
  coordinate_convention: xy  # col, row (0-indexed floats, top-left origin)

data:
  raw_dir: data/raw
  calibrated_dir: data/calibrated
  reference_dir: data/reference
  pairs_dir: data/pairs
  processed_dir: data/processed
  results_dir: results
  gt_dir: data/metadata/gt
```

---

## 2. Ingestion and Pair Creation (`configs/ohrc_nac.yaml`, `configs/tmc_wac.yaml`)

```yaml
asp:
  version_min: "3.7.0"
  env: conda
  isisdata: $HOME/projects/isisdata
  ck_window_days: 40
  max_ck_size_gb: 5

isisimport:
  preserve_filename: true

spiceinit:
  use_csm: auto

pair:
  k_pointing: 3
  sigma_pointing_m: 1000
  overlap_fraction_min: 0.50
  geo_cell_deg: 10
  wac_reference: data/reference/wac_643nm.tif
  ode_timeout_s: 30
  selene_wmts_url: "https://trek.nasa.gov/moon/"
  reference_fallback_chain: [nac_ode, wac_crop, selene_wmts]
  selene_status: future_compatible

  strata:
    lat_bins:
      equatorial: [-45, 45]
      midlat:     [45, 60]
      polar:      [60, 90]
    az_bins:
      lt30:  [0, 30]
      lt60:  [30, 60]
      lt120: [60, 120]
      gt120: [120, 180]
    crater_density_bins:
      low:    [0, 2]
      medium: [2, 5]
      high:   [5, 100]
```

---

## 3. Preprocessing and Normalization (`L1`)

```yaml
preprocessing:
  shadow_mask:
    incidence_threshold_deg: 80
    local_variance_window: 15
    flat_variance_threshold: 10
    mask_min_pct: 5
    mask_max_pct: 30

  radiometric_norm:
    percentile_clip: [2, 98]
    stat_transfer: true

  sensor_branches:
    ohrc_to_nac:
      clahe_clip_limit: 2.0
      clahe_tile_grid: [8, 8]
      pca_components: 1
      inversion: auto
    tmc_to_wac:
      histogram_match: true
      clahe_clip_limit: 1.5
      clahe_tile_grid: [8, 8]
      experimental: true
    learned_matchers:
      mode: minimal
      only_percentile_clip: true

  gsd:
    interpolation_low_angle: bilinear
    interpolation_high_angle: bicubic
    low_angle_threshold_deg: 45

  tiling:
    size_px: 512
    overlap_px: 64
    min_tile_fraction: 0.50
```

---

## 4. Matcher Registry and Configuration (`L2`)

```yaml
matchers:
  # M0: SIFT (Baseline floor)
  sift:
    nfeatures: 0
    noctave_layers: 3
    contrast_threshold: 0.04
    edge_threshold: 10
    sigma: 1.6
    lowe_ratio: 0.75
    anms:
      enabled: true
      budget: 2000
      method: ssc
    ransac:
      method: degensac
      threshold_px: 3.0
      max_iter: 10000
      confidence: 0.99999

  # M1a: RIFT2 + Scale Extension
  rift2:
    num_scales: 4
    num_orientations: 6
    scale_space_octaves: 4
    pc_threshold: 0.10
    mim_size: 96
    scale_consistency_filter:
      enabled: true
      max_log_scale_deviation: 0.30
    anms:
      enabled: true
      budget: 1500
      method: ssc
    polar_validated: false

  # M1b: LNIFT (Pilot candidate)
  lnift:
    role: m1b_pilot
    anms:
      enabled: true
      budget: 1500
      method: ssc
    scale_consistency_filter:
      enabled: true
      max_log_scale_deviation: 0.30

  # M2: SuperPoint + LightGlue
  lightglue:
    backbone: superpoint
    max_keypoints: 2048
    match_threshold: 0.0
    depth_confidence: 0.95
    width_confidence: 0.99
    requires_gpu: true
    cpu_fallback: true
    f2_checks: mandatory

  # M3: Crater Geometry
  crater:
    detector: yolov9
    min_crater_diameter_px: 8
    crater_density_gate: 3.0
    crater_density_unit: craters_per_km2
    mcr_outlier_method: mcr_structural
    topology_similarity_threshold: 0.65
    gate_terrain_classes: [highland, polar_highland, polar]
    cpu_fallback: hough_circles
    preflight_recall_check: mandatory

arbitration:
  always_run: [sift]
  inlier_ratio_floor: 0.05
  fallback_to: sift
  preference_order: [crater, lightglue, lnift, rift2, sift]
```

---

## 5. Uniform Spatial Selection (`L3`)

```yaml
selection:
  confidence_filter:
    sift: 0.0
    rift2: 0.0
    lightglue: 0.20
    crater: 0.65

  grid:
    n_rows: 8
    n_cols: 8
    cap_per_cell: 5

  budget: 250
  min_matches_after_selection: 25
  coverage_min: 0.60
  conflict_resolution: highest_confidence

  report:
    coverage_before: true
    coverage_after: true
    grid_density_std: true
```

---

## 6. Geometric Verification (`L4`)

```yaml
verification:
  f2_checks:
    enabled: mandatory
    in_domain_buffer_px: 10
    one_to_one: strict

  ransac:
    method: degensac
    max_iter: 10000
    confidence: 0.99999
    threshold_px: auto
    widen_on_failure: 1.5

  t_gsd:
    min_px: 0.5
    max_px: 3.0
    gsd_multiplier: 1.0

  model_ladder:
    - similarity
    - affine
    - homography
    stop_on_rmse_below: 1.0

  tilewise:
    enabled: true
    trigger_latitude_deg: 55
    trigger_relief_m: 500
    tile_size_px: 512
    overlap_px: 256
    min_inliers_per_tile: 12

  gcp_declustering:
    min_spacing_px: 20
    gsd_scale: true
    method: grid_nearest_center
    zscore_threshold: 3.0
    min_gcps_for_zscore: 20
```

---

## 7. Sub-Pixel Refinement (`L5`)

```yaml
refinement:
  method: ncc
  window_px: 32
  apodization: tukey
  tukey_alpha: 0.50
  pyramid_levels: 3
  peak_fit: paraboloid
  sharpness_threshold: 0.15
  second_peak_rejection:
    enabled: true
    flat_window_variance_min: 10.0
    second_peak_ratio_max: 0.80
  min_refined_fraction: 0.70
  report_before_after_rmse: true
```

---

## 8. Cartographic Product Export (`L6`)

```yaml
products:
  registered_format: GeoTIFF
  resampling: bicubic
  valid_warp_fraction_min: 0.90
  gcp_format: csv
  qc_artifacts:
    - checkerboard_overlay
    - match_overlay
    - residual_heatmap
  qc_checkerboard_tile_px: 64
  residual_heatmap_sigma: 3
```

---

## 9. Evaluation and Reporting (`L7`)

```yaml
evaluation:
  gt:
    grid_size: 6
    pairs_to_annotate: [15, 20]
    min_points_per_pair: 30
    qc_reannotate_pct: 0.20
    eval_partition: eval

  leakage:
    split_by: geo_cell
    geo_cell_size_deg: 10
    audit_required: true

  metrics:
    primary:
      - rmse_px
      - pct_lt_1px
      - pct_lt_0p5px
      - inlier_ratio
    secondary:
      - medae_px
      - inlier_count
      - spatial_coverage
      - grid_density_std
      - refinement_gain_px
      - runtime_s

  reporting:
    stratify_by: [sensor_pair, terrain_class, latitude_bin, delta_az_bin]
    never_hide: [polar, high_latitude, partial_overlap]
    aggregate: [mean, median]
```

---

## 10. Hyperspectral IIRS Module (`configs/iirs_wac.yaml`)

```yaml
extends: default

iirs:
  format: qub
  photometric_correction: true
  correction_model: hapke
  registration_band: auto
  reference: wac
  accuracy_target_m: 80
  separate_module: true
```

---

## 11. Matcher Selection Model (`configs/msm.yaml`)

```yaml
msm:
  enabled: false
  model_path: "models/msm_v1.pkl"
  model_stats_path: "models/msm_v1_stats.json"
  model_version: "msm_v1"

  tau_high: 0.65
  tau_low: 0.40

  hard_rules:
    crater_density_gate:
      enabled: true
      tau_c: 5.0
    gpu_gate:
      enabled: true
      check_at_startup: true
    iirs_track_gate:
      enabled: true

  fallback:
    on_model_load_error: "benchmark_mode"
    on_feature_extraction_error: "benchmark_mode"
    on_s4_gate_failure: "sift"
    log_all_fallback_events: true
```

---

## 12. Calibration Guidelines for Tunable Parameters

Parameters requiring empirical validation on pilot data:
- `shadow_mask.incidence_threshold_deg`: Sharp polar shadows require threshold adjustment between 75 and 85 degrees.
- `anms.budget`: Adjust according to texture density across highland versus mare terrain.
- `crater.crater_density_gate`: Calibrate `tau_c` based on YOLOv9 detection counts per unit area.
- `model_ladder.stop_on_rmse_below`: Default 1.0 px threshold may be relaxed to 1.5 px for low-resolution TMC-2 pairs.
- `refinement.sharpness_threshold`: Evaluated on pilot imagery to balance outlier rejection against valid match retention.

Tuning Hygiene Rule: All parameter tuning must be performed strictly on the training and pilot splits. Evaluating or tuning configuration values on held-out test pairs is considered data contamination.

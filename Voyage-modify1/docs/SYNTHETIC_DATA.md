# Synthetic Lunar Benchmark Generation
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document details the generation of synthetic evaluation pairs with mathematically exact sub-pixel ground truth for component-wise pipeline validation.

---

## 1. Motivation and Rationale

In planetary remote sensing, sub-pixel ground truth does not exist naturally on the lunar surface. Physical ground control points (GCPs) and survey monuments are absent. Manual human clicking exhibits an intrinsic error bound between 0.8 and 1.5 pixels, exceeding the required sub-pixel registration target ($\text{RMSE} < 0.50\text{ px}$).

Synthetic generation provides:
1. Double-Precision Mathematical Ground Truth: Transformation matrices ($\mathbf{H}_{\text{true}}$) are analytically known, yielding zero annotator bias.
2. Parameter Isolation: Individual orbital and imaging phenomena (GSD mismatch, yaw/pitch rotation, grazing solar incidence, and pushbroom sensor noise) are controlled and tested independently.
3. Component-Wise Pipeline Isolation: Tracks correspondence survival rates and Euclidean residual decay from Matcher Selection (L1.5) through Raw Matching (L2), Spatial Optimization (L3), Geometric Verification (L4), and Sub-Pixel Refinement (L5).

---

## 2. Mathematical Formulation

### Geometric Transformation Engine

For each synthetic pair, a ground-truth transformation matrix $\mathbf{H}_{\text{true}} \in \mathbb{R}^{3 \times 3}$ is generated:

$$\mathbf{H}_{\text{true}} = \mathbf{T}(c_x, c_y) \cdot \mathbf{P} \cdot \mathbf{S}(\gamma_x, \gamma_y) \cdot \mathbf{K}(s_x, s_y) \cdot \mathbf{R}(\theta) \cdot \mathbf{T}(-c_x, -c_y)$$

Where:
- $\mathbf{T}$: Translation to image center $(c_x, c_y)$
- $\mathbf{R}(\theta)$: Planetary yaw rotation $\theta \in [-180^\circ, 180^\circ]$
- $\mathbf{K}(s_x, s_y)$: Scale factor $s \in [0.4, 3.0]$ matching inter-sensor GSD ratios
- $\mathbf{S}(\gamma_x, \gamma_y)$: Spacecraft trajectory shear matrix
- $\mathbf{P}$: Projective tilt vector representing off-nadir optical pointing

### Sub-Pixel Checkpoint Mapping

Ground-truth reference coordinates $\mathbf{x}_{\text{ref}}$ are mapped from source coordinates $\mathbf{x}_{\text{src}}$ via the inverse homography:

$$\mathbf{x}_{\text{ref}} = \mathbf{H}_{\text{true}}^{-1} \cdot \mathbf{x}_{\text{src}}$$

Checkpoints are placed on a uniform sub-pixel grid across the valid image domain and assigned to partitions:
- `eval` (70%): Reserved strictly for computing held-out metrics ($\text{RMSE}_{\text{eval}}$, $\text{MedAE}$, $\text{pct\_lt\_0p5px}$).
- `fit` (20%): Used to verify model numerical conditioning.
- `qc` (10%): Injected with Gaussian jitter ($\sigma = 0.5\text{ px}$) to simulate baseline human annotator variance.

### Radiometric and Sensor Simulation

To emulate Chandrayaan-2 and LRO physical imaging conditions:
1. Grazing Solar Illumination: Directional luminance ramps and dynamic shadow masks simulate solar incidence variations up to $85^\circ$.
2. Photometric Non-Linearity: Power-law gamma curves ($I_{\text{out}} = I_{\text{in}}^\gamma, \gamma \in [0.7, 1.4]$) represent varying solar phase angles.
3. Pushbroom Sensor Noise: Poisson shot noise and Gaussian readout noise simulate sensor electronics:

$$I_{\text{noisy}} = \frac{\text{Poisson}(I \cdot g)}{g} + \mathcal{N}(0, \sigma_n^2)$$

4. Point Spread Function (PSF): Sensor-specific optical modulation transfer function (MTF) blurring.

Excluded Transformations:
- Extreme non-physical warping, JPEG artifacts, and artificial color jitter are excluded to preserve scientific fidelity with uncompressed PDS4 data products.

---

## 3. Component-Wise Pipeline Evaluation Flow

```text
[Source Lunar Tile]
       |
       v (Anchor Extraction)
[Anchor Keypoints (Hidden GT)]
       |
       v (Physical Transformation Engine)
[Target Image + Transformed GT Coordinates]
       |
       +--------------------------------------------+
       | Run Pipeline blindly (Images only)         |
       |  -> L1.5 Selection                         |
       |  -> L2 Correspondence Finding              |
       |  -> L3 Uniform Spatial Selection           |
       |  -> L4 Geometric Verification (DEGENSAC)   |
       |  -> L5 Sub-Pixel Refinement                |
       +--------------------------------------------+
       |
       v (Evaluation Engine: eval_synthetic.py)
[Component-Wise Scorecards against Hidden GT]
```

### GT-to-Prediction Assignment Rule
1. Distance Gating: Correspondences must fall within a maximum radius ($r_{\max} = 2.0\text{ px}$) of the transformed ground truth.
2. One-to-One Matching: When multiple candidates fall within the gating radius, assignments are resolved using the Hungarian algorithm to prevent duplicate count inflation.
3. Outliers: Predictions exceeding the gating threshold or unassigned duplicates are counted as false positives.

---

## 4. Benchmark Stratification and Volume

### Benchmark Validation Split (30 to 60 Pairs)
Used for automated continuous regression testing and algorithm leaderboard scoring:
- 10 pairs: Equatorial Mare (baseline low/medium texture)
- 10 pairs: Equatorial Highland (high relief topography)
- 10 pairs: Polar Highland (grazing shadows, high solar incidence)
- 10 pairs: Multi-Scale GSD Jumps (1.5x to 4.0x resolution mismatch)
- 10 pairs: Extreme Solar Azimuth Disparity ($\Delta\text{azimuth} > 90^\circ$)
- 10 pairs: Projective Off-Nadir Perspective Tilt

### Execution Command

```bash
# Generate 30 stratified synthetic benchmark pairs
python scripts/generate_synthetic_pairs.py --num-pairs 30 --patch-size 512

# Run full component-wise evaluation
python scripts/run_full_benchmark.py --synthetic-dir data/synthetic/
```

Outputs are structured into `data/synthetic/<pair_id>/` containing `src.tif`, `ref.tif`, `gt.json`, and `meta.json`.

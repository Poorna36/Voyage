# Synthetic Lunar Dataset & Benchmark Generation (`synthetic/`)

This directory houses the standalone **Synthetic Generation & Component-Wise Evaluation Track** (v3.0) for the SIH 2026 (PS-26166) project.

It provides mathematically exact, physical-orbital-condition synthetic image generation and component-wise evaluation for Chandrayaan-2 (OHRC / TMC-2 / IIRS) vs LRO (NAC / WAC).

---

## Directory Structure

```text
synthetic/
├── __init__.py           # Package exports (extract_anchors, transforms, etc.)
├── anchors.py            # GT Anchor Extraction Engine (Shi-Tomasi + 5 morphological buckets)
├── transforms.py         # Physical Transformation Engine (scale, rotation, translation, illumination, MTF, pushbroom)
├── scene_generator.py    # Multi-scale procedural lunar terrain synthesis
├── generate.py           # Synthetic Benchmark Generator CLI (Phases 1–4)
├── evaluate.py           # Component-Wise Evaluation CLI (L1.5 through L5 against hidden GT)
├── runner.py             # End-to-end benchmark orchestrator
└── README.md             # This guide
```

---

## Modules Overview

### 1. `anchors.py` — GT Anchor Extraction
Extracts natural high-gradient floating-point anchor points as hidden ground truth:
- **Phase 1 (Baseline)**: Shi-Tomasi corner detection across uniform $8 \times 8$ grid cells.
- **Phase 2+ (Stratified)**: Five morphological bucket detectors:
  - Craters (rims and floors)
  - Ridges and scarps
  - Maria (flat low-gradient regions)
  - Shadow boundaries (terminator transitions)
  - Polar terrain (high-incidence illumination gradients)

### 2. `transforms.py` — Physical Transformation Engine
Derives strictly from lunar orbital imaging conditions (no generic CV augmentations):
- Exact $3 \times 3$ matrix math (`build_transform_matrix()`, `transform_gt_points()`)
- Lanczos resampling with $[0, 1]$ clipping (`apply_transform()`)
- Illumination gamma shifts ($\gamma \in [0.7, 1.4]$)
- Shadow extension along solar azimuth
- Sensor MTF blur ($\sigma \in [0.5, 1.5]$)
- Pushbroom vertical striping

### 3. `scene_generator.py` — Procedural Lunar Terrain
Generates realistic lunar scenes with multi-scale craters and surface relief when raw PDS4 archives are offline.

### 4. CLIs (`generate.py`, `evaluate.py`, `runner.py`)
- **`generate.py`**: Generates synthetic pairs, manifests, and hidden GT files.
- **`evaluate.py`**: Evaluates correspondence pipeline outputs across all 5 stages (L1.5, L2, L3, L4, L5).
- **`runner.py`**: Runs the complete cycle: generate $\to$ pipeline $\to$ evaluate $\to$ aggregate statistical report (mean $\pm$ 95% CI).

---

## Quick Start Examples

### Generate Phase 1 Smoke Test Pairs
```bash
# Generate using built-in synthetic lunar base scene:
python -m synthetic.generate --phase 1 --synthetic-base

# Or using raw images from data/raw:
python -m synthetic.generate --config configs/synthetic_benchmark.yaml --images data/raw/ --phase 1
```

### Run Evaluation Against Hidden GT
```bash
python -m synthetic.evaluate --manifest data/synthetic/synthetic_manifest.jsonl --results results/ --out results/synthetic_benchmark/
```

### Run Full End-to-End Benchmark
```bash
python -m synthetic.runner --phase 1 --synthetic-base --matchers sift lightglue
```

---

## Python API Usage

```python
import numpy as np
from synthetic import (
    extract_anchors,
    build_transform_matrix,
    apply_transform,
    transform_gt_points,
    generate_synthetic_pair,
    generate_synthetic_lunar_scene,
)

# 1. Synthesize a base lunar scene
base_image = generate_synthetic_lunar_scene(512, 512, seed=42)

# 2. Extract hidden GT anchors
config = {"anchors": {"target_count": 80, "min_count": 50, "phase": 1}}
anchor_set = extract_anchors(base_image, pair_id="pair_001", config=config)

# 3. Apply physical transformation
tf_config = {"transforms": {"translation": {"enabled": True, "max_shift_px": 0.5}}}
target_image, params, M = generate_synthetic_pair(base_image, tf_config, pair_id="pair_001", seed=42)

# 4. Compute exact analytical target GT coordinates
src_coords = anchor_set.as_numpy()
tgt_coords = transform_gt_points(src_coords, M)
```

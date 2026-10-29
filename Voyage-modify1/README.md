# Cross-Sensor Lunar Image Correspondence and Registration
## Smart India Hackathon 2026 — Problem Statement PS-26166

Automated, scale-invariant, illumination-robust image correspondence system for Chandrayaan-2 optical payloads (OHRC, TMC-2, IIRS) against global lunar reference imagery (LRO NAC, LRO WAC, SELENE).

---

## 1. System Overview

This repository provides a modular, benchmark-first pipeline designed to resolve high-stakes cross-sensor lunar registration challenges:

- Extreme Illumination Variation: Stable matching across solar azimuth disparities up to 180 degrees and solar incidence angles up to 85 degrees.
- Large Spatial Scale Gaps: Seamless correspondence across spatial resolution mismatches up to 17x (such as OHRC at 0.3 m/px versus TMC-2 at 5 m/px).
- Topographic Relief and Curvature: Robust geometric modeling using a hierarchical ladder (Similarity -> Affine -> Homography -> Tile-wise Local Models) to prevent planar collapse and polar divergence.
- Multi-Modal Radiometry: Tailored normalization, shadow validity masking, and dedicated hyperspectral processing for Chandrayaan-2 IIRS data.
- Automated Meta-Selection (MSM): A lightweight LightGBM classifier predicts optimal matcher routing from a 13-dimensional scene feature vector, reducing production runtime by over 50% while bounding error degradation within 0.10 pixels.

---

## 2. Core Architecture

The system decouples registration concerns into discrete, reproducible layers:

```text
L0  Data and Geometry        ISRO PDS4 parsing, SPICE kernels, automated LRO reference querying
L1  Preprocessing            Shadow validity masking, radiometric normalization, GSD pyramid
L1.5 Matcher Selection (MSM)  LightGBM meta-routing based on 13-feature scene/sensor vector
L2  Correspondence Engine    Pluggable matchers: M0 (SIFT), M1 (RIFT2/LNIFT), M2 (LightGlue), M3 (Crater)
L3  Spatial Optimization     Pre-match ANMS (SSC) and post-match 8x8 grid density budgeting
L4  Geometric Verification   DEGENSAC/MAGSAC++ outlier rejection and hierarchical model ladder
L5  Sub-Pixel Refinement     Local NCC and phase correlation with 2D paraboloid peak interpolation
L6  Cartographic Export      16-bit ortho-rectified GeoTIFFs, GCP manifests, and visual QC overlays
L7  Evaluation & Arbitration Ground-truth checkpoint validation, multi-strata scoring, arbitration log
```

---

## 3. Quickstart Guide

### Prerequisites
- Python 3.9+ with scientific libraries (`numpy`, `scipy`, `opencv-python-headless`, `lightgbm`, `pyyaml`).
- Node.js 18+ and npm (for the interactive mission control dashboard).
- Optional: Ames Stereo Pipeline (ASP >= 3.7.0) with ISIS3 camera models for full SPICE ephemeris ingestion.

### 1. Running the Mission Control Dashboard

Start the FastAPI backend service:
```bash
# From repository root
python -m pip install -r requirements.txt
python -m api.server
```
The API initializes on `http://localhost:8000` (verify status at `http://localhost:8000/api/health`).

In a separate terminal, launch the frontend dashboard:
```bash
cd sih-dashboard
npm install
npm run dev
```
Open `http://localhost:5173` in your browser. The dashboard connects to the live backend API and falls back to embedded mock datasets if the backend is offline.

To build for production:
```bash
cd sih-dashboard
npm run build
```

### 2. Automated LRO Reference Data Retrieval

Acquire calibrated LRO NAC CDR science strips without manual downloads via the NASA Lunar Orbital Data Explorer (ODE) REST API:

```bash
# Fetch calibrated LRO NAC science strips for a target bounding box
python scripts/fetch_lroc_nac.py nac \
    --min-lat -6.5 --max-lat -5.5 \
    --min-lon 1.0  --max-lon 2.0  \
    --download-img --max-incidence 75 --max-res 1.5 --limit 2

# Fetch matching LRO WAC tiles for TMC-2 or regional pairing
python scripts/fetch_lroc_nac.py wac \
    --min-lat -6.5 --max-lat -5.5 \
    --min-lon 1.0  --max-lon 2.0 --limit 2

# Batch retrieval using an ROI manifest
python scripts/fetch_lroc_nac.py nac --manifest data/metadata/roi_manifest.csv --download-img
```

Outputs are automatically organized into `data/reference/nac/` and `data/reference/wac/`.

### 3. Pipeline Execution and Benchmarking

```bash
# Execute full benchmark suite across all candidate matchers on the test split
python scripts/benchmark.py --config configs/ohrc_nac.yaml --splits test \
  --matchers sift,rift2,lightglue,crater --parallel 4

# Run production mode with intelligent Matcher Selection Model (MSM) routing
python scripts/benchmark.py --pair <pair_id> --mode msm --msm-config configs/msm.yaml

# Generate cartographic products and quality control diagnostics
python scripts/register.py --pair <pair_id> --matcher lightglue

# Run automated multi-deformation stress verification suite
python scripts/stress_verification.py --patch-size 1024
```

---

## 4. Data Provenance and Scientific Attribution

- Chandrayaan-2 Data: Courtesy of the Indian Space Research Organisation (ISRO). Calibrated Level-2 data products retrieved from the [ISSDC PRADAN](https://pradan.issdc.gov.in/) portal.
- Lunar Reconnaissance Orbiter (LRO) Data: Public domain data courtesy of NASA / Goddard Space Flight Center / Arizona State University, accessed via the [Washington University Lunar Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/moon/).

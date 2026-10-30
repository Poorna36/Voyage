# Chandrayaan-2 and Lunar Reference Data Guide
## SIH 2026 PS-26166: Cross-Sensor Lunar Image Correspondence

This document specifies data collection, directory conventions, automated reference acquisition, and ground-truth checkpoint standards for the registration pipeline.

All processing operates exclusively on Calibrated Level-2 data products. Raw telemetry (Level-0/EDR) calibration steps are handled upstream by mission ground segments.

---

## 1. Primary Sensor Specifications

### Chandrayaan-2 Source Products (ISRO PRADAN)

| Sensor | Spectral Band | Spatial Resolution (GSD) | Level | Search Identifier | File Pattern | Pipeline Target |
|---|---|---|---|---|---|---|
| OHRC | Panchromatic (450-900 nm) | 0.25 - 0.32 m/px | Level-2 | Instrument: OHRC | `ch2_ohr_nrp_*_d_img_d18.zip` | Primary source (OHRC to NAC) |
| TMC-2 | Panchromatic Stereo | 5.0 m/px | Level-2 | Instrument: TMC-2 | `ch2_tmc_nrp_*_d_img_d32.zip` | Secondary source (TMC-2 to WAC) |
| IIRS | Hyperspectral (0.8-5.0 um, 250 bands) | 80 m/px | Level-2 | Instrument: IIRS | `ch2_iir_nrp_*_d_img_d32.zip` | Parallel module (IIRS to WAC) |

### LRO Reference Products (Automated via Lunar ODE)

| Product | Instrument | Nominal GSD | ODE Product Type | Script Target | Pipeline Role |
|---|---|---|---|---|---|
| LRO NAC CDR | Narrow Angle Camera | 0.50 - 1.0 m/px | `CDRNAC4` | `scripts/fetch_lroc_nac.py nac` | Reference for OHRC |
| LRO WAC CDR | Wide Angle Camera | 100 m/px | `CDRWAC4` | `scripts/fetch_lroc_nac.py wac` | Reference for TMC-2 and IIRS |
| WAC 643nm Mosaic | Wide Angle Camera | 100 m/px | Local GeoTIFF | Fallback crop via GDAL | Global baseline coverage |

---

## 2. Ingestion and Metadata Requirements

### ISRO PRADAN Retrieval
1. Register a verified account on the ISRO PRADAN portal (`https://pradan.issdc.gov.in`).
2. Download target Level-2 calibrated zip archives.
3. Unpack directly into `data/raw/`.

Unpacked file layout:
```text
data/raw/
  ch2_ohr_nrp_YYYYMMDDTHHMMSS_d_img_d18.img     (16-bit raster image)
  ch2_ohr_nrp_YYYYMMDDTHHMMSS_d_img_d18.lbl     (PDS3 label)
  ch2_ohr_nrp_YYYYMMDDTHHMMSS_d_img_d18.xml     (PDS4 XML metadata)
```

Key XML elements extracted during ingestion (`scripts/ingest.py`):
```xml
<CENTER_LATITUDE>-6.08</CENTER_LATITUDE>
<CENTER_LONGITUDE>1.25</CENTER_LONGITUDE>
<SOLAR_INCIDENCE_ANGLE>72.3</SOLAR_INCIDENCE_ANGLE>
<SOLAR_AZIMUTH_ANGLE>245.1</SOLAR_AZIMUTH_ANGLE>
<MAP_SCALE>0.25</MAP_SCALE>
```

Operational Rules:
- Original ISRO filenames must never be modified. Sub-processes depend on tokenized mission timestamps and product codes.
- The accompanying `.xml` files are required for geometric and illumination parsing.

---

## 3. Automated LRO Reference Retrieval

Reference acquisition is handled through the Lunar Orbital Data Explorer (ODE) REST API via `scripts/fetch_lroc_nac.py`.

### Retrieval Commands

```bash
# Query and download calibrated NAC science strips for a geographic bounding box
python scripts/fetch_lroc_nac.py nac \
    --min-lat -6.5 --max-lat -5.5 \
    --min-lon 1.0  --max-lon 2.0  \
    --download-img --max-incidence 75 --max-res 1.5 --limit 5

# Download matching WAC tiles for TMC-2 pairing
python scripts/fetch_lroc_nac.py wac \
    --min-lat -6.5 --max-lat -5.5 \
    --min-lon 1.0  --max-lon 2.0 --limit 2

# Batch retrieval using an ROI manifest
python scripts/fetch_lroc_nac.py nac --manifest data/metadata/roi_manifest.csv --download-img
```

### Reference Fallback Hierarchy
1. LRO NAC via ODE: Preferred for sub-meter resolution cross-matching.
2. Local LRO WAC 643 nm Mosaic: Cropped when NAC coverage is unavailable or for regional TMC-2 swaths.
3. SELENE Moon Trek WMTS: Future fallback stratum (Kaguya TC/MI).

---

## 4. Repository Data Directory Structure

```text
data/
  raw/                          Raw archives and unpacked Level-2 ISRO products
  calibrated/                   Generated ISIS cubes (.cub) with SPICE kernels
  reference/
    nac/                        Calibrated LRO NAC CDR GeoTIFFs and XML labels
    wac/                        Calibrated LRO WAC strips
    wac_643nm.tif               Local global lunar WAC 643 nm mosaic
  pairs/
    manifest.jsonl              Canonical PairRecord catalog for all candidate pairs
    skipped.jsonl               Catalog of pairs skipped during ingestion with reasons
    failures.jsonl              Catalog of pipeline run failures
  processed/
    <pair_id>/
      src.tif                   Normalized source image patch
      ref.tif                   Normalized reference image patch
      valid_mask.png            Binary validity and shadow mask
      tiles.geojson             Spatial grid tile boundaries
      meta.json                 Provenance and feature extraction statistics
  metadata/
    products.jsonl              Per-product geometric and solar metadata
    roi_manifest.csv            Target ROI coordinate definitions
    gt/
      <pair_id>_gt.json         Ground-truth control point coordinates
```

---

## 5. Ground Truth Annotation Standards

Validation accuracy relies on verified control points stored in `data/metadata/gt/<pair_id>_gt.json`.

### Coordinate System Definition
- Pixel coordinates are strictly formatted as `[col, row] = [x, y]` as 0-indexed floating point values.
- The coordinate origin `(0.0, 0.0)` corresponds to the center of the top-left pixel.
- Array indexing maps as `matrix[row, col] -> [col, row]`.
- Geographic coordinates are expressed as decimal degrees `[longitude, latitude]`.

### Checkpoint Partitioning Scheme
Every control point is allocated to one of three functional partitions:

| Partition | Purpose | Calculation Rule | Minimum Allocation |
|---|---|---|---|
| `eval` | Held-out evaluation | Exclusively used to compute algorithm RMSE, MedAE, and success rate | >= 70% of total points (>= 20 per pair) |
| `fit` | Model fitting check | Verifies homography and affine numerical conditioning | 20% to 30% of points |
| `qc` | Inter-annotator precision | Computes annotator variance and bounds measurable accuracy claims | 20% re-annotated points |

### Mathematical Invariants
Evaluation RMSE is calculated exclusively across the `eval` partition:

$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{eval}}} \sum_{i \in \text{eval}} \|\mathbf{x}_{\text{pred}, i} - \mathbf{x}_{\text{gt}, i}\|^2}$$

Inter-annotator variance is determined from duplicate annotations in the `qc` partition:

$$\text{RMSE}_{\text{interann}} = \sqrt{\frac{1}{N_{\text{qc}}} \sum_{j \in \text{qc}} \|\mathbf{x}_{\text{eval}, j} - \mathbf{x}_{\text{qc}, j}\|^2}$$

No algorithm accuracy claim is valid if the reported error is lower than `RMSE_interann`.

### Test Set Stratification
The benchmark test suite requires a minimum of 30 pairs distributed across terrain and lighting strata:
- Terrain diversity: At least 5 pairs each for `equatorial_mare`, `equatorial_highland`, `polar_highland`, `polar_mare`, `crater_floor`, and `ejecta`.
- Polar latitudes: At least 3 pairs with `|latitude| > 55 deg`.
- Illumination disparity: At least 3 pairs with `delta_azimuth > 90 deg`.
- Low crater density: At least 3 pairs with `crater_density < 1.0 craters/km^2`.
- Sensor pairs: Full representation across OHRC-NAC, TMC-2-WAC, and IIRS-WAC.

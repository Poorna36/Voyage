# Phase 1 Data Format Specification & Ingestion Guide

> **Target Audience:** AI Coding Agents / Developers building **Phase 1** (`src/ingest/label_parser.py`, `src/ingest/reference.py`, `scripts/ingest.py`, `scripts/build_pairs.py`).
>
> **Purpose:** Comprehensive data format specification for ISRO Chandrayaan datasets (OHRC, TMC, IIRS) derived from official PDS4 labels without loading multi-gigabyte binary payload files (`.img`, `.qub`).

---

## 1. Directory Structure of Raw ISRO Products

Each ISRO product archive decompresses into a standard PDS4 directory tree:

```
<product_dir>/
├── data/
│   ├── calibrated/ (or raw/)
│   │   └── <YYYYMMDD>/
│   │       ├── <product_name>.xml   <-- PDS4 Metadata XML (PARSED BY INGEST)
│   │       ├── <product_name>.img   <-- Binary Image Payload (DO NOT READ IN TEXT MODE)
│   │       ├── <product_name>.qub   <-- Binary Spectral Cube (IIRS only)
│   │       └── <product_name>.hdr   <-- ENVI Header (IIRS only)
├── geometry/
│   └── calibrated/
│       └── <YYYYMMDD>/
│           ├── <product_name>_g_grd_d18.xml
│           └── <product_name>_g_grd_d18.csv  <-- Geometry grid CSV
├── browse/
└── miscellaneous/
```

---

## 2. Sensor Metadata Overview

| Sensor | Mission | Band / Mode | Data Format | Bit Depth | Typical GSD (`isda:pixel_resolution`) | Dimensions (`[Lines, Samples]`) | Sample Label File |
|---|---|---|---|---|---|---|---|
| **OHRC** | Chandrayaan-2 | Panchromatic (500–800 nm) | `.img` (2D) | 8-bit (`UnsignedByte`) | ~0.25 m – 0.28 m | `[79796, 12000]` | [`ohrc_sample.xml`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/ohrc_sample.xml) |
| **TMC** | Chandrayaan-1 | Stereo Panchromatic | `.img` (2D) | 16-bit (`UnsignedLSB2`) | ~5.0 m – 9.79 m | `[202688, 4000]` | [`tmc_sample.xml`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/tmc_sample.xml) |
| **IIRS** | Chandrayaan-2 | Hyperspectral (0.8–5.0 µm) | `.qub` (3D) + `.hdr` | 16-bit (`UnsignedLSB2`) | ~80 m – 94.5 m | `[13965, 250, 256]` | [`iirs_sample.xml`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/iirs_sample.xml) |

---

## 3. Exact PDS4 XML Metadata Extraction Rules

All PDS4 labels use the namespace `xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1"` and root element `<Product_Observational>`.

Below is the exact XPath / tag mapping required to populate the `ProductMeta` dataclass in `src/ingest/label_parser.py`:

| Field in `ProductMeta` | Target PDS4 XML Tag Path | Example Value | Fallback / Default |
|---|---|---|---|
| `product_id` | `//Identification_Area/logical_identifier` (extracted stem after last `:`) | `ch2_ohr_ncp_20211228t2209123959_d_img_d18` | Base filename without extension |
| `sensor` | `//Observing_System_Component[type='Instrument']/name` | `OHRC` / `TMC` / `IIRS` | Derived from product_id prefix (`ohr`, `tmc`, `iir`) |
| `utc` | `//Observation_Area/Time_Coordinates/start_date_time` | `2021-12-28T22:09:12.3959Z` | `ISO 8601 string` |
| `gsd_m` | `//Mission_Area/isda:Product_Parameters/isda:pixel_resolution` | `0.28` (float) | Must be positive float |
| `solar_incidence_deg` | `//Mission_Area/isda:Product_Parameters/isda:solar_incidence` | `90.043949` (float) | Float (0–180) |
| `solar_azimuth_deg` | `//Mission_Area/isda:Product_Parameters/isda:sun_azimuth` | `152.400416` (float) | Float (0–360) |
| `footprint_ll` | `//isda:System_Level_Coordinates` (UL, UR, LR, LL lat/lon) | `[[55.56, -89.92], [110.42, -89.85], ...]` | 4-point list `[[lon, lat], ...]` |
| `footprint_shape` | `//File_Area_Observational/Array_2D_Image` or `.hdr` | `[79796, 12000]` | `[lines, samples]` |

---

## 4. Footprint Coordinate Schema & Convention

ISRO PDS4 labels record corner coordinates under `<isda:System_Level_Coordinates>` or `<isda:Refined_Corner_Coordinates>`:

```xml
<isda:System_Level_Coordinates>
    <isda:upper_left_latitude unit="deg">-89.923132</isda:upper_left_latitude>
    <isda:upper_left_longitude unit="deg">55.564452</isda:upper_left_longitude>
    <isda:upper_right_latitude unit="deg">-89.850542</isda:upper_right_latitude>
    <isda:upper_right_longitude unit="deg">110.416030</isda:upper_right_longitude>
    <isda:lower_left_latitude unit="deg">-89.257559</isda:lower_left_latitude>
    <isda:lower_left_longitude unit="deg">233.745958</isda:lower_left_longitude>
    <isda:lower_right_latitude unit="deg">-89.252796</isda:lower_right_latitude>
    <isda:lower_right_longitude unit="deg">224.351932</isda:lower_right_longitude>
</isda:System_Level_Coordinates>
```

> ⚠️ **CRITICAL COORDINATE CONVENTION**:
> - All geographic coordinates MUST be formatted as `[longitude, latitude]` in decimal degrees (WGS84 / Selenographic).
> - Longitudes in ISRO data are in $[0, 360^\circ]$. Convert to $[-180^\circ, +180^\circ]$ if `lon > 180`: `lon = ((lon + 180) % 360) - 180`.
> - Ordered standard polygon points: `[upper_left, upper_right, lower_right, lower_left]`.

---

## 5. Sample Files in `data/phase1_spec/`

The following sample files are provided in this directory for AI agent reference & unit testing:

1. **[`ohrc_sample.xml`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/ohrc_sample.xml)** — Complete PDS4 XML label for Chandrayaan-2 OHRC.
2. **[`tmc_sample.xml`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/tmc_sample.xml)** — Complete PDS4 XML label for Chandrayaan-1 TMC.
3. **[`iirs_sample.xml`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/iirs_sample.xml)** — Complete PDS4 XML label for Chandrayaan-2 IIRS.
4. **[`iirs_sample.hdr`](file:///c:/Workspace/code/Research/SIH/SIH_2026/data/phase1_spec/iirs_sample.hdr)** — ENVI Header file for IIRS 256-band spectral cube.

---

## 6. Python Blueprint for `src/ingest/label_parser.py`

Below is the reference code contract for parsing any ISRO PDS4 XML label into a `ProductMeta` dataclass:

```python
from dataclasses import dataclass
from typing import List, Tuple
import xml.etree.ElementTree as ET
from pathlib import Path

@dataclass
class ProductMeta:
    product_id: str
    cub_path: str
    gsd_m: float
    solar_incidence_deg: float
    solar_azimuth_deg: float
    sensor: str           # "OHRC", "TMC", "IIRS"
    utc: str
    footprint_ll: List[List[float]]  # [[lon, lat], [lon, lat], ...]
    footprint_shape: List[int]        # [lines, samples]

def parse_pds4_label(xml_path: str) -> ProductMeta:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Strip namespaces for robust tag matching
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
            
    # Product ID
    lid = root.findtext(".//logical_identifier", default=Path(xml_path).stem)
    product_id = lid.split(":")[-1] if ":" in lid else Path(xml_path).stem
    
    # Sensor identification
    inst_name = (root.findtext(".//Observing_System_Component[type='Instrument']/name") or "").lower()
    if "high resolution" in inst_name or "ohrc" in product_id:
        sensor = "OHRC"
    elif "terrain" in inst_name or "tmc" in product_id:
        sensor = "TMC"
    elif "infrared" in inst_name or "iir" in product_id:
        sensor = "IIRS"
    else:
        sensor = "UNKNOWN"
        
    # Time Coordinates
    utc = root.findtext(".//Time_Coordinates/start_date_time", default="")
    
    # Parameters
    gsd_m = float(root.findtext(".//pixel_resolution", default="1.0"))
    solar_inc = float(root.findtext(".//solar_incidence", default="0.0"))
    solar_az = float(root.findtext(".//sun_azimuth", default="0.0"))
    
    # Footprint Coordinates (UL, UR, LR, LL)
    def normalize_lon(lon_val: float) -> float:
        return ((lon_val + 180.0) % 360.0) - 180.0 if lon_val > 180.0 else lon_val

    ul_lat = float(root.findtext(".//upper_left_latitude", default="0.0"))
    ul_lon = normalize_lon(float(root.findtext(".//upper_left_longitude", default="0.0")))
    ur_lat = float(root.findtext(".//upper_right_latitude", default="0.0"))
    ur_lon = normalize_lon(float(root.findtext(".//upper_right_longitude", default="0.0")))
    lr_lat = float(root.findtext(".//lower_right_latitude", default="0.0"))
    lr_lon = normalize_lon(float(root.findtext(".//lower_right_longitude", default="0.0")))
    ll_lat = float(root.findtext(".//lower_left_latitude", default="0.0"))
    ll_lon = normalize_lon(float(root.findtext(".//lower_left_longitude", default="0.0")))
    
    footprint_ll = [
        [ul_lon, ul_lat],
        [ur_lon, ur_lat],
        [lr_lon, lr_lat],
        [ll_lon, ll_lat]
    ]
    
    # Image shape
    lines = int(root.findtext(".//Axis_Array[axis_name='Line']/elements", default="0"))
    samples = int(root.findtext(".//Axis_Array[axis_name='Sample']/elements", default="0"))
    footprint_shape = [lines, samples]
    
    cub_path = str(Path(xml_path).with_suffix(".cub"))
    
    return ProductMeta(
        product_id=product_id,
        cub_path=cub_path,
        gsd_m=gsd_m,
        solar_incidence_deg=solar_inc,
        solar_azimuth_deg=solar_az,
        sensor=sensor,
        utc=utc,
        footprint_ll=footprint_ll,
        footprint_shape=footprint_shape
    )
```

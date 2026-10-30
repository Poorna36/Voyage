"""
src/selector/features.py
========================
Feature extraction and representation for the Matcher Selection Model (MSM).

Extracts a 13-dimensional canonical feature vector from a PairRecord (manifest)
and L1 preprocessing metadata (meta.json):
  1. sensor_pair_enc       (int: 0=OHRC-NAC, 1=TMC-WAC, 2=IIRS-WAC)
  2. gsd_ratio             (float: source GSD / reference GSD in (0, 1.0])
  3. latitude_abs          (float: |lat| in [0.0, 90.0] degrees)
  4. delta_solar_azimuth   (float: |delta_az| in [0.0, 180.0] degrees)
  5. terrain_class_enc     (int: 0=highland, 1=maria, 2=polar, 3=mixed)
  6. crater_density        (float: log(1 + crater_density))
  7. masked_fraction       (float: [0.0, 1.0])
  8. overlap_fraction      (float: (0.0, 1.0])
  9. src_texture_contrast  (float: mean local std in 8x8 windows)
  10. ref_texture_contrast (float: mean local std in 8x8 windows)
  11. src_mean_gradient    (float: mean Sobel gradient magnitude)
  12. ref_mean_gradient    (float: mean Sobel gradient magnitude)
  13. tile_count           (int: active tile count >= 1)

References:
  - FEATURES.md F26
  - INTERFACES.md §10.1
  - ARCHITECTURE.md §3 (L1.5)
  - PROGRESS.md §5.5.2
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np


SENSOR_PAIR_MAP = {
    "OHRC-NAC": 0,
    "OHRC_NAC": 0,
    "OHRC": 0,
    "TMC-WAC": 1,
    "TMC_WAC": 1,
    "TMC2-WAC": 1,
    "TMC2_WAC": 1,
    "TMC": 1,
    "IIRS-WAC": 2,
    "IIRS_WAC": 2,
    "IIRS": 2,
}

TERRAIN_CLASS_MAP = {
    "highland": 0,
    "equatorial_highland": 0,
    "crater_floor": 0,
    "ejecta": 0,
    "maria": 1,
    "mare": 1,
    "equatorial_mare": 1,
    "polar_highland": 2,
    "polar_mare": 2,
    "polar": 2,
    "mixed": 3,
    "equatorial": 3,
}

FEATURE_NAMES = [
    "sensor_pair_enc",
    "gsd_ratio",
    "latitude_abs",
    "delta_solar_azimuth",
    "terrain_class_enc",
    "crater_density",
    "masked_fraction",
    "overlap_fraction",
    "src_texture_contrast",
    "ref_texture_contrast",
    "src_mean_gradient",
    "ref_mean_gradient",
    "tile_count",
]


@dataclass
class MSMFeatureVector:
    """
    13-dimensional canonical feature vector for Matcher Selection Model.
    """
    pair_id: str
    sensor_pair_enc: int
    gsd_ratio: float
    latitude_abs: float
    delta_solar_azimuth: float
    terrain_class_enc: int
    crater_density: float
    masked_fraction: float
    overlap_fraction: float
    src_texture_contrast: float
    ref_texture_contrast: float
    src_mean_gradient: float
    ref_mean_gradient: float
    tile_count: int
    feature_vector_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation."""
        return asdict(self)

    def to_array(self) -> np.ndarray:
        """Return 1D float32 numpy array of the 13 numeric features."""
        return vectorize_features(self)


def extract_features(
    pair_record: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]] = None,
) -> MSMFeatureVector:
    """
    Extract a canonical 13-dimensional feature vector from PairRecord and meta.json.

    Parameters
    ----------
    pair_record : dict
        Record from manifest.jsonl (or dictionary representing pair metadata).
    meta_json : dict, optional
        Metadata from data/processed/<pair_id>/meta.json (if preprocessing completed).

    Returns
    -------
    MSMFeatureVector
        Constructed feature vector with deterministic MD5 hash.
    """
    meta = meta_json or {}
    pair_id = pair_record.get("pair_id", meta.get("pair_id", "unknown"))

    # 1. Sensor pair encoding
    sensor_str = pair_record.get("sensor_pair", meta.get("sensor_pair", ""))
    if not sensor_str:
        src_sensor = pair_record.get("src", {}).get("sensor", "")
        ref_type = pair_record.get("ref", {}).get("type", "")
        if src_sensor and ref_type:
            sensor_str = f"{src_sensor}-{ref_type}"
    sensor_pair_enc = SENSOR_PAIR_MAP.get(str(sensor_str).upper(), 0)

    # 2. GSD ratio (src_gsd / ref_gsd) in (0, 1.0]
    src_gsd = float(
        pair_record.get("src_gsd_m")
        or (pair_record.get("src") or {}).get("gsd_m")
        or meta.get("src_gsd_m")
        or 0.31
    )
    ref_gsd = float(
        pair_record.get("ref_gsd_m")
        or (pair_record.get("ref") or {}).get("gsd_m")
        or meta.get("ref_gsd_m")
        or 0.50
    )
    if ref_gsd > 0:
        ratio = src_gsd / ref_gsd
        if ratio > 1.0:
            ratio = 1.0 / ratio
        gsd_ratio = float(np.clip(ratio, 0.01, 1.0))
    else:
        gsd_ratio = 1.0

    # 3. Absolute latitude [0.0, 90.0]
    lat = pair_record.get("latitude_center_deg")
    if lat is None:
        footprint = (pair_record.get("src") or {}).get("footprint_ll", [])
        if footprint and len(footprint) > 0:
            lat = float(np.mean([pt[1] for pt in footprint if len(pt) > 1]))
        else:
            lat = 0.0
    latitude_abs = float(np.clip(abs(float(lat or 0.0)), 0.0, 90.0))

    # 4. Delta solar azimuth [0.0, 180.0]
    delta_az = pair_record.get("delta_azimuth_deg")
    if delta_az is None:
        src_az = (pair_record.get("src") or {}).get("solar_azimuth_deg", 0.0)
        ref_az = (pair_record.get("ref") or {}).get("solar_azimuth_deg", 0.0)
        if src_az is not None and ref_az is not None:
            raw_diff = abs(float(src_az or 0.0) - float(ref_az or 0.0)) % 360.0
            if raw_diff > 180.0:
                raw_diff = 360.0 - raw_diff
            delta_az = raw_diff
        else:
            delta_az = 0.0
    delta_solar_azimuth = float(np.clip(abs(float(delta_az or 0.0)), 0.0, 180.0))

    # 5. Terrain class encoding
    terrain_str = str(pair_record.get("terrain_class") or "mixed").lower().strip()
    terrain_class_enc = TERRAIN_CLASS_MAP.get(terrain_str, 3)

    # 6. Crater density (log(1 + density))
    c_density = pair_record.get("crater_density_per_km2")
    if c_density is None:
        c_density = pair_record.get("crater_density", 0.0)
    crater_density = float(math.log1p(max(0.0, float(c_density or 0.0))))

    # 7. Masked fraction [0.0, 1.0]
    mask_frac = meta.get("masked_fraction") or meta.get("mask_fraction") or 0.0
    masked_fraction = float(np.clip(float(mask_frac or 0.0), 0.0, 1.0))

    # 8. Overlap fraction (0.0, 1.0]
    overlap = pair_record.get("overlap_fraction") or 1.0
    overlap_fraction = float(np.clip(float(overlap or 1.0), 0.01, 1.0))

    # 9. Source texture contrast
    src_contrast = float(meta.get("src_texture_contrast") or 0.0)
    src_texture_contrast = max(0.0, src_contrast)

    # 10. Reference texture contrast
    ref_contrast = float(meta.get("ref_texture_contrast") or 0.0)
    ref_texture_contrast = max(0.0, ref_contrast)

    # 11. Source mean gradient
    src_grad = float(meta.get("src_mean_gradient") or 0.0)
    src_mean_gradient = max(0.0, src_grad)

    # 12. Reference mean gradient
    ref_grad = float(meta.get("ref_mean_gradient") or 0.0)
    ref_mean_gradient = max(0.0, ref_grad)

    # 13. Tile count >= 1
    t_count = meta.get("tile_count") or meta.get("n_tiles") or 1
    tile_count = max(1, int(t_count or 1))


    # Construct feature object
    feat = MSMFeatureVector(
        pair_id=str(pair_id),
        sensor_pair_enc=sensor_pair_enc,
        gsd_ratio=round(gsd_ratio, 4),
        latitude_abs=round(latitude_abs, 4),
        delta_solar_azimuth=round(delta_solar_azimuth, 4),
        terrain_class_enc=terrain_class_enc,
        crater_density=round(crater_density, 4),
        masked_fraction=round(masked_fraction, 4),
        overlap_fraction=round(overlap_fraction, 4),
        src_texture_contrast=round(src_texture_contrast, 4),
        ref_texture_contrast=round(ref_texture_contrast, 4),
        src_mean_gradient=round(src_mean_gradient, 4),
        ref_mean_gradient=round(ref_mean_gradient, 4),
        tile_count=tile_count,
    )
    feat.feature_vector_hash = hash_features(feat)
    return feat


def vectorize_features(features: MSMFeatureVector) -> np.ndarray:
    """
    Convert MSMFeatureVector to a 1D float32 NumPy array in canonical feature order.

    Order:
      0: sensor_pair_enc
      1: gsd_ratio
      2: latitude_abs
      3: delta_solar_azimuth
      4: terrain_class_enc
      5: crater_density
      6: masked_fraction
      7: overlap_fraction
      8: src_texture_contrast
      9: ref_texture_contrast
      10: src_mean_gradient
      11: ref_mean_gradient
      12: tile_count
    """
    arr = np.array([
        float(features.sensor_pair_enc),
        float(features.gsd_ratio),
        float(features.latitude_abs),
        float(features.delta_solar_azimuth),
        float(features.terrain_class_enc),
        float(features.crater_density),
        float(features.masked_fraction),
        float(features.overlap_fraction),
        float(features.src_texture_contrast),
        float(features.ref_texture_contrast),
        float(features.src_mean_gradient),
        float(features.ref_mean_gradient),
        float(features.tile_count),
    ], dtype=np.float32)
    return arr


def hash_features(features: MSMFeatureVector) -> str:
    """
    Compute a deterministic MD5 hex digest string for the feature vector values.
    """
    payload = {
        "sensor_pair_enc": features.sensor_pair_enc,
        "gsd_ratio": features.gsd_ratio,
        "latitude_abs": features.latitude_abs,
        "delta_solar_azimuth": features.delta_solar_azimuth,
        "terrain_class_enc": features.terrain_class_enc,
        "crater_density": features.crater_density,
        "masked_fraction": features.masked_fraction,
        "overlap_fraction": features.overlap_fraction,
        "src_texture_contrast": features.src_texture_contrast,
        "ref_texture_contrast": features.ref_texture_contrast,
        "src_mean_gradient": features.src_mean_gradient,
        "ref_mean_gradient": features.ref_mean_gradient,
        "tile_count": features.tile_count,
    }
    dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()

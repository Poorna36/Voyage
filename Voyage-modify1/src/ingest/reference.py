"""
src/ingest/reference.py
========================
F02 — Automated Reference Patch Acquisition.

Implements the three-step reference fallback chain (per PIPELINE.md §S2 and
CONFIGURATION.md §2.2 reference_fallback_chain):

  1. NAC ODE   — query Lunar ODE RESTFUL API by bounding box -> download NAC strip
  2. WAC crop  — GDAL crop of local WAC 643nm mosaic GeoTIFF
  3. SELENE    — Moon Trek WMTS connectivity check (stub; full impl deferred per config:
                 selene_status: future_compatible)
  4. skip      — write to skipped.jsonl with reason

Coordinate convention (MANDATORY):
  - All bboxes and coordinates: [lon_min, lat_min, lon_max, lat_max] in decimal degrees [-180, 180].
  - Geographic coords: (lon, lat). NEVER (lat, lon).

pad_bbox():
  Expands a 4-corner footprint polygon by k * sigma_pointing_m in all directions.
  Output: [lon_min, lat_min, lon_max, lat_max] — the padded bounding box.

References:
  - docs/INTERFACES.md §1 (PairRecord ref.type: NAC | WAC | SELENE)
  - docs/FEATURES.md F02
  - docs/CONFIGURATION.md §2.2
  - docs/PIPELINE.md §S2
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lunar mean radius in metres (IAU 2009)
MOON_RADIUS_M = 1_737_400.0

# ODE (Orbital Data Explorer) RESTFUL API for LRO NAC
ODE_NAC_API_URL = "https://oderest.rsl.wustl.edu/live2/?target=Moon&instrumentid=26&producttype=EDR"

# Moon Trek WMTS base URL
SELENE_WMTS_URL = "https://trek.nasa.gov/moon/"

# Timeout for all HTTP requests (seconds)
HTTP_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# Bounding Box Padding
# ---------------------------------------------------------------------------

def pad_bbox(
    footprint_ll: List[List[float]],
    sigma_m: float,
    k: float,
) -> List[float]:
    """
    Compute a padded bounding box from a 4-corner footprint polygon.

    Expands the footprint bbox by (k * sigma_m) metres in all four directions.
    The expansion is done in arc-degree space:
        delta_lat_deg = (k * sigma_m) / MOON_RADIUS_M  * (180 / pi)
        delta_lon_deg = delta_lat_deg / cos(lat_center_rad)

    The lon expansion uses the centre latitude for an accurate conversion.

    Parameters:
        footprint_ll:  4-corner polygon [[lon, lat], ...].
                       Corners must be in decimal degrees [-180, 180].
        sigma_m:       Pointing uncertainty in metres (typical: 500–2000 m for OHRC).
        k:             Padding multiplier (typical: 3; from CONFIGURATION.md pair.k_pointing).

    Returns:
        [lon_min, lat_min, lon_max, lat_max] — the padded bounding box.

    Coordinate convention:
        Output is always (lon, lat) order. NEVER (lat, lon).

    Example:
        >>> pad_bbox([[55.56, -89.92], [110.42, -89.85], [224.35, -89.25], [233.75, -89.26]],
        ...          sigma_m=1000, k=3)
        [lon_min, lat_min, lon_max, lat_max]  # padded by 3*1000=3000m in each direction

    Raises:
        ValueError: If footprint_ll has fewer than 3 points.
    """
    if len(footprint_ll) < 3:
        raise ValueError(
            f"footprint_ll must have at least 3 corners, got {len(footprint_ll)}"
        )

    lons = [pt[0] for pt in footprint_ll]
    lats = [pt[1] for pt in footprint_ll]

    lon_min = min(lons)
    lon_max = max(lons)
    lat_min = min(lats)
    lat_max = max(lats)

    # Centre latitude for lon-degree conversion
    lat_center_rad = math.radians((lat_min + lat_max) / 2.0)

    # Padding in degrees
    padding_m = k * sigma_m
    delta_lat_deg = math.degrees(padding_m / MOON_RADIUS_M)

    # Avoid division by zero at poles (cos(90°) = 0)
    cos_lat = math.cos(lat_center_rad)
    if abs(cos_lat) < 1e-6:
        # At poles, longitude padding is undefined; use latitude padding for both axes
        delta_lon_deg = delta_lat_deg
        logger.warning(
            "pad_bbox: centre lat=%.2f°  is near pole; using isotropic lat padding for lon",
            math.degrees(lat_center_rad),
        )
    else:
        delta_lon_deg = delta_lat_deg / cos_lat

    padded = [
        lon_min - delta_lon_deg,   # lon_min
        lat_min - delta_lat_deg,   # lat_min
        lon_max + delta_lon_deg,   # lon_max
        lat_max + delta_lat_deg,   # lat_max
    ]

    # Clamp to valid selenographic ranges
    padded[0] = max(padded[0], -180.0)
    padded[1] = max(padded[1], -90.0)
    padded[2] = min(padded[2], 180.0)
    padded[3] = min(padded[3], 90.0)

    logger.debug(
        "pad_bbox: raw=[%.4f,%.4f,%.4f,%.4f] -> padded=[%.4f,%.4f,%.4f,%.4f] (k=%.1f, sigma=%.0fm)",
        lon_min, lat_min, lon_max, lat_max,
        padded[0], padded[1], padded[2], padded[3],
        k, sigma_m,
    )
    return padded


# ---------------------------------------------------------------------------
# NAC ODE Reference Acquisition
# ---------------------------------------------------------------------------

def query_ode_nac(
    footprint_ll: List[List[float]],
    padding_m: float,
    out_dir: str,
    timeout_s: int = HTTP_TIMEOUT_S,
    k: float = 1.0,
) -> Optional[str]:
    """
    Query the Lunar ODE RESTFUL API for an LRO NAC strip covering the given footprint.

    Constructs a bbox query from the padded footprint and downloads the best
    matching NAC EDR GeoTIFF (or JP2) to out_dir.

    Parameters:
        footprint_ll:  4-corner footprint [[lon, lat], ...] in degrees [-180, 180].
        padding_m:     Absolute padding in metres to add to the bbox (e.g. k * sigma_m).
        out_dir:       Directory where the downloaded crop should be saved.
        timeout_s:     HTTP timeout in seconds.
        k:             Padding multiplier forwarded to pad_bbox (default 1.0 if padding_m already scaled).

    Returns:
        Absolute path to the downloaded NAC GeoTIFF/JP2 file, or None if:
          - ODE returns no results for the bbox
          - HTTP request times out or fails
          - Downloaded file is empty

    Reference:
        ODE API: https://oderest.rsl.wustl.edu
        Instrument ID 26 = LRO NAC-R; product type EDR (level 1 calibrated strips)
    """
    try:
        bbox = pad_bbox(footprint_ll, sigma_m=padding_m, k=k)
    except ValueError as e:
        logger.error("query_ode_nac: pad_bbox failed: %s", e)
        return None

    lon_min, lat_min, lon_max, lat_max = bbox

    # Build ODE RESTFUL API request
    params = {
        "target": "Moon",
        "instrumentid": "26",   # LRO NAC-R
        "producttype": "CDRNAC", # Calibrated Data Records
        "query": "footprint",
        "results": "m",          # minimal metadata
        "output": "JSON",
        "loc": "f",
        "minlat": str(lat_min),
        "maxlat": str(lat_max),
        "westlon": str(lon_min),
        "eastlon": str(lon_max),
    }

    base_url = "https://oderest.rsl.wustl.edu/live2/"
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    logger.info(
        "ODE NAC query: bbox=[%.4f,%.4f,%.4f,%.4f] url=%s",
        lon_min, lat_min, lon_max, lat_max, url,
    )

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.warning("ODE NAC query failed (network): %s", e)
        return None
    except Exception as e:
        logger.warning("ODE NAC query failed: %s", e)
        return None

    # Parse ODE response — look for download URLs
    products = []
    try:
        ode_results = data.get("ODEResults", {})
        for key in ("Products", "Product"):
            products_raw = ode_results.get(key, {})
            if isinstance(products_raw, dict):
                prod_list = products_raw.get("Product", [])
                if isinstance(prod_list, dict):
                    prod_list = [prod_list]
                products = prod_list
                break
            elif isinstance(products_raw, list):
                products = products_raw
                break
    except Exception as e:
        logger.warning("ODE NAC: failed to parse response: %s", e)
        return None

    if not products:
        logger.info("ODE NAC: no products found for bbox [%.4f,%.4f,%.4f,%.4f]", *bbox)
        return None

    # Pick first product with a download URL
    download_url = None
    product_name = None
    for prod in products:
        try:
            # ODE response structure: Product > Product_files > Product_file > URL
            files = prod.get("Product_files", {}).get("Product_file", [])
            if isinstance(files, dict):
                files = [files]
            for f in files:
                url_val = f.get("URL", "")
                if url_val.endswith((".tif", ".TIF", ".jp2", ".JP2", ".img", ".IMG")):
                    download_url = url_val
                    product_name = prod.get("pds_archive_path", "nac_download")
                    break
            if download_url:
                break
        except Exception:
            continue

    if not download_url:
        logger.info("ODE NAC: no downloadable GeoTIFF/JP2 found in response")
        return None

    # Download the NAC file
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(download_url).path).name
    dest = out_path / filename

    logger.info("ODE NAC: downloading %s -> %s", download_url, dest)
    try:
        with urllib.request.urlopen(download_url, timeout=timeout_s) as resp:
            with open(dest, "wb") as fout:
                shutil.copyfileobj(resp, fout)
    except Exception as e:
        logger.warning("ODE NAC: download failed: %s", e)
        if dest.exists():
            dest.unlink()
        return None

    if not dest.exists() or dest.stat().st_size == 0:
        logger.warning("ODE NAC: downloaded file is empty: %s", dest)
        return None

    logger.info("ODE NAC: downloaded %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
    return str(dest)


# ---------------------------------------------------------------------------
# WAC Mosaic Crop
# ---------------------------------------------------------------------------

def crop_wac_mosaic(
    mosaic_path: str,
    bbox_ll: List[float],
) -> Optional[str]:
    """
    Crop the WAC 643nm mosaic GeoTIFF to the given bounding box using GDAL.

    Produces a new GeoTIFF file adjacent to the mosaic with a bbox-derived filename.
    GDAL's gdal_translate is called via gdal.Translate to crop without resampling.

    Parameters:
        mosaic_path:  Absolute path to the local WAC 643nm mosaic GeoTIFF.
        bbox_ll:      [lon_min, lat_min, lon_max, lat_max] in decimal degrees [-180, 180].

    Returns:
        Absolute path to the cropped GeoTIFF, or None if:
          - mosaic_path does not exist
          - GDAL is not installed
          - Crop produces an empty or invalid raster

    Coordinate convention:
        bbox_ll is (lon, lat). GDAL projWin uses [ulx, uly, lrx, lry] = [lon_min, lat_max, lon_max, lat_min].
    """
    src = Path(mosaic_path)
    if not src.exists():
        logger.warning("crop_wac_mosaic: mosaic not found: %s", mosaic_path)
        return None

    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except ImportError:
        logger.error("crop_wac_mosaic: GDAL Python bindings not available (pip install gdal)")
        return None

    lon_min, lat_min, lon_max, lat_max = bbox_ll

    # Output file named by bbox to enable caching
    bbox_tag = f"{lon_min:.3f}_{lat_min:.3f}_{lon_max:.3f}_{lat_max:.3f}".replace("-", "n")
    out_dir = src.parent
    out_path = out_dir / f"wac_crop_{bbox_tag}.tif"

    if out_path.exists():
        logger.debug("crop_wac_mosaic: using cached crop: %s", out_path)
        return str(out_path)

    # GDAL projWin: [ulx, uly, lrx, lry] = [lon_min, lat_max, lon_max, lat_min]
    try:
        logger.info(
            "Cropping WAC mosaic: projWin=[%.4f, %.4f, %.4f, %.4f] -> %s",
            lon_min, lat_max, lon_max, lat_min, out_path.name,
        )
        ds = gdal.Translate(
            str(out_path),
            str(src),
            projWin=[lon_min, lat_max, lon_max, lat_min],
            format="GTiff",
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
        )
    except Exception as e:
        logger.error("crop_wac_mosaic: gdal.Translate failed: %s", e)
        return None

    if ds is None:
        logger.error("crop_wac_mosaic: gdal.Translate returned None for %s", out_path)
        return None

    ds = None  # close dataset

    if not out_path.exists() or out_path.stat().st_size < 100:
        logger.warning("crop_wac_mosaic: output is empty or too small: %s", out_path)
        return None

    logger.info("crop_wac_mosaic: crop saved to %s (%.1f KB)", out_path.name, out_path.stat().st_size / 1024)
    return str(out_path)


# ---------------------------------------------------------------------------
# SELENE Moon Trek WMTS Stub
# ---------------------------------------------------------------------------

def check_selene_connectivity(
    url: str = SELENE_WMTS_URL,
    timeout_s: int = HTTP_TIMEOUT_S,
) -> bool:
    """
    Connectivity check for the SELENE Moon Trek WMTS endpoint.

    Full SELENE implementation is deferred (CONFIGURATION.md §2.2 selene_status: future_compatible).
    This stub only verifies that the Moon Trek WMTS is reachable.
    No data is downloaded.

    Parameters:
        url:        Moon Trek WMTS base URL.
        timeout_s:  HTTP timeout in seconds.

    Returns:
        True if the endpoint responds (any HTTP status), False if connection fails.

    Note:
        When a pair reaches this fallback, ref.type=SELENE should be recorded
        in PairRecord, and the pair forms a separate evaluation stratum.
        A SELENE-specific acquisition and coordinate handling implementation
        is required before full SELENE pairs can be processed.
    """
    try:
        logger.info("SELENE connectivity check: %s", url)
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout_s):
            logger.info("SELENE Moon Trek WMTS reachable: %s", url)
            return True
    except Exception as e:
        logger.warning("SELENE Moon Trek WMTS not reachable: %s | %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Reference Fallback Chain
# ---------------------------------------------------------------------------

def acquire_reference(
    footprint_ll: List[List[float]],
    padding_m: float,
    out_dir: str,
    wac_mosaic_path: Optional[str] = None,
    selene_url: str = SELENE_WMTS_URL,
    ode_timeout_s: int = HTTP_TIMEOUT_S,
) -> Tuple[Optional[str], str]:
    """
    Execute the authoritative reference fallback chain for a source product footprint.

    Chain order (per CONFIGURATION.md §2.2 reference_fallback_chain):
      1. NAC via Lunar ODE bbox search
      2. WAC 643nm GDAL crop (if wac_mosaic_path is provided and exists)
      3. SELENE Moon Trek WMTS connectivity check (stub; ref.type=SELENE recorded)
      4. Skip — returns (None, 'no_reference_found') to be written to skipped.jsonl

    Parameters:
        footprint_ll:    4-corner footprint [[lon, lat], ...] in degrees [-180, 180].
        padding_m:       Padding in metres to apply to bbox (k * sigma_pointing_m).
        out_dir:         Directory to save downloaded/cropped reference files.
        wac_mosaic_path: Path to local WAC 643nm mosaic (or None to skip WAC step).
        selene_url:      Moon Trek WMTS URL for connectivity check.
        ode_timeout_s:   HTTP timeout for ODE and SELENE requests.

    Returns:
        Tuple of (path_or_none, ref_type_string):
          - ('path/to/crop.tif', 'NAC')     — NAC ODE succeeded
          - ('path/to/crop.tif', 'WAC')     — WAC crop succeeded
          - (None, 'SELENE')                — SELENE reachable (stub); no data downloaded
          - (None, 'no_reference_found')    — all fallbacks exhausted
    """
    # Step 1: NAC ODE
    logger.info("Reference chain: trying NAC ODE for footprint %s", footprint_ll)
    nac_path = query_ode_nac(
        footprint_ll=footprint_ll,
        padding_m=padding_m,
        out_dir=out_dir,
        timeout_s=ode_timeout_s,
    )
    if nac_path:
        return nac_path, "NAC"

    # Step 2: WAC crop
    if wac_mosaic_path:
        logger.info("Reference chain: trying WAC crop for footprint %s", footprint_ll)
        try:
            bbox = pad_bbox(footprint_ll, sigma_m=padding_m, k=1.0)
        except ValueError as e:
            logger.error("Reference chain: pad_bbox for WAC failed: %s", e)
            bbox = None

        if bbox:
            wac_path = crop_wac_mosaic(wac_mosaic_path, bbox)
            if wac_path:
                return wac_path, "WAC"
    else:
        logger.info("Reference chain: WAC mosaic not configured; skipping WAC step")

    # Step 3: SELENE connectivity check (stub)
    logger.info("Reference chain: trying SELENE connectivity check")
    selene_ok = check_selene_connectivity(url=selene_url, timeout_s=ode_timeout_s)
    if selene_ok:
        logger.warning(
            "Reference chain: SELENE is reachable but full SELENE acquisition is not yet "
            "implemented (selene_status: future_compatible). Returning ref.type=SELENE with no data."
        )
        return None, "SELENE"

    # Step 4: No reference found
    logger.warning("Reference chain: all fallbacks exhausted for footprint %s", footprint_ll)
    return None, "no_reference_found"

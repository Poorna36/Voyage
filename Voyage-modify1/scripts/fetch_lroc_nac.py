#!/usr/bin/env python3
"""
fetch_lroc_nac.py — Automated LRO Reference Data Downloader
=================================================================

Downloads calibrated (Level-2) LRO reference data for the SIH26166 pipeline:
  - LRO NAC CDR strips (0.5-2 m/px GeoTIFF)  →  data/reference/nac/
  - LRO WAC 643nm global mosaic (100 m/px)  →  data/reference/wac_643nm.tif
  - LOLA DEM (optional)                     →  data/reference/lola/

Queries NASA Lunar ODE REST API (https://oderest.rsl.wustl.edu).
All LRO data is public domain — no login required.

Aligns with:
  - PIPELINE.md S2 (pair building: NAC via Lunar ODE bbox search)
  - CONFIGURATION.md §2.2 (reference_fallback_chain: [nac_ode, wac_crop, selene_wmts])
  - INTERFACES.md PairRecord (ref.type: NAC | WAC | SELENE)

Usage
-----
  # Download NAC strips covering a Chandrayaan-2 footprint:
  python scripts/fetch_lroc_nac.py nac \\
      --min-lat -6.5 --max-lat -5.5 --min-lon 1.0 --max-lon 2.0 --limit 5

  # Download WAC 643nm global mosaic tile:
  python scripts/fetch_lroc_nac.py wac \\
      --min-lat -6.5 --max-lat -5.5 --min-lon 1.0 --max-lon 2.0

  # Batch download NAC from a manifest CSV:
  python scripts/fetch_lroc_nac.py nac --manifest data/metadata/roi_manifest.csv

  # Called programmatically from build_pairs.py:
  from scripts.fetch_lroc_nac import fetch_nac_for_footprint
  results = fetch_nac_for_footprint(min_lat, max_lat, min_lon, max_lon)

Manifest CSV format
-------------------
  roi_name,min_lat,max_lat,min_lon,max_lon
  landing_site_A,-6.5,-5.5,1.0,2.0
  crater_region_B,45.0,46.0,30.0,31.0

Author: SIH 2026 Team (PS-26166)
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# ─── Configuration ──────────────────────────────────────────────────────────

ODE_BASE = "https://oderest.rsl.wustl.edu/live2/"
USER_AGENT = "SIH26166-LunarPipeline/1.0"
OVERQUERY = 4  # over-query factor: ODE cannot filter server-side

# Valid ODE product types for LRO LROC (from iiptset query)
PRODUCT_TYPES = {
    "nac":      "CDRNAC4",   # Calibrated NAC strip (0.5-2 m/px)
    "wac":      "CDRWAC4",   # Calibrated WAC color (multi-band)
    "wac_mono": "CDRWAM4",   # Calibrated WAC mono
}

# File extensions to download (keyed by lowercase suffix)
DOWNLOAD_PREFS = {
    ".tif":  True,    # Pyramided GeoTIFF — the main deliverable for pipeline
    ".xml":  True,    # PDS4 XML label — metadata (lat/lon, incidence, etc.)
    ".img":  False,   # Full calibrated IMG — huge (~500 MB), skip by default
    ".kml":  False,   # KML footprint
    ".tar.gz": False, # Shapefile archive
    ".zip":  False,   # Shapefile archive
}

# Default output paths (match CONFIGURATION.md §1)
PATHS = {
    "nac":      "data/reference/nac",
    "wac":      "data/reference",
    "wac_mono": "data/reference",
    "lola":     "data/reference/lola",
}

# WAC 643nm global mosaic direct download URL (LROC hosted)
# NOTE: host/path not verified; WAC mode currently uses ODE CDRWAC4 strips.
WAC_643NM_MOSAIC_URL = (
    "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/"
    "LROLRC_2001/DATA/BDR/WAC_GLOBAL_E300N3150_100M.TIF"
)


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _manifest_ids(path):
    ids = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                ids.add(json.loads(line).get("product_id"))
            except json.JSONDecodeError:
                pass
    return ids


# ─── HTTP Helpers ───────────────────────────────────────────────────────────

def _request_json(url: str) -> dict:
    """GET request → parsed JSON. Follows redirects."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"  ❌ Network error: {e.reason}")
        raise


def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a file with retry logic and progress display."""
    if dest.exists():
        print(f"    ⏭️  Already exists: {dest.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 256 * 1024  # 256 KB

                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            mb = downloaded / 1e6
                            print(
                                f"\r    ⬇️  {dest.name}  {mb:.1f} MB  ({pct:.0f}%)",
                                end="", flush=True,
                            )

            print(f"\r    ✅ {dest.name}  ({downloaded / 1e6:.1f} MB)               ")
            return True

        except Exception as e:
            print(f"\n    ⚠️  Attempt {attempt}/{retries} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                time.sleep(2 * attempt)

    print(f"    ❌ Failed to download {dest.name} after {retries} attempts")
    return False


# ─── ODE Query ──────────────────────────────────────────────────────────────

def query_ode_nac(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit: int = 10,
    product_type: str = "CDRNAC4",
) -> list[dict]:
    """
    Query NASA ODE for LRO NAC/WAC products covering a bounding box.

    Two-phase query:
      Phase 1 (results=m):  geographic metadata (lat, lon, incidence, resolution)
      Phase 2 (results=fp): file download URLs

    Merged by Product_name (the join key between both result sets).
    """
    min_lon = min_lon % 360.0
    max_lon = max_lon % 360.0
    if max_lon - min_lon >= 359.0:
        print("  ERROR: longitude range spans the globe - refine the ROI")
        return []
    if min_lon > max_lon:
        print("  ERROR: longitude range wraps 360/0 - split the ROI")
        return []

    base_params = {
        "target": "moon",
        "ihid": "LRO",
        "iid": "LROC",
        "pt": product_type,
        "minlat": min_lat,
        "maxlat": max_lat,
        "westernlon": min_lon,
        "easternlon": max_lon,
        "output": "JSON",
        "limit": limit,
    }

    print(f"\n🛰️  Querying ODE:  Lat [{min_lat}, {max_lat}]  Lon [{min_lon}, {max_lon}]")
    print(f"   Product type: {product_type}  |  Limit: {limit}")

    # ── Phase 1: metadata ──
    url_meta = f"{ODE_BASE}?{urllib.parse.urlencode({**base_params, 'results': 'm'})}"
    data_meta = _request_json(url_meta)

    if data_meta.get("ODEResults", {}).get("Status") != "Success":
        err = data_meta.get("ODEResults", {}).get("Error", "Unknown error")
        print(f"  ❌ ODE error: {err}")
        return []

    meta_list = data_meta["ODEResults"].get("Products", {}).get("Product", [])
    if isinstance(meta_list, dict):
        meta_list = [meta_list]

    total_count = data_meta["ODEResults"].get("Count", len(meta_list))
    print(f"  ✅ Found {total_count} total product(s), fetching {len(meta_list)}")

    if not meta_list:
        return []

    # ── Phase 2: file URLs ──
    url_files = f"{ODE_BASE}?{urllib.parse.urlencode({**base_params, 'results': 'fp'})}"
    data_files = _request_json(url_files)

    file_list = data_files.get("ODEResults", {}).get("Products", {}).get("Product", [])
    if isinstance(file_list, dict):
        file_list = [file_list]

    # Build lookup: LabelFileName → file list
    # (LabelFileName e.g. "M1410460825LC.xml" matches Product_name "M1410460825LC.IMG")
    files_by_label: dict[str, list] = {}
    for fp in file_list:
        label = fp.get("LabelFileName", "")
        stem = Path(label).stem  # "M1410460825LC"
        pf = fp.get("Product_files", {}).get("Product_file", [])
        if isinstance(pf, dict):
            pf = [pf]
        files_by_label[stem] = pf

    # ── Merge ──
    for product in meta_list:
        pname = product.get("Product_name", "")
        stem = Path(pname).stem  # "M1410460825LC"
        product["_file_list"] = files_by_label.get(stem, [])

    return meta_list


def _extract_metadata(product: dict) -> dict:
    """Extract a clean metadata dict from an ODE product record."""
    pname = product.get("Product_name", "unknown")
    product_id = Path(pname).stem  # e.g. "M1410460825LC"

    return {
        "product_id": product_id,
        "product_name": pname,
        "center_lat": float(product.get("Center_latitude", 0)),
        "center_lon": float(product.get("Center_longitude", 0)),
        "min_lat": float(product.get("Minimum_latitude", 0)),
        "max_lat": float(product.get("Maximum_latitude", 0)),
        "west_lon": float(product.get("Westernmost_longitude", 0)),
        "east_lon": float(product.get("Easternmost_longitude", 0)),
        "incidence_angle_deg": float(product.get("Incidence_angle", 0)),
        "emission_angle_deg": float(product.get("Emission_angle", 0)),
        "phase_angle_deg": float(product.get("Phase_angle", 0)),
        "resolution_m_per_px": float(product.get("Map_resolution", 0)),
        "observation_utc": product.get("Observation_time", ""),
        "orbit": product.get("Start_orbit_number", ""),
        "footprint_wkt": product.get("Footprint_C0_geometry", ""),
        "solar_distance_km": float(product.get("Solar_distance", 0)),
        "solar_longitude_deg": float(product.get("Solar_longitude", 0)),
        "ref_type": "NAC",  # INTERFACES.md PairRecord.ref.type
        "files_downloaded": [],
    }


# ─── Download Logic ─────────────────────────────────────────────────────────

def download_product(product: dict, output_dir: Path, download_img: bool = False) -> dict:
    """Download desired files for a single ODE product. Returns metadata dict."""
    meta = _extract_metadata(product)

    print(f"\n  📦 Product: {meta['product_id']}")
    print(f"     Lat: {meta['center_lat']:.3f}  Lon: {meta['center_lon']:.3f}")
    print(f"     Incidence: {meta['incidence_angle_deg']:.1f}°  "
          f"Resolution: {meta['resolution_m_per_px']:.2f} m/px")
    print(f"     Observation: {meta['observation_utc']}")
    print(f"     Orbit: {meta['orbit']}")

    files = product.get("_file_list", [])
    if not files:
        print(f"     ⚠️  No file URLs available for this product")
        return meta

    for finfo in files:
        fname = finfo.get("FileName", "")
        furl = finfo.get("URL", "")

        ext = "".join(Path(fname).suffixes).lower()  # handles .tar.gz
        want = DOWNLOAD_PREFS.get(ext, False)
        if ext == ".img" and download_img:
            want = True

        if not want:
            continue

        dest = output_dir / fname
        if _download_file(furl, dest):
            meta["files_downloaded"].append(str(dest))

    if meta["files_downloaded"] and all("_PYR" in f for f in meta["files_downloaded"]):
        print("     WARNING: only BROWSE (_PYR) files downloaded - science product is the .IMG; pass --download-img for full resolution")
    return meta


# ─── Public API (called by build_pairs.py) ──────────────────────────────────

def fetch_nac_for_footprint(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    output_dir: str = PATHS["nac"],
    limit: int = 10,
    max_incidence_deg: Optional[float] = None,
    min_res_m: Optional[float] = None,
    max_res_m: Optional[float] = None,
    download_img: bool = False,
) -> list[dict]:
    """
    Query ODE and download calibrated LRO NAC products for a bounding box.

    This is the function that build_pairs.py calls during S2 (PIPELINE.md).
    It implements the first entry in reference_fallback_chain (CONFIGURATION.md §2.2).

    Parameters
    ----------
    min_lat, max_lat : float
        Latitude bounds (degrees, planetocentric).
    min_lon, max_lon : float
        Longitude bounds (degrees East, 0–360).
    output_dir : str
        Where to save downloaded files.
    limit : int
        Max number of products to retrieve.
    max_incidence_deg : float, optional
        If set, filter out products with incidence angle above this.
    download_img : bool
        If True, also download the large .IMG files.

    Returns
    -------
    list[dict]
        Metadata dicts for each downloaded product (for manifest.jsonl).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Over-query so post-filters still leave `limit` usable products
    # (ODE cannot filter by incidence/resolution server-side).
    products = query_ode_nac(min_lat, max_lat, min_lon, max_lon, limit * OVERQUERY)

    if max_incidence_deg is not None:
        before = len(products)
        products = [
            p for p in products
            if _safe_float(p.get("Incidence_angle"), 999.0) <= max_incidence_deg
        ]
        print("  🔍 Filtered incidence <= " + str(max_incidence_deg) + " deg: " + str(before) + " -> " + str(len(products)))

    if min_res_m is not None:
        products = [p for p in products
                    if _safe_float(p.get("Map_resolution"), 0.0) >= min_res_m]
    if max_res_m is not None:
        products = [p for p in products
                    if _safe_float(p.get("Map_resolution"), 0.0) <= max_res_m]

    # prefer best illumination, then finest resolution; trim to limit
    products.sort(key=lambda p: (_safe_float(p.get("Incidence_angle"), 999.0),
                                 _safe_float(p.get("Map_resolution"), 999.0)))
    products = products[:limit]

    results = []
    already = _manifest_ids(out / "manifest.jsonl")
    for p in products:
        meta = _extract_metadata(p)
        pid = meta["product_id"]
        if pid in already:
            print("  skip " + pid + " already in manifest")
            continue
        meta = download_product(p, out, download_img)
        results.append(meta)

    # Write manifest
    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n📋 Manifest appended: {manifest_path}  ({len(results)} entries)")

    return results


def fetch_wac_for_footprint(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    output_dir: str = PATHS["wac"],
    limit: int = 5,
    download_img: bool = False,
    max_incidence_deg: Optional[float] = None,
    min_res_m: Optional[float] = None,
    max_res_m: Optional[float] = None,
) -> list[dict]:
    """
    Query ODE and download calibrated LRO WAC products for a bounding box.
    Second entry in reference_fallback_chain (CONFIGURATION.md §2.2).
    """
    out = Path(output_dir) / "wac"
    out.mkdir(parents=True, exist_ok=True)

    products = query_ode_nac(min_lat, max_lat, min_lon, max_lon, limit * OVERQUERY, "CDRWAC4")

    if max_incidence_deg is not None:
        before = len(products)
        products = [
            p for p in products
            if _safe_float(p.get("Incidence_angle"), 999.0) <= max_incidence_deg
        ]
        print("  🔍 WAC filtered incidence <= " + str(max_incidence_deg) + " deg: " + str(before) + " -> " + str(len(products)))

    if min_res_m is not None:
        products = [p for p in products
                    if _safe_float(p.get("Map_resolution"), 0.0) >= min_res_m]
    if max_res_m is not None:
        products = [p for p in products
                    if _safe_float(p.get("Map_resolution"), 0.0) <= max_res_m]

    products.sort(key=lambda p: (_safe_float(p.get("Incidence_angle"), 999.0),
                                 _safe_float(p.get("Map_resolution"), 999.0)))
    products = products[:limit]

    results = []
    already = _manifest_ids(out / "manifest.jsonl")
    for p in products:
        meta = _extract_metadata(p)
        pid = meta["product_id"]
        if pid in already:
            print("  skip " + pid + " already in manifest")
            continue
        meta = download_product(p, out, download_img)
        meta["ref_type"] = "WAC"
        results.append(meta)

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n📋 WAC manifest appended: {manifest_path}  ({len(results)} entries)")

    return results


# ─── Batch Mode ─────────────────────────────────────────────────────────────

def fetch_from_manifest_csv(
    csv_path: str,
    product_type: str = "nac",
    output_base: str = PATHS["nac"],
    limit: int = 5,
    download_img: bool = False,
    max_incidence_deg: Optional[float] = None,
    min_res_m: Optional[float] = None,
    max_res_m: Optional[float] = None,
):
    """Batch-download for multiple ROIs defined in a CSV manifest."""
    print(f"\n📂 Batch mode: reading {csv_path}")

    fetch_fn = fetch_nac_for_footprint if product_type == "nac" else fetch_wac_for_footprint

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roi = row.get("roi_name", "unnamed")
            print(f"\n{'='*60}")
            print(f"  🌍 ROI: {roi}")
            print(f"{'='*60}")

            fetch_fn(
                min_lat=float(row["min_lat"]),
                max_lat=float(row["max_lat"]),
                min_lon=float(row["min_lon"]),
                max_lon=float(row["max_lon"]),
                output_dir=output_base,
                limit=limit,
                download_img=download_img,
                max_incidence_deg=max_incidence_deg,
                min_res_m=min_res_m,
                max_res_m=max_res_m,
            )


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated LRO Reference Data Downloader (NASA ODE REST API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 5 calibrated NAC strips:
  python scripts/fetch_lroc_nac.py nac --min-lat -6.5 --max-lat -5.5 \\
                                             --min-lon 1.0  --max-lon 2.0  --limit 5

  # Filter only well-illuminated (low-incidence) images:
  python scripts/fetch_lroc_nac.py nac --max-incidence 70 --limit 10

  # Download WAC strips:
  python scripts/fetch_lroc_nac.py wac --min-lat -6.5 --max-lat -5.5

  # Batch download from manifest CSV:
  python scripts/fetch_lroc_nac.py nac --manifest data/metadata/roi_manifest.csv
        """,
    )

    parser.add_argument(
        "product",
        choices=["nac", "wac"],
        help="Which LRO product to download: 'nac' (0.5 m/px) or 'wac' (100 m/px)",
    )
    parser.add_argument("--min-lat", type=float, default=-6.5)
    parser.add_argument("--max-lat", type=float, default=-5.5)
    parser.add_argument("--min-lon", type=float, default=1.0)
    parser.add_argument("--max-lon", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=5, help="Max products (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--max-incidence", type=float, default=None,
                        help="Max solar incidence angle filter (degrees)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="CSV manifest for batch downloads")
    parser.add_argument("--download-img", action="store_true",
                        help="Also download large .IMG files (default: GeoTIFF only)")
    parser.add_argument("--min-res", type=float, default=None,
                        help="Min GSD filter (m/px)")
    parser.add_argument("--max-res", type=float, default=None,
                        help="Max GSD filter (m/px)")

    args = parser.parse_args()

    output = args.output or PATHS.get(args.product, PATHS["nac"])

    print("=" * 60)
    print("  🚀 LRO Reference Data Downloader  (NASA ODE REST API)")
    print("  📡 SIH 2026 — PS-26166 Lunar Image Registration")
    print(f"  📦 Product: {args.product.upper()}")
    print("=" * 60)

    if args.manifest:
        fetch_from_manifest_csv(
            args.manifest, args.product, output, args.limit,
            download_img=args.download_img,
            max_incidence_deg=args.max_incidence,
            min_res_m=args.min_res,
            max_res_m=args.max_res,
        )
    elif args.product == "nac":
        fetch_nac_for_footprint(
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            output_dir=output,
            limit=args.limit,
            max_incidence_deg=args.max_incidence,
            download_img=args.download_img,
            min_res_m=args.min_res,
            max_res_m=args.max_res,
        )
    elif args.product == "wac":
        fetch_wac_for_footprint(
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            output_dir=output,
            limit=args.limit,
            download_img=args.download_img,
            max_incidence_deg=args.max_incidence,
            min_res_m=args.min_res,
            max_res_m=args.max_res,
        )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

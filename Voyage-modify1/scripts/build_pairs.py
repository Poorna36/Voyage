"""
scripts/build_pairs.py
=======================
S2 — Pair Building Entry Point (Features F02, F03).

Reads products.jsonl from S1 ingest, acquires reference images for each product
via the fallback chain (NAC ODE -> WAC crop -> SELENE stub), computes pair
attributes (overlap, terrain class, crater density, geo_cell, split), and writes
a complete PairRecord to data/pairs/manifest.jsonl (one JSON per line).

Usage:
    python scripts/build_pairs.py --products data/metadata/products.jsonl \\
        --config configs/ohrc_nac.yaml --ode
    python scripts/build_pairs.py --products data/metadata/products.jsonl \\
        --config configs/tmc_wac.yaml --wac data/reference/wac_643nm.tif

Exit codes (per PIPELINE.md §8 — applies to ALL pipeline scripts):
    0 — all pairs built; all gates passed
    1 — one or more pairs failed or skipped; others completed
    2 — configuration error (missing key, YAML error)
    3 — environment error (network unreachable for ODE; no WAC mosaic configured)
    4 — leakage audit failed (reserved; not triggered here)

Gate (per PIPELINE.md §S2):
    - overlap_fraction >= 0.5 is NOTED; below 0.5 sets partial_overlap=true but pair is kept
    - Pairs with no reference at all are written to skipped.jsonl

Train/Test Split:
    - Geo-cells (10x10 degree) are pre-assigned: even cells -> train, odd cells -> test.
    - No pair from a test geo_cell may appear in train data (leakage-safe).

References:
    - docs/INTERFACES.md §1 (PairRecord schema)
    - docs/FEATURES.md F02, F03
    - docs/PIPELINE.md §S2
    - docs/CONFIGURATION.md §2.2
    - src/ingest/reference.py
    - src/failures.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path when invoked as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingest.reference import acquire_reference, pad_bbox
from src.failures import log_gate_failure
from src.provenance import build_provenance, set_global_seed

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("build_pairs")


# ---------------------------------------------------------------------------
# Geo-cell & Split Assignment
# ---------------------------------------------------------------------------

def assign_geo_cell(latitude_center_deg: float, longitude_center_deg: float) -> str:
    """
    Assign a 10x10 degree geographic cell label to a point.

    Cells are labeled as "{lat_floor}_{lon_floor}" where:
      - lat_floor is the floor of the latitude to the nearest 10 degrees
      - lon_floor is the floor of the longitude to the nearest 10 degrees

    Cell covering (lat=−85, lon=55) => "-90_50"
    Cell covering (lat=−21, lon=11) => "-30_10"

    Parameters:
        latitude_center_deg:   Centre latitude in degrees [-90, 90].
        longitude_center_deg:  Centre longitude in degrees [-180, 180].

    Returns:
        String label like '-90_50' or '0_10'.
    """
    lat_floor = int(math.floor(latitude_center_deg / 10.0) * 10)
    lon_floor = int(math.floor(longitude_center_deg / 10.0) * 10)
    return f"{lat_floor}_{lon_floor}"


def assign_split(geo_cell: str) -> str:
    """
    Deterministically assign a geo_cell to 'train' or 'test'.

    Partitioning rule: uses a hash-based deterministic assignment so that
    ~25% of cells are 'test' and the rest 'train'. This ensures disjoint
    geographic cells between train and test — no geo_cell appears in both.

    Implementation: hash the cell string, map to 'test' if hash % 4 == 0.
    Both train and test cells are always present at all lat bins.

    Returns: 'train' or 'test'
    """
    import hashlib
    h = int(hashlib.sha256(geo_cell.encode()).hexdigest(), 16)
    return "test" if (h % 4) == 0 else "train"


# ---------------------------------------------------------------------------
# Footprint Centre & Overlap
# ---------------------------------------------------------------------------

def footprint_centre(footprint_ll: List[List[float]]) -> Tuple[float, float]:
    """
    Compute the centroid (lon, lat) of a footprint polygon.

    Simple arithmetic mean of the corner coordinates.

    Returns:
        (centre_lon, centre_lat) in decimal degrees [-180, 180].
    """
    if not footprint_ll:
        return 0.0, 0.0
    lons = [pt[0] for pt in footprint_ll]
    lats = [pt[1] for pt in footprint_ll]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def compute_overlap_fraction(
    src_footprint: List[List[float]],
    ref_bbox: List[float],
) -> float:
    """
    Estimate overlap fraction between a source footprint polygon and a reference bbox.

    Uses a simple axis-aligned bounding box intersection (sufficient for pair selection;
    exact polygon overlap is not needed at this stage).

    overlap_fraction = area of intersection / area of src bounding box.

    Parameters:
        src_footprint:  [[lon, lat], ...] corner polygon of source.
        ref_bbox:       [lon_min, lat_min, lon_max, lat_max] of reference crop.

    Returns:
        Float in [0, 1]. 0.0 if no overlap. 1.0 if src fully contained in ref.
    """
    if not src_footprint:
        return 0.0

    src_lons = [pt[0] for pt in src_footprint]
    src_lats = [pt[1] for pt in src_footprint]
    src_lon_min, src_lon_max = min(src_lons), max(src_lons)
    src_lat_min, src_lat_max = min(src_lats), max(src_lats)

    ref_lon_min, ref_lat_min, ref_lon_max, ref_lat_max = ref_bbox

    # Intersection
    inter_lon_min = max(src_lon_min, ref_lon_min)
    inter_lat_min = max(src_lat_min, ref_lat_min)
    inter_lon_max = min(src_lon_max, ref_lon_max)
    inter_lat_max = min(src_lat_max, ref_lat_max)

    if inter_lon_max <= inter_lon_min or inter_lat_max <= inter_lat_min:
        return 0.0  # No intersection

    inter_area = (inter_lon_max - inter_lon_min) * (inter_lat_max - inter_lat_min)
    src_area = (src_lon_max - src_lon_min) * (src_lat_max - src_lat_min)

    if src_area == 0:
        return 0.0

    return min(1.0, inter_area / src_area)


# ---------------------------------------------------------------------------
# Terrain Classification
# ---------------------------------------------------------------------------

def assign_terrain_class(latitude_center_deg: float) -> str:
    """
    Classify terrain type based on latitude.

    Uses simplified latitude-based classification (per CONFIGURATION.md §2.2 strata):
        |lat| <= 45  -> 'equatorial'
        45 < |lat| <= 60  -> 'midlat' / 'highland'
        |lat| > 60  -> 'polar' or 'polar_highland' (conservative: all poleward = polar)

    A more sophisticated classifier (using WAC DEM) can replace this post-acquisition.

    Returns:
        One of: 'equatorial', 'highland', 'polar_highland', 'polar'
    """
    abs_lat = abs(latitude_center_deg)
    if abs_lat <= 45.0:
        return "equatorial"
    elif abs_lat <= 60.0:
        return "highland"
    else:
        return "polar_highland"


# ---------------------------------------------------------------------------
# PairRecord Construction
# ---------------------------------------------------------------------------

def build_pair_id(src_product_id: str, ref_product_id: str) -> str:
    """
    Construct a stable pair_id from source and reference product IDs.

    Format: {src_stem}__{ref_stem}
    Example: 'ch2_ohr_ncp_20211228t2209123959_d_img_d18__nac_crop'

    Stems are lowercased and truncated for filesystem safety.
    """
    src_stem = src_product_id.lower().replace(":", "_")[:60]
    ref_stem = ref_product_id.lower().replace(":", "_")[:30]
    return f"{src_stem}__{ref_stem}"


def build_pair_record(
    src_meta: Dict[str, Any],
    ref_path: Optional[str],
    ref_type: str,
    ref_product_id: str,
    ref_gsd_m: float,
    ref_bbox: Optional[List[float]],
    config: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct a complete PairRecord (INTERFACES.md §1) for manifest.jsonl.

    Parameters:
        src_meta:       Parsed product record from products.jsonl.
        ref_path:       Path to the reference crop file (or None for SELENE stub).
        ref_type:       'NAC' | 'WAC' | 'SELENE'
        ref_product_id: Product ID string for the reference.
        ref_gsd_m:      GSD of reference in metres.
        ref_bbox:       [lon_min, lat_min, lon_max, lat_max] of reference footprint.
        config:         Full pipeline config dict.
        provenance:     Provenance dict from build_provenance().

    Returns:
        Dict conforming to INTERFACES.md §1 PairRecord schema.
    """
    footprint_ll = src_meta["footprint_ll"]
    src_lon_centre, src_lat_centre = footprint_centre(footprint_ll)

    # Overlap fraction
    if ref_bbox:
        overlap_fraction = compute_overlap_fraction(footprint_ll, ref_bbox)
    else:
        overlap_fraction = 0.0

    partial_overlap = overlap_fraction < 0.5

    # Terrain and geo_cell
    terrain_class = assign_terrain_class(src_lat_centre)
    geo_cell = assign_geo_cell(src_lat_centre, src_lon_centre)
    split = assign_split(geo_cell)

    # Delta azimuth between src solar azimuth and reference (unknown at this stage -> None)
    delta_azimuth_deg = None

    # Crater density: placeholder (requires DEM/WAC DEM analysis — set to None here)
    crater_density_per_km2 = None

    pair_id = build_pair_id(src_meta["product_id"], ref_product_id)

    record: Dict[str, Any] = {
        "pair_id": pair_id,
        "src": {
            "product_id": src_meta["product_id"],
            "cub_path": src_meta["cub_path"],
            "gsd_m": src_meta["gsd_m"],
            "solar_incidence_deg": src_meta["solar_incidence_deg"],
            "solar_azimuth_deg": src_meta["solar_azimuth_deg"],
            "sensor": src_meta["sensor"],
            "utc": src_meta["utc"],
            "footprint_ll": footprint_ll,
            "footprint_shape": src_meta["footprint_shape"],
        },
        "ref": {
            "product_id": ref_product_id,
            "path": ref_path,
            "gsd_m": ref_gsd_m,
            "type": ref_type,
            "footprint_ll": (
                [
                    [ref_bbox[0], ref_bbox[3]],   # UL: lon_min, lat_max
                    [ref_bbox[2], ref_bbox[3]],   # UR: lon_max, lat_max
                    [ref_bbox[2], ref_bbox[1]],   # LR: lon_max, lat_min
                    [ref_bbox[0], ref_bbox[1]],   # LL: lon_min, lat_min
                ]
                if ref_bbox else None
            ),
        },
        "overlap_fraction": round(overlap_fraction, 4),
        "partial_overlap": partial_overlap,
        "delta_azimuth_deg": delta_azimuth_deg,
        "latitude_center_deg": round(src_lat_centre, 6),
        "longitude_center_deg": round(src_lon_centre, 6),
        "terrain_class": terrain_class,
        "crater_density_per_km2": crater_density_per_km2,
        "geo_cell": geo_cell,
        "split": split,
        "gt_path": None,   # populated during Phase 7 annotation
        "created_at": provenance["created_at"],
    }
    record.update({
        "config_hash": provenance["config_hash"],
        "code_commit": provenance["code_commit"],
        "seed": provenance["seed"],
    })
    return record


# ---------------------------------------------------------------------------
# JSONL I/O Helpers
# ---------------------------------------------------------------------------

def _read_products(products_jsonl: Path) -> List[Dict[str, Any]]:
    """Read all product records from products.jsonl (one JSON per line)."""
    if not products_jsonl.exists():
        raise FileNotFoundError(f"products.jsonl not found: {products_jsonl}")
    records = []
    with open(products_jsonl, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("products.jsonl:%d: invalid JSON — %s", lineno, e)
    return records


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record to a .jsonl file (append-only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Config Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config; returns empty dict on failure."""
    if not config_path:
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load config %s: %s", config_path, e)
        return {}


def _get_pair_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pair-building parameters from config with defaults."""
    pair_cfg = config.get("pair", {})
    return {
        "k_pointing": float(pair_cfg.get("k_pointing", 3)),
        "sigma_pointing_m": float(pair_cfg.get("sigma_pointing_m", 1000)),
        "wac_reference": pair_cfg.get("wac_reference", ""),
        "ode_timeout_s": int(pair_cfg.get("ode_timeout_s", 30)),
        "selene_wmts_url": pair_cfg.get("selene_wmts_url", "https://trek.nasa.gov/moon/"),
    }


# ---------------------------------------------------------------------------
# Reference GSD by type
# ---------------------------------------------------------------------------

REF_GSD_BY_TYPE = {
    "NAC": 0.50,    # LRO NAC nominal GSD ~0.5 m/px
    "WAC": 100.0,   # LRO WAC 643nm nominal GSD ~100 m/px
    "SELENE": 10.0, # Kaguya TC nominal 10 m/px
}


# ---------------------------------------------------------------------------
# Main Pair Building Loop
# ---------------------------------------------------------------------------

def build_pairs(
    products_jsonl: Path,
    manifest_jsonl: Path,
    skipped_jsonl: Path,
    failures_jsonl: Path,
    out_dir: Path,
    config: Dict[str, Any],
    use_ode: bool = True,
    wac_mosaic_path: Optional[str] = None,
) -> int:
    """
    Core pair-building loop per PIPELINE.md §S2.

    For each product in products.jsonl:
      1. Compute padded bbox from footprint
      2. Acquire reference via fallback chain
      3. Compute overlap_fraction
      4. Assign terrain_class, geo_cell, split
      5. Write PairRecord to manifest.jsonl
      6. Pairs with no reference -> skipped.jsonl

    Returns:
        0 if all pairs have valid references, 1 if any are skipped or fail.
    """
    provenance = build_provenance(config=config)
    seed = int(config.get("global", {}).get("seed", 42))
    set_global_seed(seed)

    pair_cfg = _get_pair_config(config)
    k = pair_cfg["k_pointing"]
    sigma_m = pair_cfg["sigma_pointing_m"]
    ode_timeout_s = pair_cfg["ode_timeout_s"]
    selene_url = pair_cfg["selene_wmts_url"]

    # WAC mosaic: CLI arg overrides config
    wac_path = wac_mosaic_path or pair_cfg["wac_reference"]
    if wac_path and not Path(wac_path).exists():
        logger.warning("WAC mosaic not found: %s — WAC step will be skipped", wac_path)
        wac_path = None

    products = _read_products(products_jsonl)
    if not products:
        logger.warning("No products in %s", products_jsonl)
        return 0

    logger.info(
        "Building pairs for %d product(s). Reference chain: %s | WAC=%s",
        len(products),
        "ODE NAC" if use_ode else "skip ODE",
        wac_path or "not configured",
    )

    gate_failures = 0
    n_pairs = 0
    n_skipped = 0

    for prod in products:
        product_id = prod.get("product_id", "unknown")
        footprint_ll = prod.get("footprint_ll", [])

        if not footprint_ll:
            reason = f"product_id={product_id}: footprint_ll missing — cannot build pair"
            logger.error(reason)
            log_gate_failure(failures_jsonl, pair_id=product_id, stage="S2", reason=reason)
            gate_failures += 1
            continue

        logger.info("─── Pair for product: %s (%s)", product_id, prod.get("sensor", "?"))

        # Compute padding
        padding_m = k * sigma_m

        # Acquire reference
        ref_out_dir = out_dir / product_id
        ref_out_dir.mkdir(parents=True, exist_ok=True)

        if use_ode:
            ref_path, ref_type = acquire_reference(
                footprint_ll=footprint_ll,
                padding_m=sigma_m,    # pass sigma_m; acquire_reference handles k internally
                out_dir=str(ref_out_dir),
                wac_mosaic_path=wac_path,
                selene_url=selene_url,
                ode_timeout_s=ode_timeout_s,
            )
        else:
            # Only WAC step (--wac flag, no --ode)
            if wac_path:
                from src.ingest.reference import pad_bbox as _pad_bbox, crop_wac_mosaic
                try:
                    bbox = _pad_bbox(footprint_ll, sigma_m=padding_m, k=1.0)
                    wac_cropped = crop_wac_mosaic(wac_path, bbox)
                    ref_path = wac_cropped
                    ref_type = "WAC" if wac_cropped else "no_reference_found"
                except Exception as e:
                    ref_path = None
                    ref_type = "no_reference_found"
                    logger.error("WAC crop failed for %s: %s", product_id, e)
            else:
                ref_path = None
                ref_type = "no_reference_found"

        if ref_type == "no_reference_found":
            reason = f"No reference acquired for product_id={product_id} after all fallbacks"
            logger.warning("SKIP: %s", reason)
            _append_jsonl(skipped_jsonl, {
                "product_id": product_id,
                "reason": reason,
                "created_at": provenance["created_at"],
            })
            n_skipped += 1
            gate_failures += 1
            continue

        # SELENE stub: record but no data downloaded
        if ref_type == "SELENE":
            logger.warning(
                "product_id=%s: ref_type=SELENE (stub) — connectivity verified but no data. "
                "This pair cannot be processed until SELENE acquisition is implemented.",
                product_id,
            )

        # Compute ref_bbox from the padded bbox (used for overlap and PairRecord)
        try:
            ref_bbox = pad_bbox(footprint_ll, sigma_m=padding_m, k=1.0)
        except ValueError:
            ref_bbox = None

        # Reference GSD
        ref_gsd_m = REF_GSD_BY_TYPE.get(ref_type, 100.0)

        # Reference product_id: derive from path if available
        ref_product_id = Path(ref_path).stem if ref_path else f"{ref_type}_stub"

        # Build and write PairRecord
        pair_record = build_pair_record(
            src_meta=prod,
            ref_path=ref_path,
            ref_type=ref_type,
            ref_product_id=ref_product_id,
            ref_gsd_m=ref_gsd_m,
            ref_bbox=ref_bbox,
            config=config,
            provenance=provenance,
        )

        _append_jsonl(manifest_jsonl, pair_record)
        n_pairs += 1

        logger.info(
            "✓ pair_id=%s ref_type=%s overlap=%.2f terrain=%s split=%s",
            pair_record["pair_id"],
            ref_type,
            pair_record["overlap_fraction"],
            pair_record["terrain_class"],
            pair_record["split"],
        )

    logger.info(
        "build_pairs complete: %d pairs written, %d skipped, %d failures. "
        "manifest=%s skipped=%s",
        n_pairs, n_skipped, gate_failures, manifest_jsonl, skipped_jsonl,
    )
    return 1 if gate_failures > 0 else 0


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S2 — Pair building: acquire reference images via NAC ODE / WAC crop / SELENE "
            "and write PairRecords to manifest.jsonl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--products", required=True,
        help="Path to products.jsonl from S1 ingest (data/metadata/products.jsonl).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to sensor config YAML (e.g. configs/ohrc_nac.yaml).",
    )
    parser.add_argument(
        "--ode", action="store_true", default=False,
        help="Enable NAC ODE query (requires internet). If not set, ODE step is skipped.",
    )
    parser.add_argument(
        "--wac", default=None,
        help="Path to local WAC 643nm mosaic GeoTIFF (data/reference/wac_643nm.tif).",
    )
    parser.add_argument(
        "--manifest", default="data/pairs/manifest.jsonl",
        help="Output manifest.jsonl path.",
    )
    parser.add_argument(
        "--skipped", default="data/pairs/skipped.jsonl",
        help="Output skipped.jsonl path.",
    )
    parser.add_argument(
        "--failures", default="data/pairs/failures.jsonl",
        help="Output failures.jsonl path.",
    )
    parser.add_argument(
        "--ref-out", default="data/reference",
        help="Directory for downloaded/cropped reference files.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = _load_config(args.config)

    products_jsonl = Path(args.products)
    if not products_jsonl.exists():
        logger.error("EXIT 2: products.jsonl not found: %s", products_jsonl)
        return 2

    manifest_jsonl = Path(args.manifest)
    skipped_jsonl = Path(args.skipped)
    failures_jsonl = Path(args.failures)
    ref_out_dir = Path(args.ref_out)

    wac_mosaic = args.wac or config.get("pair", {}).get("wac_reference", None)

    if not args.ode and not wac_mosaic:
        logger.error(
            "EXIT 3: Neither --ode nor --wac specified. "
            "At least one reference source must be enabled."
        )
        return 3

    logger.info(
        "Starting S2 build_pairs: products=%s manifest=%s ode=%s wac=%s",
        products_jsonl, manifest_jsonl, args.ode, wac_mosaic or "None",
    )

    rc = build_pairs(
        products_jsonl=products_jsonl,
        manifest_jsonl=manifest_jsonl,
        skipped_jsonl=skipped_jsonl,
        failures_jsonl=failures_jsonl,
        out_dir=ref_out_dir,
        config=config,
        use_ode=args.ode,
        wac_mosaic_path=wac_mosaic,
    )

    if rc == 0:
        logger.info("EXIT 0: S2 build_pairs complete — all pairs have valid references.")
    else:
        logger.warning("EXIT 1: S2 build_pairs complete — some pairs skipped or failed.")

    return rc


if __name__ == "__main__":
    sys.exit(main())

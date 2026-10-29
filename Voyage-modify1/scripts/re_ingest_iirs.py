"""
re_ingest_iirs.py
=================
Re-parses all 8 real IIRS PDS4 XML labels using the fixed label_parser.py
(with multi-field solar angle extraction) and rewrites products.jsonl.

Run from the repo root:
    py -3.12 scripts/re_ingest_iirs.py
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingest.label_parser import parse_pds4_label

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("re_ingest_iirs")

REPO = Path(r"d:\neo\hachathon\SIH 2026\code")
RAW_IIRS = REPO / "data" / "raw" / "iirs"
PRODUCTS_JSONL = REPO / "data" / "metadata" / "products.jsonl"


def find_iirs_xml_labels(raw_dir: Path):
    """Find all IIRS data XML files (non-browse, non-geometry).
    Note: .qub files may not be present locally (large data cubes);
    metadata extraction only requires the XML + optional .hdr.
    """
    found = []
    for xml_path in raw_dir.rglob("*.xml"):
        if "browse" in str(xml_path).lower():
            continue
        if "geometry" in str(xml_path).lower():
            continue
        # Must be a real data label (check XML contains 'iir' in filename)
        if "iir" in xml_path.stem.lower():
            found.append(xml_path)
    return sorted(found)


def meta_to_record(meta, created_at: str) -> dict:
    d = {
        "product_id": meta.product_id,
        "cub_path": meta.cub_path,
        "gsd_m": meta.gsd_m,
        "solar_incidence_deg": meta.solar_incidence_deg,
        "solar_azimuth_deg": meta.solar_azimuth_deg,
        "sensor": meta.sensor,
        "utc": meta.utc,
        "footprint_ll": meta.footprint_ll,
        "footprint_shape": meta.footprint_shape,
        "processing_level": meta.processing_level,
        "spacecraft_altitude_km": meta.spacecraft_altitude_km,
        "xml_path": meta.xml_path,
        "created_at": created_at,
    }
    if meta.iirs_n_bands is not None:
        d["iirs_n_bands"] = meta.iirs_n_bands
    if meta.iirs_registration_band is not None:
        d["iirs_registration_band"] = meta.iirs_registration_band
    return d


def main():
    xml_labels = find_iirs_xml_labels(RAW_IIRS)
    logger.info("Found %d IIRS XML labels in %s", len(xml_labels), RAW_IIRS)

    records = []
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for xml_path in xml_labels:
        try:
            meta = parse_pds4_label(str(xml_path))
            record = meta_to_record(meta, created_at)
            records.append(record)
            logger.info(
                "  OK  %s | gsd=%.1fm | solar_inc=%.1f° | solar_az=%.1f° | bands=%s",
                meta.product_id, meta.gsd_m,
                meta.solar_incidence_deg, meta.solar_azimuth_deg,
                meta.iirs_n_bands,
            )
        except Exception as e:
            logger.error("  FAIL  %s: %s", xml_path.name, e)

    # Overwrite products.jsonl with fresh records
    PRODUCTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(PRODUCTS_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(
        "products.jsonl written with %d records → %s",
        len(records), PRODUCTS_JSONL,
    )

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'#':3} {'Product ID':45} {'GSD':8} {'solar_inc':10} {'solar_az':10}")
    print("-" * 70)
    for i, r in enumerate(records, 1):
        print(
            f"{i:3} {r['product_id']:45} {r['gsd_m']:7.1f}m "
            f"{r['solar_incidence_deg']:9.1f}° {r['solar_azimuth_deg']:9.1f}°"
        )
    print("=" * 70)
    print(f"Total: {len(records)} products | Solar angles: "
          f"{sum(1 for r in records if r['solar_incidence_deg'] > 0)} with non-zero incidence")


if __name__ == "__main__":
    main()

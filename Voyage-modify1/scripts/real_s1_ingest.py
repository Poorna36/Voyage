#!/usr/bin/env python3
"""
scripts/real_s1_ingest.py
=========================
S1 — real-product label ingestion (ISIS-free path).

Parses the verified real PDS4 labels available locally:
  - 1x Chandrayaan-2 OHRC calibrated product  (ch2_ohr_ncp_20211228T2209123959_d_img_d18)
  - 3x Chandrayaan-2 IIRS hyperspectral products
and writes data/metadata/products_real.jsonl with footprints, solar angles
and GSD for each (INTERFACES.md §1 ProductRecord fields).

Usage:
  python scripts/real_s1_ingest.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest.label_parser import parse_pds4_label  # noqa: E402

OHRC_XML = (
    _ROOT / "data/raw/ohrc/ch2_ohr_ncp_20211228T2209123959_d_img_d18"
    / "data/calibrated/20211228/ch2_ohr_ncp_20211228T2209123959_d_img_d18.xml"
)
IIRS_XMLS = [
    _ROOT / "data/raw/iirs/ch2_iir_nri_20240524T1710004919_d_img_d18/data/raw/20240524/ch2_iir_nri_20240524T1710004919_d_img_d18.xml",
    _ROOT / "data/raw/iirs/ch2_iir_nri_20240427T1010597893_d_img_d18/data/raw/20240427/ch2_iir_nri_20240427T1010597893_d_img_d18.xml",
    _ROOT / "data/raw/iirs/ch2_iir_nri_20210620T2058297275_d_img_hw1/data/raw/20210620/ch2_iir_nri_20210620T2058297275_d_img_hw1.xml",
]
TMC_XMLS = sorted(list((_ROOT / "data/raw/tmc").glob("**/*_d_img_*.xml")))


def main() -> int:
    xmls = [OHRC_XML] + IIRS_XMLS + TMC_XMLS
    records = []
    errors = []
    for xp in xmls:
        try:
            meta = parse_pds4_label(str(xp))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{xp.name}: {exc}")
            continue
        if not meta.footprint_ll or len(meta.footprint_ll) < 3:
            errors.append(f"{meta.product_id}: empty footprint")
            continue
        rec = asdict(meta)
        rec["cub_path"] = (
            str(xp.parent / f"{meta.product_id}.cub")
            .replace(str(_ROOT) + "/", "")
        )
        rec["xml_path"] = str(xp.relative_to(_ROOT))
        rec["created_at"] = "2026-09-01T00:00:00Z"
        records.append(rec)

    out = _ROOT / "data" / "metadata" / "products_real.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "products_written": len(records),
        "files": [r["product_id"] for r in records],
        "all_have_footprint": all(
            r["footprint_ll"] and len(r["footprint_ll"]) >= 3 for r in records),
        "all_have_solar": all(
            float(r["solar_incidence_deg"]) > 0 for r in records),
        "all_have_gsd": all(float(r["gsd_m"]) > 0 for r in records),
        "errors": errors,
    }
    (out.parent / "products_real_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if len(records) >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
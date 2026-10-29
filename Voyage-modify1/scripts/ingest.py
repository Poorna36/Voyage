"""
scripts/ingest.py
==================
S1 — Product Ingestion Entry Point (Feature F01).

Reads ISRO product archives from data/raw/, runs isisimport + spiceinit on each,
parses PDS4 XML labels, and writes one line per product to products.jsonl.

Usage:
    python scripts/ingest.py --raw data/raw --out data/calibrated \\
        --meta data/metadata/products.jsonl --kernels $ISISDATA

Exit codes (per PIPELINE.md §8 — applies to ALL pipeline scripts):
    0 — all products completed; all S1 gates passed
    1 — one or more products failed (spiceinit, footprint, missing fields); others completed
    2 — configuration error (missing argument, ISISDATA not set, ASP version < 3.7.0)
    3 — environment error (ASP/ISIS not on PATH; kernel fetch failed)
    4 — leakage audit failed (not triggered here; reserved)

Gate (per PIPELINE.md §S1):
    - spiceinit exits 0
    - footprint polygon non-empty (>= 3 corners)
    - solar angles present (non-zero)

On failure:
    - Failure written to failures.jsonl (stage=S1, reason, product_id)
    - Remaining products continue processing

References:
    - docs/FEATURES.md F01
    - docs/PIPELINE.md §S1
    - docs/CONFIGURATION.md §2.1
    - src/ingest/label_parser.py
    - src/failures.py
    - src/provenance.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path when invoked as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingest.label_parser import ProductMeta, parse_pds4_label, run_isisimport, run_spiceinit
from src.failures import log_gate_failure
from src.provenance import build_provenance, get_code_commit, set_global_seed

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# ASP Version Check
# ---------------------------------------------------------------------------

ASP_VERSION_MIN = (3, 7, 0)


def _check_asp_version() -> bool:
    """
    Verify ASP (stereo_gui) is >= 3.7.0. Returns True if OK, False otherwise.
    Logs error on failure; caller must raise SystemExit(3).
    """
    try:
        result = subprocess.run(
            ["stereo_gui", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        text = (result.stdout + result.stderr).strip()
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
        if not m:
            logger.error("ASP: could not parse version from: %s", text)
            return False
        ver = tuple(int(x) for x in m.groups())
        if ver < ASP_VERSION_MIN:
            logger.error(
                "ASP version %d.%d.%d < required %d.%d.%d",
                *ver, *ASP_VERSION_MIN,
            )
            return False
        logger.info("ASP version %d.%d.%d OK", *ver)
        return True
    except FileNotFoundError:
        logger.error("stereo_gui not found on PATH — is ASP activated?")
        return False
    except Exception as e:
        logger.error("ASP version check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Unzip Helpers
# ---------------------------------------------------------------------------

def _find_products_in_dir(search_root: Path) -> List[Path]:
    """
    Recursively find all PDS4 XML labels (excluding geometry/*) in a directory.

    Looks for .xml files paired with a .img or .qub file in the same directory.
    Skips geometry XML files (they describe geometry grids, not images).

    Returns list of XML paths.
    """
    xml_files = []
    for xml_path in search_root.rglob("*.xml"):
        # Skip geometry directory XMLs
        if "geometry" in xml_path.parts:
            continue
        # Must have a sibling .img or .qub
        has_img = xml_path.with_suffix(".img").exists()
        has_qub = xml_path.with_suffix(".qub").exists()
        if has_img or has_qub:
            xml_files.append(xml_path)
    return xml_files


def _unzip_if_needed(raw_dir: Path) -> List[Path]:
    """
    Unzip any .zip files in raw_dir in place, preserving original filenames.

    Returns list of unzipped directory paths (plus existing unzipped dirs).
    Product files (*.img, *.qub) are NEVER renamed.
    """
    extracted_dirs: List[Path] = []

    for zip_path in sorted(raw_dir.rglob("*.zip")):
        extract_to = zip_path.parent / zip_path.stem
        if extract_to.exists():
            logger.debug("Already unzipped: %s", zip_path.name)
            extracted_dirs.append(extract_to)
            continue
        logger.info("Unzipping %s -> %s", zip_path.name, extract_to)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            extracted_dirs.append(extract_to)
        except Exception as e:
            logger.error("Failed to unzip %s: %s", zip_path.name, e)

    return extracted_dirs


# ---------------------------------------------------------------------------
# products.jsonl Writer
# ---------------------------------------------------------------------------

def _meta_to_record(meta: ProductMeta, provenance: dict) -> dict:
    """
    Serialize a ProductMeta to a products.jsonl record dict.

    Only serializable fields are included. cub_path is stored as a string.
    Provenance (config_hash, code_commit, created_at, seed) is merged in.
    """
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
    }
    if meta.iirs_n_bands is not None:
        d["iirs_n_bands"] = meta.iirs_n_bands
    if meta.iirs_registration_band is not None:
        d["iirs_registration_band"] = meta.iirs_registration_band
    d.update(provenance)
    return d


def _append_product_record(products_jsonl: Path, record: dict) -> None:
    """Append one product record to products.jsonl (append-only, per F25 provenance rules)."""
    products_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(products_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.debug("Wrote to %s: %s", products_jsonl.name, record["product_id"])


# ---------------------------------------------------------------------------
# S1 Gate Check
# ---------------------------------------------------------------------------

def _check_s1_gate(meta: ProductMeta, spiceinit_ok: bool) -> List[str]:
    """
    Check S1 gate conditions (per PIPELINE.md §S1).

    Gate passes if:
      - spiceinit exits 0
      - footprint polygon non-empty
      - solar angles present (solar_incidence_deg != 0 or solar_azimuth_deg != 0)

    Returns list of failure reasons (empty list = gate passed).
    """
    failures = []
    if not spiceinit_ok:
        failures.append("spiceinit_failed")
    if not meta.footprint_ll or len(meta.footprint_ll) < 3:
        failures.append("footprint_empty_or_too_few_corners")
    if meta.solar_incidence_deg == 0.0 and meta.solar_azimuth_deg == 0.0:
        failures.append("solar_angles_missing")
    if meta.gsd_m <= 0:
        failures.append(f"gsd_m_invalid={meta.gsd_m}")
    if not meta.utc:
        failures.append("utc_missing")
    return failures


# ---------------------------------------------------------------------------
# Main Ingestion Loop
# ---------------------------------------------------------------------------

def ingest(
    raw_dir: Path,
    out_dir: Path,
    products_jsonl: Path,
    failures_path: Path,
    config: Optional[dict] = None,
    use_csm: str = "auto",
) -> int:
    """
    Core ingestion loop: find products, isisimport, spiceinit, parse label, write record.

    Parameters:
        raw_dir:          data/raw directory (searched recursively for XMLs + .img/.qub).
        out_dir:          data/calibrated directory (isisimport output).
        products_jsonl:   Append-only JSONL record file.
        failures_path:    Path to failures.jsonl.
        config:           Optional full config dict for provenance hashing.
        use_csm:          'auto' | 'yes' | 'no' for spiceinit CSM mode.

    Returns:
        0 if all products pass gate, 1 if any product fails gate.
    """
    provenance = build_provenance(config=config)
    seed = int((config or {}).get("global", {}).get("seed", 42))
    set_global_seed(seed)

    # Unzip any archives
    _unzip_if_needed(raw_dir)

    # Discover all product XML labels
    xml_paths = _find_products_in_dir(raw_dir)
    if not xml_paths:
        logger.warning("No PDS4 product XML labels found under %s", raw_dir)
        return 0

    logger.info("Found %d product(s) to ingest", len(xml_paths))

    gate_failures = 0
    processed = 0

    for xml_path in sorted(xml_paths):
        product_id = xml_path.stem  # fallback if parse fails
        logger.info("─── Processing: %s", xml_path.relative_to(raw_dir))

        # Determine source binary file
        img_path = xml_path.with_suffix(".img")
        qub_path = xml_path.with_suffix(".qub")
        if img_path.exists():
            src_binary = img_path
        elif qub_path.exists():
            src_binary = qub_path
        else:
            reason = f"No .img or .qub found adjacent to {xml_path}"
            logger.error("SKIP: %s", reason)
            log_gate_failure(failures_path, pair_id=xml_path.stem, stage="S1", reason=reason)
            gate_failures += 1
            continue

        # Step 1: Parse PDS4 XML label
        try:
            meta = parse_pds4_label(str(xml_path))
            product_id = meta.product_id
        except Exception as e:
            reason = f"parse_pds4_label failed: {e}"
            logger.error("product_id=%s SKIP: %s", product_id, reason)
            log_gate_failure(failures_path, pair_id=product_id, stage="S1", reason=reason)
            gate_failures += 1
            continue

        # Step 2: Run isisimport
        try:
            cub_path_str = run_isisimport(str(src_binary), str(out_dir))
            meta.cub_path = cub_path_str
            logger.info("isisimport OK: %s", Path(cub_path_str).name)
        except FileNotFoundError as e:
            reason = f"isisimport: source not found: {e}"
            logger.error("product_id=%s: %s", product_id, reason)
            log_gate_failure(failures_path, pair_id=product_id, stage="S1", reason=reason)
            gate_failures += 1
            continue
        except subprocess.CalledProcessError as e:
            reason = f"isisimport failed (rc={e.returncode}): {(e.stderr or '').strip()[:300]}"
            logger.error("product_id=%s: %s", product_id, reason)
            log_gate_failure(
                failures_path, pair_id=product_id, stage="S1", reason=reason,
                fallback_taken="check original ISRO filename — isisimport depends on it"
            )
            gate_failures += 1
            continue
        except subprocess.TimeoutExpired:
            reason = "isisimport timed out"
            logger.error("product_id=%s: %s", product_id, reason)
            log_gate_failure(failures_path, pair_id=product_id, stage="S1", reason=reason)
            gate_failures += 1
            continue

        # Step 3: Run spiceinit
        spiceinit_ok = run_spiceinit(meta.cub_path, use_csm=use_csm)
        if not spiceinit_ok:
            logger.warning(
                "product_id=%s: spiceinit failed — may need correct CK kernel window", product_id
            )

        # Step 4: S1 Gate check
        gate_reasons = _check_s1_gate(meta, spiceinit_ok)
        if gate_reasons:
            reason = "; ".join(gate_reasons)
            logger.error("product_id=%s: S1 GATE FAIL: %s", product_id, reason)
            log_gate_failure(
                failures_path, pair_id=product_id, stage="S1", reason=reason,
                fallback_taken=(
                    "fetch correct CK kernel window and re-run" if "spiceinit_failed" in reason else None
                )
            )
            gate_failures += 1
            continue

        # Step 5: Write to products.jsonl
        record = _meta_to_record(meta, provenance)
        _append_product_record(products_jsonl, record)
        processed += 1
        logger.info(
            "✓ product_id=%s sensor=%s gsd=%.2fm solar_inc=%.1f°",
            meta.product_id, meta.sensor, meta.gsd_m, meta.solar_incidence_deg,
        )

    logger.info(
        "Ingest complete: %d processed, %d gate failures. See %s",
        processed, gate_failures, products_jsonl,
    )
    return 1 if gate_failures > 0 else 0


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S1 — ISRO product ingestion: unzip, isisimport, spiceinit, parse PDS4 XML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw", required=True,
        help="Path to raw data directory (data/raw). Searched recursively for .img/.qub + .xml.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for .cub files (data/calibrated).",
    )
    parser.add_argument(
        "--meta", required=True,
        help="Path to products.jsonl append-only output file (data/metadata/products.jsonl).",
    )
    parser.add_argument(
        "--kernels", default=os.environ.get("ISISDATA", ""),
        help="ISISDATA directory (default: $ISISDATA env var). Required for spiceinit.",
    )
    parser.add_argument(
        "--use-csm", choices=["auto", "yes", "no"], default="auto",
        help="spiceinit CSM mode: auto (try CSM, fall back), yes, no. Default: auto.",
    )
    parser.add_argument(
        "--failures", default=None,
        help="Path to failures.jsonl (default: same dir as --meta).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file for provenance hashing.",
    )
    parser.add_argument(
        "--skip-asp-check", action="store_true",
        help="Skip ASP version check (useful when ISIS is on PATH but stereo_gui is not).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Environment checks ---
    isisdata = args.kernels or os.environ.get("ISISDATA", "")
    if not isisdata:
        logger.error(
            "EXIT 2: ISISDATA not set. Set $ISISDATA or pass --kernels. "
            "Required for spiceinit kernel lookup."
        )
        return 2

    if not Path(isisdata).is_dir():
        logger.error("EXIT 2: ISISDATA directory does not exist: %s", isisdata)
        return 2

    if not args.skip_asp_check:
        if not _check_asp_version():
            logger.error(
                "EXIT 3: ASP version check failed. Install ASP >= 3.7.0 from usgs-astrogeology."
            )
            return 3

    # --- Paths ---
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    products_jsonl = Path(args.meta)

    if not raw_dir.is_dir():
        logger.error("EXIT 2: raw directory does not exist: %s", raw_dir)
        return 2

    failures_path = (
        Path(args.failures) if args.failures
        else products_jsonl.parent / "failures.jsonl"
    )

    # --- Optional config loading ---
    config = None
    if args.config:
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning("Could not load config %s: %s — continuing with defaults", args.config, e)

    logger.info(
        "Starting S1 ingest: raw=%s out=%s meta=%s kernels=%s",
        raw_dir, out_dir, products_jsonl, isisdata,
    )

    rc = ingest(
        raw_dir=raw_dir,
        out_dir=out_dir,
        products_jsonl=products_jsonl,
        failures_path=failures_path,
        config=config,
        use_csm=args.use_csm,
    )

    if rc == 0:
        logger.info("EXIT 0: S1 ingest completed — all products passed gate.")
    else:
        logger.warning("EXIT 1: S1 ingest completed — some products failed gate. See failures.jsonl.")

    return rc


if __name__ == "__main__":
    sys.exit(main())

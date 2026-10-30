"""
scripts/setup_and_download.py
==============================
Downloads the SIH 2026 dataset from Google Drive into the correct
data/ directory structure, then validates what was downloaded.

Usage:
    python scripts/setup_and_download.py [--out data/raw] [--dry-run]

Requirements:
    pip install gdown

Drive folder: https://drive.google.com/drive/folders/1pPBNhzNVrs9jkjM6uqfAAmeiMJg5y25s
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DRIVE_FOLDER_ID = "1pPBNhzNVrs9jkjM6uqfAAmeiMJg5y25s"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

# Expected data directory structure (created if missing)
DATA_DIRS = [
    "data/raw",
    "data/raw/ohrc",
    "data/raw/iirs",
    "data/raw/tmc",
    "data/reference/nac",
    "data/reference/wac",
    "data/calibrated",
    "data/processed",
    "data/metadata",
    "data/pairs",
    "results",
    "results/pair_results",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_gdown() -> bool:
    """Install gdown if not already available."""
    try:
        import gdown  # noqa: F401
        print("[OK] gdown is already installed.")
        return True
    except ImportError:
        print("[INFO] Installing gdown...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
                check=True,
            )
            print("[OK] gdown installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install gdown: {e}", file=sys.stderr)
            return False


def _create_dirs(base: Path) -> None:
    """Create all required data directories."""
    print("\n[INFO] Creating directory structure...")
    for d in DATA_DIRS:
        target = base / d
        target.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {target.relative_to(base)}")


def _download_folder(out_dir: Path, dry_run: bool = False) -> bool:
    """Download the entire Drive folder using gdown."""
    import gdown  # noqa: F401 (installed above)

    print(f"\n[INFO] Downloading from Google Drive folder:")
    print(f"  URL : {DRIVE_FOLDER_URL}")
    print(f"  Dest: {out_dir}")

    if dry_run:
        print("[DRY-RUN] Would run: gdown --folder <url> -O <dest>")
        return True

    try:
        import gdown
        out_dir.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            id=DRIVE_FOLDER_ID,
            output=str(out_dir),
            quiet=False,
            use_cookies=False,
        )
        print(f"[OK] Download complete -> {out_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        print(
            "\n[HINT] If gdown fails due to Google Drive quota/auth:\n"
            "  1. Open the Drive link in a browser:\n"
            f"     {DRIVE_FOLDER_URL}\n"
            "  2. Select all files -> Download as ZIP\n"
            f"  3. Extract the ZIP into: {out_dir}\n",
            file=sys.stderr,
        )
        return False


def _detect_and_organize(raw_dir: Path) -> dict:
    """
    Auto-detect downloaded files and suggest/perform organization
    into ohrc/, iirs/, tmc/ subdirectories based on filename patterns.

    Returns a summary dict.
    """
    print("\n[INFO] Scanning downloaded files...")

    extensions = {".img", ".qub", ".xml", ".hdr", ".zip", ".tif", ".TIF", ".IMG", ".QUB"}
    all_files = [f for f in raw_dir.rglob("*") if f.suffix.lower() in {e.lower() for e in extensions}]

    summary = {"ohrc": [], "iirs": [], "tmc": [], "nac": [], "wac": [], "other": []}

    for f in all_files:
        name_lower = f.name.lower()
        if "ohr" in name_lower or "ohrc" in name_lower:
            summary["ohrc"].append(f)
        elif "iirs" in name_lower:
            summary["iirs"].append(f)
        elif "tmc" in name_lower:
            summary["tmc"].append(f)
        elif "nac" in name_lower or "m1" in name_lower:
            summary["nac"].append(f)
        elif "wac" in name_lower:
            summary["wac"].append(f)
        else:
            summary["other"].append(f)

    print("\n  Detected files by sensor type:")
    for sensor, files in summary.items():
        if files:
            print(f"    {sensor.upper():6s}: {len(files)} file(s)")
            for f in files[:5]:  # show first 5
                print(f"           {f.relative_to(raw_dir)}")
            if len(files) > 5:
                print(f"           ... and {len(files)-5} more")

    return summary


def _validate(base: Path) -> None:
    """Print a summary of what's in data/ and what the next steps are."""
    raw_dir = base / "data" / "raw"
    manifest = base / "data" / "pairs" / "manifest.jsonl"
    products = base / "data" / "metadata" / "products.jsonl"

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Count raw files
    img_count = len(list(raw_dir.rglob("*.img"))) + len(list(raw_dir.rglob("*.IMG")))
    qub_count = len(list(raw_dir.rglob("*.qub"))) + len(list(raw_dir.rglob("*.QUB")))
    xml_count = len(list(raw_dir.rglob("*.xml")))
    zip_count = len(list(raw_dir.rglob("*.zip")))

    print(f"\n  data/raw/  ->  {img_count} .img, {qub_count} .qub, {xml_count} .xml, {zip_count} .zip")
    print(f"  manifest.jsonl exists: {manifest.exists()}")
    print(f"  products.jsonl exists: {products.exists()}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)

    print("""
  IMPORTANT: S1 Ingest (isisimport + spiceinit) requires ISIS3/ASP.
  ISIS3 is a Linux tool — if you are on Windows, use WSL2 or a Linux VM.

  === Option A: Run full pipeline (requires ISIS3/ASP on Linux) ===

    # 1. Set up ISIS3 environment
    export ISISDATA=/path/to/isisdata
    export ALESPICEROOT=/path/to/ale

    # 2. S1 -- Ingest raw products
    python scripts/ingest.py \\
        --raw data/raw \\
        --out data/calibrated \\
        --meta data/metadata/products.jsonl \\
        --kernels $ISISDATA

    # 3. S2 -- Build pairs
    python scripts/build_pairs.py \\
        --products data/metadata/products.jsonl \\
        --config configs/ohrc_nac.yaml

    # 4. S3 -- Preprocess
    python scripts/preprocess.py \\
        --manifest data/pairs/manifest.jsonl \\
        --config configs/ohrc_nac.yaml \\
        --out data/processed

    # 5. S4-S7 -- Benchmark matchers
    python scripts/benchmark.py \\
        --manifest data/pairs/manifest.jsonl \\
        --matchers configs/matchers.yaml \\
        --out results/

    # 6. S8 -- Register
    python scripts/register.py \\
        --pair <pair_id> \\
        --matcher sift \\
        --geometry results/<pair_id>/sift/geometry.json \\
        --matches results/<pair_id>/sift/matches_refined.json

    # 7. S9 -- Evaluate
    python -m src.evaluation.aggregate --results results/ --out results/leaderboard.csv
    python -m src.evaluation.leakage_audit --manifest data/pairs/manifest.jsonl

  === Option B: Skip ISIS3, run from preprocessed data ===

    If images are already GeoTIFFs (from PRADAN portal):
    - Place src.tif / ref.tif directly in data/processed/<pair_id>/
    - Skip S1-S2, go straight to S3:

    python scripts/preprocess.py \\
        --manifest data/pairs/manifest.jsonl \\
        --config configs/ohrc_nac.yaml \\
        --out data/processed

""")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SIH 2026 dataset from Google Drive and set up data directory.",
    )
    parser.add_argument(
        "--out", default="data/raw",
        help="Destination directory for downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without actually downloading",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, just validate existing data/ structure",
    )
    args = parser.parse_args()

    base = PROJECT_ROOT
    out_dir = base / args.out

    print("=" * 60)
    print("SIH 2026 — Dataset Setup & Download")
    print("=" * 60)
    print(f"  Project root : {base}")
    print(f"  Download dest: {out_dir}")
    print(f"  Drive folder : {DRIVE_FOLDER_URL}")

    # Step 1: Create directory structure
    _create_dirs(base)

    # Step 2: Install gdown & download
    if not args.skip_download:
        if not _ensure_gdown():
            return 1

        ok = _download_folder(out_dir, dry_run=args.dry_run)
        if not ok:
            return 1

        # Step 3: Detect and organize
        if not args.dry_run:
            _detect_and_organize(out_dir)

    # Step 4: Validate and print next steps
    _validate(base)

    return 0


if __name__ == "__main__":
    sys.exit(main())

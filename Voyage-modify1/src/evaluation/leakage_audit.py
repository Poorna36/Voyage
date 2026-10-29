"""
src/evaluation/leakage_audit.py
--------------------------------
F22 — Leakage Audit (ARCHITECTURE.md L7, VALIDATION.md §6)

Verifies train/test split integrity before any leaderboard number is published.
Exits non-zero (code 4) and writes NO leaderboard output if audit fails.

Checks (per VALIDATION.md §6):
  1. No pair appears in both train AND test split
  2. No pair's geo_cell appears in both splits
  3. Any gt_path present must correspond to a test-split pair
  4. leaderboard.csv split column matches manifest.jsonl (if CSV provided)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when a leakage violation is detected."""


def load_manifest(manifest_path: Path) -> List[dict]:
    """Load manifest.jsonl (one JSON object per line)."""
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error("manifest.jsonl line %d: JSON error — %s", lineno, exc)
                raise
    return records


def audit_manifest(records: List[dict]) -> Tuple[bool, List[str]]:
    """
    Run all leakage checks against manifest records.

    Returns
    -------
    (passed: bool, violations: List[str])
    """
    violations = []

    train_pairs: set = set()
    test_pairs:  set = set()
    train_cells: set = set()
    test_cells:  set = set()
    test_pair_ids: set = set()

    for rec in records:
        pair_id = rec.get("pair_id", "")
        split   = rec.get("split", "")
        cell    = rec.get("geo_cell", "")
        gt_path = rec.get("gt_path")

        if split == "train":
            train_pairs.add(pair_id)
            if cell:
                train_cells.add(cell)
        elif split == "test":
            test_pairs.add(pair_id)
            test_pair_ids.add(pair_id)
            if cell:
                test_cells.add(cell)
        else:
            violations.append(
                f"pair_id={pair_id!r} has unknown split={split!r} "
                "(must be 'train' or 'test')"
            )

    # Check 1: No pair in both splits
    both_pairs = train_pairs & test_pairs
    if both_pairs:
        violations.append(
            f"CRITICAL: {len(both_pairs)} pair(s) appear in BOTH train and test: "
            f"{sorted(both_pairs)[:5]}{'...' if len(both_pairs) > 5 else ''}"
        )

    # Check 2: No geo_cell in both splits
    both_cells = train_cells & test_cells
    if both_cells:
        violations.append(
            f"CRITICAL: {len(both_cells)} geo_cell(s) appear in BOTH train and test: "
            f"{sorted(both_cells)[:5]}{'...' if len(both_cells) > 5 else ''}"
        )

    # Check 3: gt_path only for test-split pairs
    for rec in records:
        pair_id = rec.get("pair_id", "")
        gt_path = rec.get("gt_path")
        split   = rec.get("split", "")
        if gt_path and split != "test":
            violations.append(
                f"pair_id={pair_id!r} has gt_path but split={split!r} "
                "(GT must only exist for test pairs)"
            )

    passed = len(violations) == 0
    return passed, violations


def audit_leaderboard_csv(
    csv_path: Path,
    manifest_records: List[dict],
) -> Tuple[bool, List[str]]:
    """
    Check that leaderboard.csv split column matches manifest.jsonl.

    Returns (passed, violations).
    """
    violations = []

    # Build pair_id → split map from manifest
    manifest_splits: Dict[str, str] = {
        r["pair_id"]: r.get("split", "") for r in manifest_records if "pair_id" in r
    }

    if not csv_path.exists():
        logger.info("leaderboard.csv not found — skipping CSV audit")
        return True, []

    import csv as csv_mod
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        if "split" not in (reader.fieldnames or []):
            violations.append("leaderboard.csv missing 'split' column")
            return False, violations

        for row in reader:
            pair_id = row.get("pair_id", "")
            csv_split = row.get("split", "")
            if pair_id and pair_id in manifest_splits:
                manifest_split = manifest_splits[pair_id]
                if csv_split != manifest_split:
                    violations.append(
                        f"pair_id={pair_id!r}: leaderboard split={csv_split!r} "
                        f"but manifest split={manifest_split!r}"
                    )

    return len(violations) == 0, violations


def audit_msm_dataset(
    manifest_records: List[dict],
    msm_stats_path: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """
    Verify MSM training data strictly excludes test geo_cells (F15/F27).
    """
    violations = []
    train_cells = {r.get("geo_cell") for r in manifest_records if r.get("split") == "train" and r.get("geo_cell")}
    test_cells = {r.get("geo_cell") for r in manifest_records if r.get("split") == "test" and r.get("geo_cell")}

    overlap = train_cells & test_cells
    if overlap:
        violations.append(
            f"MSM LEAKAGE: {len(overlap)} geo_cell(s) in both MSM train pool and test split: {sorted(overlap)}"
        )

    if msm_stats_path and msm_stats_path.exists():
        try:
            with open(msm_stats_path, "r", encoding="utf-8") as fh:
                st = json.load(fh)
            audit_msg = st.get("geo_cell_leakage_audit", "")
            if "PASSED" not in audit_msg:
                violations.append(f"MSM stats audit flag indicates failure: {audit_msg}")
        except Exception as exc:
            violations.append(f"Failed to parse MSM stats at {msm_stats_path}: {exc}")

    return len(violations) == 0, violations


def run_audit(
    manifest_path: str | Path,
    leaderboard_csv: Optional[str | Path] = None,
    check_msm: bool = False,
    msm_stats_path: Optional[str | Path] = None,
) -> bool:
    """
    Run the full leakage audit.

    Returns True if all checks pass, False otherwise.
    Logs all violations.

    If audit fails, exits with code 4 when called as main.
    """
    manifest_path = Path(manifest_path)
    logger.info("Loading manifest: %s", manifest_path)
    records = load_manifest(manifest_path)
    logger.info("Loaded %d pair records", len(records))

    passed, violations = audit_manifest(records)

    if leaderboard_csv:
        csv_passed, csv_violations = audit_leaderboard_csv(
            Path(leaderboard_csv), records
        )
        passed = passed and csv_passed
        violations.extend(csv_violations)

    if check_msm:
        stats_p = Path(msm_stats_path) if msm_stats_path else Path("models/msm_v1_stats.json")
        msm_passed, msm_violations = audit_msm_dataset(records, msm_stats_path=stats_p)
        passed = passed and msm_passed
        violations.extend(msm_violations)

    if violations:
        logger.error("=== LEAKAGE AUDIT FAILED: %d violation(s) ===", len(violations))
        for v in violations:
            logger.error("  • %s", v)
    else:
        logger.info("=== LEAKAGE AUDIT PASSED ===")
        # Print split statistics
        train_n = sum(1 for r in records if r.get("split") == "train")
        test_n  = sum(1 for r in records if r.get("split") == "test")
        train_cells = {r.get("geo_cell") for r in records if r.get("split") == "train"}
        test_cells  = {r.get("geo_cell") for r in records if r.get("split") == "test"}
        logger.info(
            "  train: %d pairs, %d geo_cells | test: %d pairs, %d geo_cells",
            train_n, len(train_cells), test_n, len(test_cells),
        )

    return passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Leakage audit: verify train/test split integrity"
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to data/pairs/manifest.jsonl")
    parser.add_argument("--leaderboard", default=None,
                        help="Path to results/leaderboard.csv (optional cross-check)")
    parser.add_argument("--check-msm", action="store_true",
                        help="Audit MSM training dataset and stats for geo-cell disjointness")
    parser.add_argument("--msm-stats", default="models/msm_v1_stats.json",
                        help="Path to models/msm_v1_stats.json")
    args = parser.parse_args()

    ok = run_audit(
        args.manifest,
        leaderboard_csv=args.leaderboard,
        check_msm=args.check_msm,
        msm_stats_path=args.msm_stats,
    )
    sys.exit(0 if ok else 4)   # Exit code 4 = leakage audit failed (per PIPELINE.md §8)


"""
src/evaluation/aggregate.py
----------------------------
F22 — Leaderboard Aggregation (ARCHITECTURE.md L7)

Reads all pair_results/*.json, aggregates by (matcher × sensor_pair × stratum),
and writes results/leaderboard.csv atomically.

Rules (per VALIDATION.md §4 and INTERFACES.md §5):
  - Polar and high-latitude strata NEVER aggregated away — always separate rows
  - SELENE pairs form a separate stratum; never merged with NAC/WAC rows
  - split column always present; no test-split leakage from train
  - Write atomically: write to .tmp, then rename (prevents partial CSV on crash)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Metric columns in leaderboard.csv (per INTERFACES.md §5)
METRIC_COLS = [
    "rmse_px_mean", "rmse_px_median",
    "pct_lt_1px_mean", "pct_lt_0p5px_mean",
    "medae_px_mean",
    "inlier_count_mean", "inlier_ratio_mean",
    "spatial_coverage_mean", "grid_density_std_mean",
    "refinement_gain_mean", "runtime_s_mean",
    "n_failures",
]

STRATUM_COLS = [
    "matcher", "sensor_pair", "split",
    "terrain_class", "latitude_bin", "delta_az_bin",
    "crater_density_bin", "ref_type",
    "n_pairs",
]


def _latitude_bin(lat: Optional[float]) -> str:
    if lat is None:
        return "unknown"
    if abs(lat) > 55:
        return "polar"
    if abs(lat) > 30:
        return "mid_latitude"
    return "equatorial"


def _delta_az_bin(daz: Optional[float]) -> str:
    if daz is None:
        return "unknown"
    if daz < 30:
        return "lt30"
    if daz < 90:
        return "30_90"
    return "gt90"


def _crater_density_bin(density: Optional[float]) -> str:
    if density is None:
        return "unknown"
    if density < 1.0:
        return "low"
    if density < 5.0:
        return "medium"
    return "high"


def _stratum_key(record: dict) -> tuple:
    """Build the grouping key for a pair result."""
    stratum = record.get("stratum", {})
    return (
        record.get("matcher", "unknown"),
        stratum.get("sensor_pair", "unknown"),
        record.get("split", "unknown"),
        stratum.get("terrain_class", "unknown"),
        stratum.get("latitude_bin", "unknown"),
        stratum.get("delta_az_bin", "unknown"),
        stratum.get("crater_density_bin", "unknown"),
        stratum.get("ref_type", "unknown"),
    )


def _mean_safe(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else None


def _median_safe(vals: list) -> Optional[float]:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.median(v)) if v else None


def load_pair_results(results_dir: Path) -> List[dict]:
    """Load all pair result JSON files from results/pair_results/."""
    pair_results_dir = results_dir / "pair_results"
    if not pair_results_dir.exists():
        logger.warning("pair_results dir not found: %s", pair_results_dir)
        return []

    records = []
    for json_file in sorted(pair_results_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            records.append(rec)
        except Exception as exc:
            logger.error("Failed to load %s: %s", json_file, exc)
    logger.info("Loaded %d pair result records", len(records))
    return records


def aggregate(
    records: List[dict],
    split_filter: Optional[str] = None,
) -> List[dict]:
    """
    Group records by stratum and compute aggregate metrics.

    Parameters
    ----------
    records : list of EvaluationRecord dicts
    split_filter : if set, only aggregate this split ("train" | "test")

    Returns
    -------
    list of leaderboard row dicts
    """
    if split_filter:
        records = [r for r in records if r.get("split") == split_filter]

    # Group by stratum key
    groups: Dict[tuple, List[dict]] = {}
    for rec in records:
        key = _stratum_key(rec)
        groups.setdefault(key, []).append(rec)

    rows = []
    for key, group in groups.items():
        matcher, sensor_pair, split, terrain, lat_bin, daz_bin, den_bin, ref_type = key

        metrics_list = [r.get("metrics", {}) for r in group]
        n_pairs = len(group)
        n_failures = sum(
            1 for r in group
            if r.get("metrics", {}).get("rmse_px") is None
        )

        rmse_vals   = [m.get("rmse_px") for m in metrics_list]
        p1_vals     = [m.get("pct_lt_1px") for m in metrics_list]
        p05_vals    = [m.get("pct_lt_0p5px") for m in metrics_list]
        med_vals    = [m.get("medae_px") for m in metrics_list]
        ic_vals     = [m.get("inlier_count") for m in metrics_list]
        ir_vals     = [m.get("inlier_ratio") for m in metrics_list]
        cov_vals    = [m.get("spatial_coverage") for m in metrics_list]
        gds_vals    = [m.get("grid_density_std") for m in metrics_list]
        gain_vals   = [m.get("refinement_gain_px") for m in metrics_list]
        rt_vals     = [m.get("runtime_s") for m in metrics_list]

        row = {
            "matcher": matcher,
            "sensor_pair": sensor_pair,
            "split": split,
            "terrain_class": terrain,
            "latitude_bin": lat_bin,
            "delta_az_bin": daz_bin,
            "crater_density_bin": den_bin,
            "ref_type": ref_type,
            "n_pairs": n_pairs,
            "rmse_px_mean": _mean_safe(rmse_vals),
            "rmse_px_median": _median_safe(rmse_vals),
            "pct_lt_1px_mean": _mean_safe(p1_vals),
            "pct_lt_0p5px_mean": _mean_safe(p05_vals),
            "medae_px_mean": _mean_safe(med_vals),
            "inlier_count_mean": _mean_safe(ic_vals),
            "inlier_ratio_mean": _mean_safe(ir_vals),
            "spatial_coverage_mean": _mean_safe(cov_vals),
            "grid_density_std_mean": _mean_safe(gds_vals),
            "refinement_gain_mean": _mean_safe(gain_vals),
            "runtime_s_mean": _mean_safe(rt_vals),
            "n_failures": n_failures,
        }
        rows.append(row)

    # Sort: test split first, then by matcher, then sensor_pair
    rows.sort(key=lambda r: (r["split"] != "test", r["matcher"], r["sensor_pair"]))
    return rows


def write_leaderboard_csv(rows: List[dict], output_path: Path) -> None:
    """
    Write leaderboard rows to CSV atomically (write temp → rename).
    Prevents partial file on crash.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_cols = STRATUM_COLS + [c for c in METRIC_COLS if c not in STRATUM_COLS]

    tmp_path = None
    try:
        # Write to temp file in same directory (same filesystem for atomic rename)
        fd, tmp_str = tempfile.mkstemp(
            dir=output_path.parent, prefix=".leaderboard_tmp_", suffix=".csv"
        )
        tmp_path = Path(tmp_str)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(",".join(all_cols) + "\n")
            for row in rows:
                line = []
                for col in all_cols:
                    val = row.get(col)
                    if val is None:
                        line.append("")
                    elif isinstance(val, float):
                        line.append(f"{val:.6f}")
                    else:
                        line.append(str(val))
                f.write(",".join(line) + "\n")

        # Atomic rename
        tmp_path.replace(output_path)
        logger.info("Leaderboard written: %s (%d rows)", output_path, len(rows))

    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def run_aggregation(
    results_dir: str | Path,
    output_path: str | Path,
    split_filter: Optional[str] = None,
) -> List[dict]:
    """
    End-to-end: load pair results → aggregate → write CSV.

    Parameters
    ----------
    results_dir   : path to results/ directory
    output_path   : path for leaderboard.csv output
    split_filter  : "test" | "train" | None (all)

    Returns
    -------
    list of leaderboard row dicts
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    records = load_pair_results(results_dir)
    if not records:
        logger.warning("No pair results found — leaderboard will be empty")

    rows = aggregate(records, split_filter=split_filter)
    write_leaderboard_csv(rows, output_path)

    # Validate: polar and SELENE strata are represented (warn if missing)
    lat_bins = {r["latitude_bin"] for r in rows}
    if "polar" not in lat_bins and records:
        logger.warning(
            "VALIDATION: polar stratum is MISSING from leaderboard — "
            "check that polar pairs are in the results."
        )

    ref_types = {r["ref_type"] for r in rows}
    if "SELENE" not in ref_types and records:
        logger.info("No SELENE pairs in results (expected if none were acquired)")

    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Aggregate pair results into leaderboard.csv")
    parser.add_argument("--results", required=True, help="Path to results/ directory")
    parser.add_argument("--out", required=True, help="Output leaderboard.csv path")
    parser.add_argument("--split", default=None, choices=["train", "test"],
                        help="Filter to one split (default: all)")
    args = parser.parse_args()

    rows = run_aggregation(args.results, args.out, split_filter=args.split)
    print(f"Leaderboard written: {args.out} ({len(rows)} rows)")

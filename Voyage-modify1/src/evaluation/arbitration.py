"""
src/evaluation/arbitration.py
------------------------------
F23 — Arbitration Log (ARCHITECTURE.md §4, PIPELINE.md S4)

Determines the winning matcher per pair based on the arbitration policy,
writes one entry per pair to results/arbitration.log, and records
total-failure cases in results/failures.jsonl.

Arbitration policy (production mode):
  1. M3 (crater) if crater_density >= tau_c AND terrain in {highland, polar_highland, polar}
                 AND detector_validated=True
  2. M2 (lightglue) if learned_confidence_ok   [NOT gated on GPU availability]
  3. M1 (rift2 or lnift)                       [flag no_validated_primary_matcher if polar]
  4. M0 (sift) fallback if primary inlier_ratio < inlier_ratio_floor
  Total failure: M0 also fails → record pair_outcome=TOTAL_FAILURE

Tie-break rule:
  If abs(RMSE_A - RMSE_B) < gt_interannotator_rmse_px
  AND abs(inlier_ratio_A - inlier_ratio_B) < 0.05
  → apply preference_order; record tie_break=True

Coordinate convention: (col, row). NEVER (row, col).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arbitration constants (per ARCHITECTURE.md §4 and CONFIGURATION.md)
# ---------------------------------------------------------------------------
PREFERENCE_ORDER = ["crater", "lightglue", "lnift", "rift2", "sift"]
INLIER_RATIO_FLOOR = 0.05        # below this → fall back to M0
TIE_INLIER_RATIO_DELTA = 0.05   # tie-break: |ratio_A - ratio_B| < this
DEFAULT_GT_INTERANN_RMSE = 0.3  # px — used if not provided per-pair


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ArbitrationEntry:
    """One entry per pair in arbitration.log."""

    def __init__(
        self,
        pair_id: str,
        winner: Optional[str],
        winner_inlier_ratio: float,
        winner_rmse: Optional[float],
        fallback_occurred: bool,
        fallback_from: Optional[str],
        fallback_reason: Optional[str],
        tie_break: bool,
        tie_break_candidates: List[str],
        no_validated_primary_matcher: bool,
        pair_outcome: str,          # "success" | "fallback" | "total_failure"
        terrain_class: str,
        latitude_center_deg: float,
        crater_density_per_km2: Optional[float],
        detector_validated: bool,
        created_at: str,
        msm_routing: Optional[Dict[str, Any]] = None,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}



# ---------------------------------------------------------------------------
# Arbitration logic
# ---------------------------------------------------------------------------

def _preference_index(matcher: str) -> int:
    try:
        return PREFERENCE_ORDER.index(matcher)
    except ValueError:
        return len(PREFERENCE_ORDER)  # unknown matchers go last


def arbitrate_pair(
    pair_id: str,
    matcher_results: Dict[str, dict],
    pair_record: dict,
    gt_interannotator_rmse_px: float = DEFAULT_GT_INTERANN_RMSE,
    msm_routing: Optional[Dict[str, Any]] = None,
) -> ArbitrationEntry:

    """
    Determine the winning matcher for one pair.

    Parameters
    ----------
    pair_id : str
    matcher_results : dict of {matcher_id: geometry_record_dict}
        Each value is a geometry.json dict with keys:
          inlier_ratio, inlier_count, rmse_px, model_type,
          detector_validated (M3 only)
    pair_record : PairRecord dict from manifest.jsonl
    gt_interannotator_rmse_px : inter-annotator error bound (for tie-break)

    Returns
    -------
    ArbitrationEntry
    """
    terrain = pair_record.get("terrain_class", "")
    lat = float(pair_record.get("latitude_center_deg", 0.0))
    crater_density = pair_record.get("crater_density_per_km2")
    is_polar = abs(lat) > 55 or terrain in {"polar", "polar_highland"}

    # --- Helper: is this matcher result a "success"? ---
    def is_valid(m_id: str) -> bool:
        r = matcher_results.get(m_id, {})
        return (
            r.get("inlier_ratio", 0.0) >= INLIER_RATIO_FLOOR and
            r.get("inlier_count", 0) >= 20 and
            r.get("model_type", "none") != "none"
        )

    # --- Evaluate M3 gate ---
    tau_c = pair_record.get("tau_c", 2.0)  # from config; default 2.0 /km²
    m3_res = matcher_results.get("crater", {})
    m3_gate = (
        crater_density is not None and
        crater_density >= tau_c and
        terrain in {"highland", "polar_highland", "polar"} and
        m3_res.get("detector_validated", False)
    )

    # --- Evaluate M2 confidence ---
    m2_res = matcher_results.get("lightglue", {})
    m2_ok = is_valid("lightglue")  # NOT gated on GPU

    primary: Optional[str] = None
    no_validated = False

    if m3_gate and is_valid("crater"):
        primary = "crater"
    elif m2_ok:
        primary = "lightglue"
    else:
        # Try M1 variants
        for m1 in ["lnift", "rift2"]:
            if is_valid(m1):
                primary = m1
                break
        if primary and is_polar:
            no_validated = True
            logger.warning(
                "pair %s: M1 (%s) used at polar lat=%.1f — no_validated_primary_matcher=True",
                pair_id, primary, lat,
            )

    # --- Check primary inlier_ratio floor → M0 fallback ---
    fallback_occurred = False
    fallback_from = None
    fallback_reason = None

    if primary is not None:
        primary_ratio = matcher_results.get(primary, {}).get("inlier_ratio", 0.0)
        if primary_ratio < INLIER_RATIO_FLOOR:
            fallback_from = primary
            fallback_reason = f"inlier_ratio={primary_ratio:.3f} < floor={INLIER_RATIO_FLOOR}"
            primary = "sift" if is_valid("sift") else None
            fallback_occurred = True
    else:
        # No primary found at all — try M0
        if is_valid("sift"):
            primary = "sift"
            fallback_occurred = True
            fallback_reason = "no valid primary matcher found"
        else:
            primary = None

    # --- Total failure path ---
    if primary is None or not is_valid(primary or ""):
        logger.error("pair %s: TOTAL_FAILURE — no matcher produced valid output", pair_id)
        return ArbitrationEntry(
            pair_id=pair_id,
            winner=None,
            winner_inlier_ratio=0.0,
            winner_rmse=None,
            fallback_occurred=True,
            fallback_from=fallback_from or primary,
            fallback_reason="all matchers failed",
            tie_break=False,
            tie_break_candidates=[],
            no_validated_primary_matcher=no_validated,
            pair_outcome="total_failure",
            terrain_class=terrain,
            latitude_center_deg=lat,
            crater_density_per_km2=crater_density,
            detector_validated=m3_res.get("detector_validated", False),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    primary_result = matcher_results[primary]

    # --- Tie-break check ---
    tie_break = False
    tie_candidates = []
    primary_rmse = primary_result.get("rmse_px", None)
    primary_ratio = primary_result.get("inlier_ratio", 0.0)

    if primary_rmse is not None:
        for alt_id, alt_res in matcher_results.items():
            if alt_id == primary or not is_valid(alt_id):
                continue
            alt_rmse = alt_res.get("rmse_px")
            alt_ratio = alt_res.get("inlier_ratio", 0.0)
            if alt_rmse is None:
                continue
            rmse_diff = abs(primary_rmse - alt_rmse)
            ratio_diff = abs(primary_ratio - alt_ratio)
            if rmse_diff < gt_interannotator_rmse_px and ratio_diff < TIE_INLIER_RATIO_DELTA:
                tie_candidates.append(alt_id)

    if tie_candidates:
        tie_break = True
        # Apply preference order to break tie
        all_candidates = [primary] + tie_candidates
        all_candidates.sort(key=_preference_index)
        winner = all_candidates[0]
        if winner != primary:
            logger.info(
                "pair %s: tie-break resolved %s → %s (RMSE diff < %.3f px)",
                pair_id, primary, winner, gt_interannotator_rmse_px,
            )
            primary = winner
            primary_result = matcher_results[primary]
            primary_rmse = primary_result.get("rmse_px")
            primary_ratio = primary_result.get("inlier_ratio", 0.0)

    outcome = "fallback" if fallback_occurred else "success"

    logger.info(
        "pair %s: winner=%s, inlier_ratio=%.3f, RMSE=%s, fallback=%s, tie=%s",
        pair_id, primary, primary_ratio,
        f"{primary_rmse:.3f}" if primary_rmse else "N/A",
        fallback_occurred, tie_break,
    )

    return ArbitrationEntry(
        pair_id=pair_id,
        winner=primary,
        winner_inlier_ratio=primary_ratio,
        winner_rmse=primary_rmse,
        fallback_occurred=fallback_occurred,
        fallback_from=fallback_from,
        fallback_reason=fallback_reason,
        tie_break=tie_break,
        tie_break_candidates=tie_candidates,
        no_validated_primary_matcher=no_validated,
        pair_outcome=outcome,
        terrain_class=terrain,
        latitude_center_deg=lat,
        crater_density_per_km2=crater_density,
        detector_validated=m3_res.get("detector_validated", False),
        created_at=datetime.now(timezone.utc).isoformat(),
        msm_routing=msm_routing,
    )



# ---------------------------------------------------------------------------
# Write arbitration log + failures.jsonl
# ---------------------------------------------------------------------------

def write_arbitration_log(
    entries: List[ArbitrationEntry],
    log_path: Path,
    failures_path: Path,
) -> None:
    """
    Append arbitration entries to arbitration.log (one JSON per line).
    Write total_failure entries to failures.jsonl as well.

    Both files are append-only (per ARCHITECTURE.md orchestration rules).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_f, \
         open(failures_path, "a", encoding="utf-8") as fail_f:
        for entry in entries:
            line = json.dumps(entry.to_dict(), ensure_ascii=False)
            log_f.write(line + "\n")

            if entry.pair_outcome == "total_failure":
                # Write empty registered.tif placeholder mention to failures
                failure_rec = {
                    "pair_id": entry.pair_id,
                    "stage": "S4-arbitration",
                    "pair_outcome": "TOTAL_FAILURE",
                    "reason": entry.fallback_reason,
                    "registered_tif": f"results/{entry.pair_id}/registered_EMPTY.tif",
                    "included_in_failure_rate": True,  # NEVER silently omitted
                    "created_at": entry.created_at,
                }
                fail_f.write(json.dumps(failure_rec, ensure_ascii=False) + "\n")

    logger.info(
        "Arbitration log: %d entries written to %s", len(entries), log_path
    )
    n_failures = sum(1 for e in entries if e.pair_outcome == "total_failure")
    if n_failures:
        logger.warning("%d total failures written to %s", n_failures, failures_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run arbitration on pair results")
    parser.add_argument("--results", required=True, help="results/ directory")
    parser.add_argument("--manifest", required=True, help="data/pairs/manifest.jsonl")
    parser.add_argument("--log", default="results/arbitration.log")
    parser.add_argument("--failures", default="results/failures.jsonl")
    args = parser.parse_args()

    results_dir = Path(args.results)
    manifest_path = Path(args.manifest)

    # Load manifest
    with open(manifest_path, "r") as f:
        pair_records = [json.loads(line) for line in f if line.strip()]
    pair_map = {r["pair_id"]: r for r in pair_records}

    # Load geometry results per pair × matcher
    entries = []
    for pair_dir in sorted(results_dir.iterdir()):
        if not pair_dir.is_dir() or pair_dir.name == "pair_results":
            continue
        pair_id = pair_dir.name
        pair_record = pair_map.get(pair_id, {"pair_id": pair_id})
        matcher_results = {}
        for matcher_dir in pair_dir.iterdir():
            if not matcher_dir.is_dir():
                continue
            geo_file = matcher_dir / "geometry.json"
            if geo_file.exists():
                with open(geo_file) as f:
                    matcher_results[matcher_dir.name] = json.load(f)

        if matcher_results:
            entry = arbitrate_pair(pair_id, matcher_results, pair_record)
            entries.append(entry)

    write_arbitration_log(entries, Path(args.log), Path(args.failures))
    print(f"Arbitration complete: {len(entries)} pairs processed")

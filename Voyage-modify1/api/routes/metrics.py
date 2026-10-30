"""
backend/api/routes/metrics.py
------------------------------
Serves registration metrics and results data to the frontend dashboard.
Reads from the results/ directory for completed pipeline runs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/metrics", tags=["metrics"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = PROJECT_ROOT / "data" / "pairs" / "manifest.jsonl"
GT_DIR = PROJECT_ROOT / "data" / "metadata" / "gt"


class MatchMetrics(BaseModel):
    pair_id: str
    matcher: Optional[str] = None
    rmse_px: Optional[float] = None
    ssim: Optional[float] = None
    inlier_ratio: Optional[float] = None
    inlier_count: Optional[int] = None
    candidate_count: Optional[int] = None
    spatial_coverage: Optional[float] = None
    runtime_s: Optional[float] = None
    terrain_class: Optional[str] = None
    src_sensor: Optional[str] = None
    src_gsd_m: Optional[float] = None
    ref_type: Optional[str] = None
    ref_gsd_m: Optional[float] = None
    solar_incidence_deg: Optional[float] = None
    solar_azimuth_deg: Optional[float] = None
    latitude_center_deg: Optional[float] = None
    longitude_center_deg: Optional[float] = None
    has_ground_truth: bool = False


def _load_manifest() -> List[Dict[str, Any]]:
    """Load all pairs from the JSONL manifest."""
    if not MANIFEST_PATH.exists():
        return []
    pairs = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pairs


def _load_result(pair_id: str) -> Optional[Dict[str, Any]]:
    """Try to load a result JSON for a given pair_id from the results directory."""
    # Check multiple possible result file patterns
    for pattern in [f"{pair_id}.json", f"{pair_id}_result.json", f"result_{pair_id}.json"]:
        path = RESULTS_DIR / pattern
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Check for result dirs containing a result.json
    pair_dir = RESULTS_DIR / pair_id
    if pair_dir.is_dir():
        result_file = pair_dir / "result.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)

    return None


def _has_gt(pair: Dict[str, Any]) -> bool:
    """Check if a pair has ground-truth data."""
    gt_path = pair.get("gt_path")
    if gt_path:
        full_path = PROJECT_ROOT / gt_path
        return full_path.exists()
    return False


@router.get("/", response_model=List[MatchMetrics])
async def list_all_metrics():
    """Return metrics for all pairs (from results or manifest metadata)."""
    manifest = _load_manifest()
    results = []

    for pair in manifest:
        pair_id = pair["pair_id"]
        src = pair.get("src", {})
        ref = pair.get("ref", {})

        # Try to load computed results
        result = _load_result(pair_id)

        metrics = MatchMetrics(
            pair_id=pair_id,
            matcher=result.get("matcher") if result else None,
            rmse_px=result.get("rmse_px") if result else None,
            ssim=result.get("ssim") if result else None,
            inlier_ratio=result.get("inlier_ratio") if result else None,
            inlier_count=result.get("inlier_count") if result else None,
            candidate_count=result.get("candidate_count") if result else None,
            spatial_coverage=result.get("spatial_coverage") if result else None,
            runtime_s=result.get("runtime_s") if result else None,
            terrain_class=pair.get("terrain_class"),
            src_sensor=src.get("sensor"),
            src_gsd_m=src.get("gsd_m"),
            ref_type=ref.get("type"),
            ref_gsd_m=ref.get("gsd_m"),
            solar_incidence_deg=src.get("solar_incidence_deg"),
            solar_azimuth_deg=src.get("solar_azimuth_deg"),
            latitude_center_deg=pair.get("latitude_center_deg"),
            longitude_center_deg=pair.get("longitude_center_deg"),
            has_ground_truth=_has_gt(pair),
        )
        results.append(metrics)

    return results


@router.get("/{pair_id}", response_model=MatchMetrics)
async def get_pair_metrics(pair_id: str):
    """Get metrics for a specific pair."""
    manifest = _load_manifest()
    target = None
    for pair in manifest:
        if pair["pair_id"] == pair_id:
            target = pair
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_id}' not found")

    src = target.get("src", {})
    ref = target.get("ref", {})
    result = _load_result(pair_id)

    return MatchMetrics(
        pair_id=pair_id,
        matcher=result.get("matcher") if result else None,
        rmse_px=result.get("rmse_px") if result else None,
        ssim=result.get("ssim") if result else None,
        inlier_ratio=result.get("inlier_ratio") if result else None,
        inlier_count=result.get("inlier_count") if result else None,
        candidate_count=result.get("candidate_count") if result else None,
        spatial_coverage=result.get("spatial_coverage") if result else None,
        runtime_s=result.get("runtime_s") if result else None,
        terrain_class=target.get("terrain_class"),
        src_sensor=src.get("sensor"),
        src_gsd_m=src.get("gsd_m"),
        ref_type=ref.get("type"),
        ref_gsd_m=ref.get("gsd_m"),
        solar_incidence_deg=src.get("solar_incidence_deg"),
        solar_azimuth_deg=src.get("solar_azimuth_deg"),
        latitude_center_deg=target.get("latitude_center_deg"),
        longitude_center_deg=target.get("longitude_center_deg"),
        has_ground_truth=_has_gt(target),
    )


@router.get("/ground-truth/{pair_id}")
async def get_ground_truth(pair_id: str):
    """Retrieve ground-truth checkpoint data for a pair if available."""
    manifest = _load_manifest()
    target = None
    for pair in manifest:
        if pair["pair_id"] == pair_id:
            target = pair
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_id}' not found")

    gt_path = target.get("gt_path")
    if not gt_path:
        raise HTTPException(status_code=404, detail=f"No ground truth for pair '{pair_id}'")

    full_path = PROJECT_ROOT / gt_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"GT file not found at {gt_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)

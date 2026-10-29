#!/usr/bin/env python3
"""
scripts/register.py
--------------------
S8 — Product Generation (ARCHITECTURE.md L6, FEATURES.md F20 + F21)

Applies the fitted geometric model to warp the source image onto the
reference coordinate grid and produces all output artifacts:

  registered.tif       — GeoTIFF on reference grid (F20)
  match_points.csv     — pixel coords both images + lon/lat + residual (F20)
  match_points.gcp     — GDAL-loadable GCP file (F20)
  qc_checkerboard.png  — 64px interleaved tiles for visual alignment check (F21)
  qc_matches.png       — match overlay with colour-coded residuals (F21)
  qc_residuals.png     — Gaussian heat map of registration error (F21)

Gate: warp valid >= 90% of footprint. On failure: report partial extent.

Exit codes (per PIPELINE.md §8):
  0  all pairs completed, all gates passed
  1  gate failure(s) logged; non-failed pairs completed
  2  configuration error
  3  environment error
  4  leakage audit failed (not applicable here; kept for consistency)

Coordinate convention: (col, row) = (x, y), 0-indexed. NEVER (row, col).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import rasterio
    from rasterio.transform import from_bounds, Affine
    from rasterio.crs import CRS
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (per CONFIGURATION.md §8)
# ---------------------------------------------------------------------------
VALID_WARP_FRACTION_MIN = 0.90
QC_CHECKERBOARD_TILE_PX = 64
RESIDUAL_HEATMAP_SIGMA = 3   # px Gaussian spread per match
MATCH_OVERLAY_GREEN_THRESH  = 0.5   # px  → green
MATCH_OVERLAY_YELLOW_THRESH = 1.0   # px  → yellow (else red)


# ---------------------------------------------------------------------------
# Load geometry and match records
# ---------------------------------------------------------------------------

def load_geometry(geometry_path: Path) -> dict:
    with open(geometry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_matches(matches_path: Path) -> dict:
    with open(matches_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------

def _build_transform_matrix(geometry: dict) -> np.ndarray:
    """Extract the 3×3 warp matrix from a geometry record."""
    if geometry.get("tilewise"):
        # For tile-wise, use identity — caller handles tile blend
        return np.eye(3, dtype=np.float64)
    M = np.array(geometry["model_matrix"], dtype=np.float64)
    if M.shape == (2, 3):
        H = np.eye(3)
        H[:2, :] = M
        return H
    return M


def warp_image(
    src_image: np.ndarray,
    geometry: dict,
    ref_shape: Tuple[int, int],
) -> Tuple[np.ndarray, float]:
    """
    Warp source image onto the reference grid using the fitted model.

    For tile-wise models, applies Gaussian-blended displacement field.
    For global models (similarity/affine/homography), uses cv2.warpPerspective.

    Returns (warped_image, valid_fraction).
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for image warping.")

    ref_h, ref_w = ref_shape

    if geometry.get("tilewise") and geometry.get("tile_models"):
        from src.registration.tilewise import blend_displacement
        tile_models = geometry["tile_models"]

        # Build dense displacement map
        grid_col, grid_row = np.meshgrid(
            np.arange(ref_w, dtype=np.float32),
            np.arange(ref_h, dtype=np.float32),
        )
        flat_col = grid_col.ravel()
        flat_row = grid_row.ravel()

        dcol, drow = blend_displacement(flat_col, flat_row, tile_models)
        map_x = (flat_col + dcol).reshape(ref_h, ref_w).astype(np.float32)
        map_y = (flat_row + drow).reshape(ref_h, ref_w).astype(np.float32)

        warped = cv2.remap(src_image, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        H = _build_transform_matrix(geometry)
        warped = cv2.warpPerspective(
            src_image, H, (ref_w, ref_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # Valid fraction = non-zero pixels / total
    if warped.ndim == 3:
        valid_mask = (warped.sum(axis=2) > 0)
    else:
        valid_mask = warped > 0
    valid_fraction = float(valid_mask.sum()) / (ref_h * ref_w)

    return warped, valid_fraction


# ---------------------------------------------------------------------------
# Write GeoTIFF
# ---------------------------------------------------------------------------

def write_geotiff(
    warped: np.ndarray,
    output_path: Path,
    ref_tif_path: Optional[Path] = None,
) -> None:
    """
    Write warped image as GeoTIFF, inheriting CRS and transform from reference.
    Falls back to a plain TIFF if rasterio unavailable or ref has no georeference.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_RASTERIO and ref_tif_path and ref_tif_path.exists():
        with rasterio.open(ref_tif_path) as ref_ds:
            ref_transform = ref_ds.transform
            ref_crs = ref_ds.crs
            ref_count = ref_ds.count
            ref_dtype = ref_ds.dtypes[0]

        h, w = warped.shape[:2]
        count = 1 if warped.ndim == 2 else warped.shape[2]

        with rasterio.open(
            output_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=count,
            dtype=str(warped.dtype),
            crs=ref_crs,
            transform=ref_transform,
            compress="lzw",
        ) as dst:
            if warped.ndim == 2:
                dst.write(warped, 1)
            else:
                for b in range(count):
                    dst.write(warped[:, :, b], b + 1)
        logger.info("GeoTIFF written: %s (CRS=%s)", output_path, ref_crs)
    else:
        # Fallback: write plain image with cv2
        if _HAS_CV2:
            cv2.imwrite(str(output_path), warped)
            logger.warning(
                "Written as plain image (no CRS): %s — "
                "rasterio or ref_tif not available", output_path
            )
        else:
            logger.error("Cannot write GeoTIFF: neither rasterio nor cv2 available")
            raise RuntimeError("No image writer available")


# ---------------------------------------------------------------------------
# Write match points CSV + GCP file
# ---------------------------------------------------------------------------

def write_match_csv(
    matches: dict,
    geometry: dict,
    ref_tif_path: Optional[Path],
    output_csv: Path,
    output_gcp: Path,
) -> None:
    """
    Write match_points.csv and match_points.gcp.

    CSV columns: src_col, src_row, ref_col, ref_row, lon, lat, residual_px
    GCP format:  loadable by gdal_translate -gcp
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build reference pixel → (lon, lat) lookup via rasterio if available
    ref_transform = None
    if _HAS_RASTERIO and ref_tif_path and ref_tif_path.exists():
        with rasterio.open(ref_tif_path) as ds:
            ref_transform = ds.transform

    def pix_to_lonlat(col: float, row: float) -> Tuple[float, float]:
        if ref_transform is None:
            return (0.0, 0.0)
        lon, lat = ref_transform * (col + 0.5, row + 0.5)
        return float(lon), float(lat)

    # Compute residuals from model
    H = _build_transform_matrix(geometry)
    match_list = matches.get("matches", [])

    rows_csv = []
    rows_gcp = []

    for m in match_list:
        if not m.get("is_inlier", True):
            continue

        src_col, src_row = m["src_xy"]
        ref_col_refined, ref_row_refined = m.get(
            "ref_xy_refined", m["ref_xy"]
        )

        # Warp src through model to get predicted ref position
        pt = np.array([[[src_col, src_row]]], dtype=np.float32)
        if H is not None and not geometry.get("tilewise"):
            pred = cv2.perspectiveTransform(pt, H)[0][0] if _HAS_CV2 else (ref_col_refined, ref_row_refined)
        else:
            pred = (ref_col_refined, ref_row_refined)

        pred_col, pred_row = float(pred[0]), float(pred[1])
        residual = float(np.sqrt(
            (pred_col - ref_col_refined)**2 + (pred_row - ref_row_refined)**2
        ))

        lon, lat = pix_to_lonlat(ref_col_refined, ref_row_refined)

        rows_csv.append({
            "src_col": src_col, "src_row": src_row,
            "ref_col": ref_col_refined, "ref_row": ref_row_refined,
            "lon": lon, "lat": lat,
            "residual_px": residual,
        })
        rows_gcp.append((src_col, src_row, lon, lat))

    # Write CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("src_col,src_row,ref_col,ref_row,lon,lat,residual_px\n")
        for r in rows_csv:
            f.write(
                f"{r['src_col']:.4f},{r['src_row']:.4f},"
                f"{r['ref_col']:.4f},{r['ref_row']:.4f},"
                f"{r['lon']:.8f},{r['lat']:.8f},{r['residual_px']:.6f}\n"
            )
    logger.info("Match CSV: %d points → %s", len(rows_csv), output_csv)

    # Write GCP file (GDAL format: -gcp pixel line lon lat [elev])
    with open(output_gcp, "w", encoding="utf-8") as f:
        for col, row, lon, lat in rows_gcp:
            f.write(f"-gcp {col:.4f} {row:.4f} {lon:.8f} {lat:.8f}\n")
    logger.info("GCP file: %d points → %s", len(rows_gcp), output_gcp)


# ---------------------------------------------------------------------------
# QC artifact: checkerboard overlay
# ---------------------------------------------------------------------------

def make_checkerboard(
    warped: np.ndarray,
    ref_image: np.ndarray,
    tile_px: int = QC_CHECKERBOARD_TILE_PX,
) -> np.ndarray:
    """
    Create checkerboard interleaving of warped source and reference.
    Alternates between source and reference tiles of tile_px size.
    """
    h = min(warped.shape[0], ref_image.shape[0])
    w = min(warped.shape[1], ref_image.shape[1])

    def to_rgb(img: np.ndarray) -> np.ndarray:
        img = img[:h, :w]
        img_n = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img_n.ndim == 2:
            img_n = cv2.cvtColor(img_n, cv2.COLOR_GRAY2BGR)
        return img_n

    src_rgb = to_rgb(warped)
    ref_rgb = to_rgb(ref_image)
    result = src_rgb.copy()

    for r in range(0, h, tile_px):
        for c in range(0, w, tile_px):
            tile_r = r // tile_px
            tile_c = c // tile_px
            use_ref = (tile_r + tile_c) % 2 == 1
            r1 = min(r + tile_px, h)
            c1 = min(c + tile_px, w)
            if use_ref:
                result[r:r1, c:c1] = ref_rgb[r:r1, c:c1]

    return result


# ---------------------------------------------------------------------------
# QC artifact: match overlay
# ---------------------------------------------------------------------------

def make_match_overlay(
    ref_image: np.ndarray,
    match_rows: List[dict],
) -> np.ndarray:
    """
    Draw matches on reference image with colour-coded residuals:
      green  < 0.5 px
      yellow 0.5–1.0 px
      red    > 1.0 px
    """
    if not _HAS_CV2:
        return ref_image

    img = cv2.normalize(ref_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()

    for m in match_rows:
        res = m.get("residual_px", 0.0)
        col = int(round(m["ref_col"]))
        row = int(round(m["ref_row"]))
        if res < MATCH_OVERLAY_GREEN_THRESH:
            color = (0, 255, 0)    # green
        elif res < MATCH_OVERLAY_YELLOW_THRESH:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 0, 255)    # red
        cv2.circle(img, (col, row), 3, color, -1, lineType=cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# QC artifact: residual heat map
# ---------------------------------------------------------------------------

def make_residual_heatmap(
    ref_shape: Tuple[int, int],
    match_rows: List[dict],
    sigma: int = RESIDUAL_HEATMAP_SIGMA,
) -> np.ndarray:
    """
    Create Gaussian heat map of registration residuals.
    Each match spreads its residual over a Gaussian blob of radius sigma px.
    """
    h, w = ref_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for m in match_rows:
        res = float(m.get("residual_px", 0.0))
        col = int(round(m["ref_col"]))
        row = int(round(m["ref_row"]))
        r0 = max(0, row - sigma * 3)
        r1 = min(h, row + sigma * 3 + 1)
        c0 = max(0, col - sigma * 3)
        c1 = min(w, col + sigma * 3 + 1)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                d2 = (rr - row) ** 2 + (cc - col) ** 2
                heat[rr, cc] += res * np.exp(-d2 / (2 * sigma ** 2))

    # Normalise and apply colour map
    heat_n = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if _HAS_CV2:
        heatmap_colour = cv2.applyColorMap(heat_n, cv2.COLORMAP_JET)
    else:
        heatmap_colour = np.stack([heat_n] * 3, axis=2)

    return heatmap_colour


# ---------------------------------------------------------------------------
# Main registration function
# ---------------------------------------------------------------------------

def register_pair(
    pair_id: str,
    matcher: str,
    geometry_path: Path,
    matches_path: Path,
    src_tif_path: Path,
    ref_tif_path: Path,
    output_dir: Path,
    failures_path: Path,
) -> int:
    """
    Full S8 pipeline for one (pair, matcher).

    Returns 0 on success, 1 on gate failure.
    """
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    geometry = load_geometry(geometry_path)
    matches  = load_matches(matches_path)

    if not _HAS_CV2:
        logger.error("OpenCV not available — cannot warp image")
        return 3

    src_image = cv2.imread(str(src_tif_path), cv2.IMREAD_UNCHANGED)
    ref_image = cv2.imread(str(ref_tif_path), cv2.IMREAD_UNCHANGED)

    if src_image is None or ref_image is None:
        logger.error("Failed to load src or ref image for %s", pair_id)
        return 1

    ref_shape = ref_image.shape[:2]

    # --- Warp ---
    try:
        warped, valid_fraction = warp_image(src_image, geometry, ref_shape)
    except Exception as exc:
        logger.error("Warp failed for %s/%s: %s", pair_id, matcher, exc)
        _write_failure(failures_path, pair_id, matcher, "S8", str(exc))
        return 1

    logger.info("%s/%s: valid warp fraction = %.3f", pair_id, matcher, valid_fraction)

    gate_passed = valid_fraction >= VALID_WARP_FRACTION_MIN
    if not gate_passed:
        logger.warning(
            "S8 gate FAILED: valid_fraction=%.3f < %.2f for %s/%s",
            valid_fraction, VALID_WARP_FRACTION_MIN, pair_id, matcher,
        )

    # --- registered.tif ---
    reg_path = output_dir / "registered.tif"
    write_geotiff(warped, reg_path, ref_tif_path)

    # --- match_points.csv + .gcp ---
    csv_path = output_dir / "match_points.csv"
    gcp_path = output_dir / "match_points.gcp"
    write_match_csv(matches, geometry, ref_tif_path, csv_path, gcp_path)

    # Reload CSV rows for QC
    match_rows: List[dict] = []
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                match_rows.append({
                    "ref_col": float(row["ref_col"]),
                    "ref_row": float(row["ref_row"]),
                    "residual_px": float(row["residual_px"]),
                })

    # --- QC: checkerboard ---
    cb_path = output_dir / "qc_checkerboard.png"
    cb = make_checkerboard(warped, ref_image)
    cv2.imwrite(str(cb_path), cb)
    logger.info("QC checkerboard: %s", cb_path)

    # --- QC: match overlay ---
    ov_path = output_dir / "qc_matches.png"
    ov = make_match_overlay(ref_image, match_rows)
    cv2.imwrite(str(ov_path), ov)
    logger.info("QC match overlay: %s", ov_path)

    # --- QC: residual heat map ---
    hm_path = output_dir / "qc_residuals.png"
    hm = make_residual_heatmap(ref_shape, match_rows)
    cv2.imwrite(str(hm_path), hm)
    logger.info("QC residual heatmap: %s", hm_path)

    elapsed = time.time() - t0
    logger.info(
        "S8 done for %s/%s in %.1fs — valid=%.1f%%, gate=%s",
        pair_id, matcher, elapsed, valid_fraction * 100, "PASS" if gate_passed else "FAIL",
    )

    return 0 if gate_passed else 1


def _write_failure(failures_path: Path, pair_id: str, matcher: str, stage: str, reason: str) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "pair_id": pair_id,
        "matcher": matcher,
        "stage": stage,
        "reason": reason,
        "fallback_taken": "none",
    }
    with open(failures_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="S8 — Product generation: warp + GeoTIFF + CSV + QC artifacts"
    )
    parser.add_argument("--pair",     required=True, help="pair_id")
    parser.add_argument("--matcher",  required=True, help="matcher_id (sift|rift2|lightglue|crater)")
    parser.add_argument("--geometry", required=True, help="Path to geometry.json")
    parser.add_argument("--matches",  required=True, help="Path to matches_refined.json")
    parser.add_argument("--src-tif",  required=True, dest="src_tif", help="Preprocessed source TIF")
    parser.add_argument("--ref-tif",  required=True, dest="ref_tif", help="Reference TIF")
    parser.add_argument("--out-dir",  required=True, dest="out_dir",
                        help="Output directory (results/<pair_id>/<matcher>/)")
    parser.add_argument("--failures", default="results/failures.jsonl",
                        help="Path to failures.jsonl (append)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Version check
    if not _HAS_CV2:
        logger.error("OpenCV not found. Install: pip install opencv-python")
        return 2
    if not _HAS_RASTERIO:
        logger.warning("rasterio not found — GeoTIFF will be written without CRS")

    rc = register_pair(
        pair_id=args.pair,
        matcher=args.matcher,
        geometry_path=Path(args.geometry),
        matches_path=Path(args.matches),
        src_tif_path=Path(args.src_tif),
        ref_tif_path=Path(args.ref_tif),
        output_dir=Path(args.out_dir),
        failures_path=Path(args.failures),
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""
src/registration/tilewise.py
-----------------------------
F17 — Tile-wise Local Models (ARCHITECTURE.md L4)

Fallback for pairs at latitude > ±55° or high relief, where a single
global homography is unreliable. Fits local affine/homography models
on overlapping tiles and blends them with Gaussian distance weighting.

Blending formula (MANDATORY per FEATURES.md F17):
    w_T(x) = exp(-‖x - c_T‖² / (2·σ²)),  σ = 256 px
    Weights normalized to sum to 1. NOT uniform averaging.

Coordinate convention: (col, row) = (x, y), 0-indexed. NEVER (row, col).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

logger = logging.getLogger(__name__)

# Per CONFIGURATION.md §6
TILE_SIZE_PX: int = 512
OVERLAP_PX: int = 256          # 50% overlap
MIN_INLIERS_PER_TILE: int = 12
GAUSSIAN_SIGMA_PX: float = 256.0
RANSAC_THRESHOLD_DEFAULT: float = 1.5
RANSAC_ITER: int = 10_000
RANSAC_CONF: float = 0.99999


@dataclass
class TileModel:
    tile_id: str
    center_col: float
    center_row: float
    model_type: str         # "affine" | "homography"
    model_matrix: np.ndarray
    inlier_count: int
    rmse_px: float


def _residuals_H(src_xy: np.ndarray, ref_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    n = len(src_xy)
    pts = np.hstack([src_xy, np.ones((n, 1), dtype=np.float64)])
    proj = (H @ pts.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj_xy - ref_xy, axis=1)


def _fit_tile_model(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    t_gsd: float,
    tile_id: str,
    center: Tuple[float, float],
) -> Optional[TileModel]:
    """Fit affine, fallback to homography, for one tile's inlier set."""
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    if not _HAS_CV2 or len(src_xy) < MIN_INLIERS_PER_TILE:
        return None

    src = src_xy.astype(np.float32)
    ref = ref_xy.astype(np.float32)

    # Try affine first (6-DOF — safer for small tile areas)
    M, mask = cv2.estimateAffine2D(
        src, ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=t_gsd,
        maxIters=RANSAC_ITER,
        confidence=RANSAC_CONF,
    )
    model_type = "affine"

    if M is None or int(mask.sum()) < MIN_INLIERS_PER_TILE:
        # Fallback to homography
        H, mask_h = cv2.findHomography(src, ref, cv2.RANSAC, t_gsd)
        if H is not None and int(mask_h.sum()) >= MIN_INLIERS_PER_TILE:
            M_full = H
            mask = mask_h.ravel().astype(bool)
            model_type = "homography"
        else:
            logger.debug("Tile %s: not enough inliers for any model", tile_id)
            return None
    else:
        # Embed 2x3 affine into 3x3
        M_full = np.eye(3, dtype=np.float64)
        M_full[:2, :] = M
        mask = mask.ravel().astype(bool)

    inlier_count = int(mask.sum())
    inlier_src = src_xy[mask]
    inlier_ref = ref_xy[mask]
    residuals = _residuals_H(inlier_src, inlier_ref, M_full)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    logger.debug(
        "Tile %s [%s]: %d inliers, RMSE=%.3f px", tile_id, model_type, inlier_count, rmse
    )

    return TileModel(
        tile_id=tile_id,
        center_col=center[0],
        center_row=center[1],
        model_type=model_type,
        model_matrix=M_full,
        inlier_count=inlier_count,
        rmse_px=rmse,
    )


def _gaussian_weight(
    query_col: float,
    query_row: float,
    center_col: float,
    center_row: float,
    sigma: float = GAUSSIAN_SIGMA_PX,
) -> float:
    """Gaussian weight for a point at (query_col, query_row) w.r.t. tile centre."""
    dist2 = (query_col - center_col) ** 2 + (query_row - center_row) ** 2
    return float(np.exp(-dist2 / (2.0 * sigma ** 2)))


def tilewise_models(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    src_shape: Tuple[int, int],
    ref_shape: Tuple[int, int],
    t_gsd: float = RANSAC_THRESHOLD_DEFAULT,
    ref_gsd_m: float = 0.5,
    tile_size: int = TILE_SIZE_PX,
    overlap_px: int = OVERLAP_PX,
    min_inliers: int = MIN_INLIERS_PER_TILE,
    sigma: float = GAUSSIAN_SIGMA_PX,
    trigger_reason: str = "high_latitude_or_relief",
) -> Optional["ModelResult"]:  # noqa: F821  (ModelResult imported at runtime)
    """
    Fit tile-wise local models over the source image.

    For each tile:
      1. Select matches whose src_xy falls inside the tile
      2. Fit affine (fallback homography) with RANSAC
      3. Store TileModel with centre coordinates

    Returns a ModelResult with tilewise=True and tile_models list.
    Returns None if no tile produces enough inliers.
    """
    from src.registration.ladder import ModelResult  # avoid circular at module load

    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

    src_h, src_w = src_shape
    step = tile_size - overlap_px  # step = 256 for 512 tile, 256 overlap

    tile_results: List[TileModel] = []

    col_starts = list(range(0, src_w - overlap_px, step))
    row_starts = list(range(0, src_h - overlap_px, step))

    for ri, row0 in enumerate(row_starts):
        for ci, col0 in enumerate(col_starts):
            col1 = min(col0 + tile_size, src_w)
            row1 = min(row0 + tile_size, src_h)
            tile_id = f"{ri}_{ci}"
            center = ((col0 + col1) / 2.0, (row0 + row1) / 2.0)

            # Select matches inside this tile
            in_tile = (
                (src_xy[:, 0] >= col0) & (src_xy[:, 0] < col1) &
                (src_xy[:, 1] >= row0) & (src_xy[:, 1] < row1)
            )
            tile_src = src_xy[in_tile]
            tile_ref = ref_xy[in_tile]

            if len(tile_src) < min_inliers:
                continue

            tm = _fit_tile_model(tile_src, tile_ref, t_gsd, tile_id, center)
            if tm is not None:
                tile_results.append(tm)

    if not tile_results:
        logger.warning("Tile-wise: no tiles produced enough inliers")
        return None

    # Aggregate stats
    total_inliers = sum(tm.inlier_count for tm in tile_results)
    mean_rmse = float(np.mean([tm.rmse_px for tm in tile_results]))

    logger.info(
        "Tile-wise: %d tiles fitted, total_inliers=%d, mean_RMSE=%.3f px",
        len(tile_results), total_inliers, mean_rmse,
    )

    # Build a dummy global model matrix (identity) — real warping uses tile_models
    tile_dicts = [
        {
            "tile_id": tm.tile_id,
            "model_type": tm.model_type,
            "model_matrix": tm.model_matrix.tolist(),
            "inlier_count": tm.inlier_count,
            "rmse_px": tm.rmse_px,
            "center_col": tm.center_col,
            "center_row": tm.center_row,
        }
        for tm in tile_results
    ]

    inlier_ratio = total_inliers / max(len(src_xy), 1)

    return ModelResult(
        model_type="tilewise",
        model_dof=0,
        ladder_level=-1,
        model_matrix=np.eye(3),
        inlier_mask=np.ones(len(src_xy), dtype=bool),   # approximate
        inlier_indices=np.arange(len(src_xy)),
        inlier_count=total_inliers,
        inlier_ratio=inlier_ratio,
        rmse_px=mean_rmse,
        t_gsd_used=t_gsd,
        tilewise=True,
        tile_models=tile_dicts,
        gsd_scale_factor=ref_gsd_m / 0.5,
    )


def blend_displacement(
    query_col: np.ndarray,
    query_row: np.ndarray,
    tile_models: List[Dict],
    sigma: float = GAUSSIAN_SIGMA_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gaussian-weighted blended displacement at query points.

    For each query point, computes the weighted mean displacement
    from all tile models whose model_matrix covers that area.

    Parameters
    ----------
    query_col, query_row : (N,) arrays — query pixel positions (col, row)
    tile_models : list of tile model dicts (from ModelResult.tile_models)
    sigma : Gaussian sigma in pixels (default 256 px per FEATURES.md F17)

    Returns
    -------
    (dcol, drow) : blended displacement in pixels
    """
    n = len(query_col)
    dcol_acc = np.zeros(n, dtype=np.float64)
    drow_acc = np.zeros(n, dtype=np.float64)
    w_acc    = np.zeros(n, dtype=np.float64)

    pts_h = np.stack([query_col, query_row, np.ones(n)], axis=0)  # (3, N)

    for tm in tile_models:
        H = np.array(tm["model_matrix"], dtype=np.float64)         # (3, 3)
        cc = float(tm["center_col"])
        cr = float(tm["center_row"])

        # Gaussian weight for each query point w.r.t. this tile centre
        dist2 = (query_col - cc) ** 2 + (query_row - cr) ** 2
        w = np.exp(-dist2 / (2.0 * sigma ** 2))                    # (N,)

        # Project through this tile's model
        proj = H @ pts_h                                            # (3, N)
        proj_col = proj[0] / proj[2]
        proj_row = proj[1] / proj[2]

        # Displacement this model predicts
        d_col = proj_col - query_col
        d_row = proj_row - query_row

        dcol_acc += w * d_col
        drow_acc += w * d_row
        w_acc    += w

    # Avoid div-by-zero where no tile covers a point
    safe = w_acc > 1e-12
    dcol_out = np.where(safe, dcol_acc / w_acc, 0.0)
    drow_out = np.where(safe, drow_acc / w_acc, 0.0)

    return dcol_out, drow_out

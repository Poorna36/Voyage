"""
src/registration/ladder.py
---------------------------
F16 — DEGENSAC Geometric Verification + Model Ladder (ARCHITECTURE.md L4)

Runs DEGENSAC with the model ladder: similarity → affine → homography.
Accepts the simplest model whose inlier RMSE <= stop_on_rmse_below.
Falls back to tile-wise local models for high-latitude / high-relief.

IMPORTANT — two separate thresholds (see INTERFACES.md §3 CLARIFICATION):
  t_gsd_used          : DEGENSAC reprojection threshold (px), GSD-based
  stop_on_rmse_below  : model-ladder acceptance RMSE (px), fixed at 1.0
  These must NOT be conflated.

Coordinate convention: (col, row) = (x, y). NEVER (row, col).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pydegensac
    _HAS_DEGENSAC = True
except ImportError:  # pragma: no cover
    _HAS_DEGENSAC = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

from src.registration.checks import f2_checks
from src.registration.declustering import decluster_and_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults (per CONFIGURATION.md §6)
# ---------------------------------------------------------------------------
RANSAC_ITER       = 10_000
RANSAC_CONF       = 0.99999
STOP_ON_RMSE      = 1.0      # model-ladder acceptance threshold (px)
T_GSD_MIN         = 0.5      # min t_gsd (px)
T_GSD_MAX         = 3.0      # max t_gsd (px)
T_GSD_GSD_FACTOR  = 1.0      # t_gsd = max(T_GSD_MIN, min(gsd_ratio * factor, T_GSD_MAX))
INLIER_RATIO_MIN  = 0.05
INLIER_COUNT_MIN  = 20
WIDEN_FACTOR      = 1.5      # widen t_gsd once on first failure


@dataclass
class ModelResult:
    """Result of one model estimation attempt."""
    model_type: str          # "similarity" | "affine" | "homography"
    model_dof: int
    ladder_level: int        # 0=similarity, 1=affine, 2=homography
    model_matrix: np.ndarray # 3x3
    inlier_mask: np.ndarray  # (N,) bool
    inlier_indices: np.ndarray
    inlier_count: int
    inlier_ratio: float
    rmse_px: float
    t_gsd_used: float
    ransac_method: str = "degensac"
    ransac_iter: int = RANSAC_ITER
    ransac_conf: float = RANSAC_CONF
    tilewise: bool = False
    tile_models: List[Dict] = field(default_factory=list)
    gsd_scale_factor: float = 1.0
    final_gcp_count: int = 0


def _compute_t_gsd(src_gsd_m: float, ref_gsd_m: float) -> float:
    """Compute DEGENSAC reprojection threshold from GSD ratio."""
    gsd_ratio = src_gsd_m / ref_gsd_m if ref_gsd_m > 0 else 1.0
    t = max(T_GSD_MIN, min(gsd_ratio * T_GSD_GSD_FACTOR, T_GSD_MAX))
    return t


def _residuals(src_xy: np.ndarray, ref_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Compute per-point reprojection residuals using homography H."""
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    n = len(src_xy)
    pts = np.hstack([src_xy, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
    proj = (H @ pts.T).T                                           # (N, 3)
    proj_xy = proj[:, :2] / proj[:, 2:3]                          # (N, 2)
    return np.linalg.norm(proj_xy - ref_xy, axis=1)               # (N,)


def _rmse(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) > 0 else np.inf


def _run_degensac(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    threshold: float,
    max_iter: int = RANSAC_ITER,
    confidence: float = RANSAC_CONF,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Run DEGENSAC homography estimation.
    Falls back to cv2.findHomography (RANSAC) if pydegensac unavailable.

    Returns (H, inlier_mask) where H is 3x3 or None on failure.
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    src = src_xy.astype(np.float64)
    ref = ref_xy.astype(np.float64)

    if _HAS_DEGENSAC:
        try:
            H, mask = pydegensac.findHomography(
                src, ref,
                px_th=threshold,
                conf=confidence,
                max_iters=max_iter,
            )
            return H, mask.astype(bool)
        except Exception as exc:  # pragma: no cover
            logger.warning("DEGENSAC failed (%s), falling back to cv2 RANSAC", exc)

    if _HAS_CV2:
        H, mask = cv2.findHomography(src, ref, cv2.RANSAC, threshold)
        if H is None:
            return None, np.zeros(len(src), dtype=bool)
        return H, mask.ravel().astype(bool)

    raise RuntimeError("Neither pydegensac nor cv2 is available for RANSAC.")


def _fit_similarity(
    src_xy: np.ndarray, ref_xy: np.ndarray, threshold: float
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Estimate similarity (4-DOF: scale, rotation, 2 translations) via RANSAC."""
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    if not _HAS_CV2:
        return None, np.zeros(len(src_xy), dtype=bool)
    src = src_xy.astype(np.float32)
    ref = ref_xy.astype(np.float32)
    M, mask = cv2.estimateAffinePartial2D(src, ref, method=cv2.RANSAC,
                                           ransacReprojThreshold=threshold,
                                           maxIters=RANSAC_ITER,
                                           confidence=RANSAC_CONF)
    if M is None:
        return None, np.zeros(len(src), dtype=bool)
    # embed 2x3 into 3x3
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H, mask.ravel().astype(bool)


def _fit_affine(
    src_xy: np.ndarray, ref_xy: np.ndarray, threshold: float
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Estimate affine (6-DOF) via RANSAC."""
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    if not _HAS_CV2:
        return None, np.zeros(len(src_xy), dtype=bool)
    src = src_xy.astype(np.float32)
    ref = ref_xy.astype(np.float32)
    M, mask = cv2.estimateAffine2D(src, ref, method=cv2.RANSAC,
                                    ransacReprojThreshold=threshold,
                                    maxIters=RANSAC_ITER,
                                    confidence=RANSAC_CONF)
    if M is None:
        return None, np.zeros(len(src), dtype=bool)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H, mask.ravel().astype(bool)


def _evaluate_model(
    H: np.ndarray,
    inlier_mask: np.ndarray,
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
) -> Tuple[float, float]:
    """Return (inlier_ratio, rmse_px) for a fitted model."""
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    n = len(src_xy)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / n if n > 0 else 0.0
    if inlier_count == 0:
        return inlier_ratio, np.inf
    res = _residuals(src_xy[inlier_mask], ref_xy[inlier_mask], H)
    return inlier_ratio, _rmse(res)


def model_ladder(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    confidence: np.ndarray,
    src_shape: Tuple[int, int],
    ref_shape: Tuple[int, int],
    src_gsd_m: float,
    ref_gsd_m: float,
    latitude_center_deg: float = 0.0,
    stop_on_rmse_below: float = STOP_ON_RMSE,
    min_spacing_px: float = 20.0,
    zscore_threshold: float = 3.0,
    buffer_px: int = 10,
) -> ModelResult:
    """
    Full L4 geometric verification pipeline:
      1. F2 checks (mandatory, before any RANSAC)
      2. Compute t_gsd from GSD ratio
      3. Try model ladder: similarity → affine → homography
         - Accept simplest model with RMSE <= stop_on_rmse_below
         - Widen t_gsd x1.5 once on first failure
      4. Tile-wise fallback if latitude > 55° or all global models fail
      5. GCP declustering + Z-score filter on inliers
      6. Return ModelResult with all fields for geometry.json

    Parameters
    ----------
    src_xy, ref_xy  : (N, 2) float — match coordinates (col, row)
    confidence      : (N,) float — per-match confidence
    src_shape       : (H, W) — source image shape
    ref_shape       : (H, W) — reference image shape
    src_gsd_m       : source GSD in m/px
    ref_gsd_m       : reference GSD in m/px
    latitude_center_deg : centre latitude of the pair (triggers tile-wise)
    stop_on_rmse_below  : model-ladder acceptance RMSE threshold (px)
    """
    assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Step 1: F2 checks
    # -----------------------------------------------------------------------
    f2 = f2_checks(src_xy, ref_xy, confidence, src_shape, ref_shape, buffer_px)
    src_f2 = f2.src_xy
    ref_f2 = f2.ref_xy

    logger.info(
        "L4 start: %d matches after F2 (oob=%d, dup=%d)",
        f2.final_count, f2.removed_oob, f2.removed_dup,
    )

    # -----------------------------------------------------------------------
    # Step 2: t_gsd computation
    # -----------------------------------------------------------------------
    t_gsd = _compute_t_gsd(src_gsd_m, ref_gsd_m)
    logger.debug("t_gsd=%.3f px (src_gsd=%.2f, ref_gsd=%.2f)", t_gsd, src_gsd_m, ref_gsd_m)

    # -----------------------------------------------------------------------
    # Step 3: Model ladder with optional t_gsd widening
    # -----------------------------------------------------------------------
    MODELS = [
        ("similarity",  0, _fit_similarity),
        ("affine",      1, _fit_affine),
        ("homography",  2, _run_degensac),
    ]
    DOF_MAP = {"similarity": 4, "affine": 6, "homography": 8}

    best: Optional[ModelResult] = None
    t_gsd_used = t_gsd

    for widen_attempt in range(2):  # attempt 0 = normal, attempt 1 = widened
        if widen_attempt == 1:
            t_gsd_used = min(t_gsd * WIDEN_FACTOR, T_GSD_MAX)
            logger.info("Widening t_gsd to %.3f px for retry", t_gsd_used)

        fitted_models: List[ModelResult] = []
        for model_name, level, fit_fn in MODELS:
            if len(src_f2) < 4:
                logger.warning("Too few matches (%d) for %s", len(src_f2), model_name)
                continue

            H, mask = fit_fn(src_f2, ref_f2, t_gsd_used)
            if H is None:
                continue

            inlier_count = int(mask.sum())
            inlier_ratio = inlier_count / len(src_f2) if len(src_f2) > 0 else 0.0

            if inlier_count < INLIER_COUNT_MIN or inlier_ratio < INLIER_RATIO_MIN:
                logger.debug(
                    "%s: insufficient inliers (%d, ratio=%.3f)",
                    model_name, inlier_count, inlier_ratio,
                )
                continue

            res = _residuals(src_f2[mask], ref_f2[mask], H)
            rmse = _rmse(res)

            logger.info(
                "%s: inliers=%d (%.1f%%), RMSE=%.3f px",
                model_name, inlier_count, inlier_ratio * 100, rmse,
            )

            result = ModelResult(
                model_type=model_name,
                model_dof=DOF_MAP[model_name],
                ladder_level=level,
                model_matrix=H,
                inlier_mask=mask,
                inlier_indices=np.where(mask)[0],
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                rmse_px=rmse,
                t_gsd_used=t_gsd_used,
            )
            fitted_models.append(result)

        if fitted_models:
            max_inliers = max(m.inlier_count for m in fitted_models)
            # Accept simplest model that explains >= 90% of max inliers with RMSE <= stop_on_rmse_below
            candidates = [m for m in fitted_models if m.inlier_count >= 0.90 * max_inliers and m.rmse_px <= stop_on_rmse_below]
            if candidates:
                best = min(candidates, key=lambda m: m.ladder_level)
                break
            else:
                # If none passes RMSE threshold, choose the one with lowest RMSE / highest inliers
                best = min(fitted_models, key=lambda m: (m.rmse_px, -m.inlier_count))

        if widen_attempt == 0 and (best is None or best.inlier_ratio < INLIER_RATIO_MIN):
            continue  # try widened
        else:
            break

    # -----------------------------------------------------------------------
    # Step 4: Tile-wise fallback
    # -----------------------------------------------------------------------
    use_tilewise = (
        abs(latitude_center_deg) > 55 or
        best is None or
        best.inlier_count < INLIER_COUNT_MIN
    )

    if use_tilewise:
        logger.info(
            "Triggering tile-wise models (lat=%.1f, global_failed=%s)",
            latitude_center_deg, best is None,
        )
        from src.registration.tilewise import tilewise_models
        tw_result = tilewise_models(
            src_f2, ref_f2, src_shape, ref_shape,
            t_gsd=t_gsd_used,
            ref_gsd_m=ref_gsd_m,
        )
        if tw_result is not None and (best is None or tw_result.inlier_count >= best.inlier_count):
            best = tw_result
            best.t_gsd_used = t_gsd_used

    if best is None:
        # Total failure — return an empty result so the caller can record it
        logger.error("L4 TOTAL FAILURE: no valid model found")
        return ModelResult(
            model_type="none", model_dof=0, ladder_level=-1,
            model_matrix=np.eye(3), inlier_mask=np.zeros(len(src_f2), dtype=bool),
            inlier_indices=np.array([], dtype=int),
            inlier_count=0, inlier_ratio=0.0, rmse_px=np.inf,
            t_gsd_used=t_gsd_used,
        )

    # -----------------------------------------------------------------------
    # Step 5: GCP declustering + Z-score on inliers
    # -----------------------------------------------------------------------
    if not best.tilewise and best.inlier_count > 0:
        inlier_src = src_f2[best.inlier_mask]
        inlier_ref = ref_f2[best.inlier_mask]
        inlier_res = _residuals(inlier_src, inlier_ref, best.model_matrix)

        d_src, d_ref, d_res, gsd_scale, final_gcp = decluster_and_filter(
            inlier_src, inlier_ref, inlier_res,
            ref_gsd_m=ref_gsd_m,
        )
        best.gsd_scale_factor = gsd_scale
        best.final_gcp_count = final_gcp

    logger.info(
        "L4 done: model=%s, inliers=%d, RMSE=%.3f px, t_gsd=%.2f, time=%.1fs",
        best.model_type, best.inlier_count, best.rmse_px,
        best.t_gsd_used, time.time() - t0,
    )
    return best

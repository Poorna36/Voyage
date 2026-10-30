"""
synthetic/anchors.py — GT Anchor Extraction Engine

Extracts natural high-gradient floating-point anchor points from a source image
to serve as hidden Ground Truth reference points for the Synthetic Benchmark.

Two extraction phases (per SYNTHETIC_BENCHMARK_ARCHITECTURE.md §4.1):
  Phase 1 (baseline): Shi-Tomasi corner detection across uniform grid cells.
  Phase 2+ (stratified): Morphological feature class quotas — craters, ridges,
    maria, shadow boundaries, highlands, polar terrain.

Coordinate convention: all pixel coordinates are (col, row) = (x, y), 0-indexed.
The returned AnchorSet is HIDDEN from the matching pipeline; it is only loaded
by the evaluation engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnchorPoint:
    """A single GT anchor point (in source image space, float precision)."""
    id: int
    src_x: float   # col (x), 0-indexed, sub-pixel float
    src_y: float   # row (y), 0-indexed, sub-pixel float
    feature_class: str   # "shi_tomasi" | "crater" | "ridge" | "maria" | "shadow_boundary" | "polar"
    gradient_magnitude: float  # Sobel gradient magnitude at this point


@dataclass
class AnchorSet:
    """Collection of GT anchors extracted from a source image."""
    pair_id: str
    image_shape: Tuple[int, int]   # (height_px, width_px)
    anchors: List[AnchorPoint] = field(default_factory=list)
    extraction_phase: int = 1
    n_grid_cells: Tuple[int, int] = (8, 8)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        d["image_shape"] = list(self.image_shape)
        d["n_grid_cells"] = list(self.n_grid_cells)
        return d

    def save(self, path: Path) -> None:
        """Save anchor set to JSON (hidden GT file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved %d anchors to %s", len(self.anchors), path)

    @classmethod
    def load(cls, path: Path) -> "AnchorSet":
        """Load anchor set from JSON."""
        with open(path) as f:
            d = json.load(f)
        anchors = [AnchorPoint(**a) for a in d.pop("anchors")]
        d["image_shape"] = tuple(d["image_shape"])
        d["n_grid_cells"] = tuple(d["n_grid_cells"])
        return cls(**d, anchors=anchors)

    def as_numpy(self) -> np.ndarray:
        """Return anchor source coordinates as (N, 2) float64 array (col, row)."""
        if not self.anchors:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([[a.src_x, a.src_y] for a in self.anchors], dtype=np.float64)


def _build_gradient_map(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude map (float32)."""
    img_f32 = image.astype(np.float32)
    gx = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def _extract_shi_tomasi_grid(
    image: np.ndarray,
    grid_cells: Tuple[int, int],
    max_corners_per_cell: int,
    quality_level: float,
    min_distance_px: float,
    block_size: int,
    gradient_map: Optional[np.ndarray] = None,
) -> List[AnchorPoint]:
    """Phase 1: Shi-Tomasi detection across a uniform NxM grid.

    Each grid cell is processed independently to ensure spatial spread.
    Returns anchor points sorted by descending gradient magnitude.
    """
    if gradient_map is None:
        gradient_map = _build_gradient_map(image)

    img_u8 = (image / image.max() * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    h, w = image.shape[:2]
    n_rows, n_cols = grid_cells
    cell_h = h // n_rows
    cell_w = w // n_cols

    all_anchors: List[AnchorPoint] = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, x0 = r * cell_h, c * cell_w
            y1, x1 = min(y0 + cell_h, h), min(x0 + cell_w, w)
            cell = img_u8[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            corners = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=max_corners_per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance_px,
                blockSize=block_size,
            )
            if corners is None:
                continue
            for pt in corners.reshape(-1, 2):
                global_x = float(pt[0]) + x0
                global_y = float(pt[1]) + y0
                global_x = float(np.clip(global_x, 0, w - 1))
                global_y = float(np.clip(global_y, 0, h - 1))
                grad = float(gradient_map[int(global_y), int(global_x)])
                all_anchors.append(AnchorPoint(
                    id=-1,
                    src_x=global_x,
                    src_y=global_y,
                    feature_class="shi_tomasi",
                    gradient_magnitude=grad,
                ))

    return all_anchors


def _extract_craters(
    image: np.ndarray,
    gradient_map: np.ndarray,
    quota: int,
    rng: np.random.Generator,
) -> List[AnchorPoint]:
    """Phase 2+: Crater rim / floor anchor extraction."""
    h, w = image.shape[:2]
    img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (5, 5), 1.5)

    anchors: List[AnchorPoint] = []
    min_r = max(5, min(h, w) // 40)
    max_r = max(min_r + 5, min(h, w) // 8)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,
        param1=80,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            if len(anchors) >= quota:
                break
            for angle_deg in [0, 90, 180, 270]:
                ang = np.deg2rad(angle_deg)
                rim_x = float(np.clip(cx + r * np.cos(ang), 0, w - 1))
                rim_y = float(np.clip(cy + r * np.sin(ang), 0, h - 1))
                grad = float(gradient_map[int(rim_y), int(rim_x)])
                anchors.append(AnchorPoint(
                    id=-1, src_x=rim_x, src_y=rim_y,
                    feature_class="crater", gradient_magnitude=grad,
                ))
            floor_x = float(np.clip(cx + rng.uniform(-r * 0.3, r * 0.3), 0, w - 1))
            floor_y = float(np.clip(cy + rng.uniform(-r * 0.3, r * 0.3), 0, h - 1))
            grad = float(gradient_map[int(floor_y), int(floor_x)])
            anchors.append(AnchorPoint(
                id=-1, src_x=floor_x, src_y=floor_y,
                feature_class="crater", gradient_magnitude=grad,
            ))

    logger.debug("_extract_craters: found %d anchors (quota=%d)", len(anchors), quota)
    return anchors[:quota]


def _extract_ridges(
    image: np.ndarray,
    gradient_map: np.ndarray,
    quota: int,
) -> List[AnchorPoint]:
    """Phase 2+: Ridge / scarp anchor extraction."""
    h, w = image.shape[:2]
    img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)

    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    top_hat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, kernel_line)
    kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    top_hat_h = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, kernel_line_h)
    ridge_response = cv2.addWeighted(top_hat, 0.5, top_hat_h, 0.5, 0).astype(np.float32)

    thresh = float(np.percentile(ridge_response, 85))
    ridge_mask = ridge_response > thresh

    ys, xs = np.where(ridge_mask)
    if len(xs) == 0:
        return []

    grads = gradient_map[ys, xs].astype(np.float64)
    order = np.argsort(grads)[::-1]
    xs, ys = xs[order], ys[order]

    anchors: List[AnchorPoint] = []
    used_positions = []
    min_spacing = 20.0
    for x, y in zip(xs.tolist(), ys.tolist()):
        if len(anchors) >= quota:
            break
        too_close = any(np.hypot(x - px, y - py) < min_spacing for px, py in used_positions)
        if too_close:
            continue
        grad = float(gradient_map[int(y), int(x)])
        anchors.append(AnchorPoint(
            id=-1, src_x=float(x), src_y=float(y),
            feature_class="ridge", gradient_magnitude=grad,
        ))
        used_positions.append((x, y))

    return anchors


def _extract_maria(
    image: np.ndarray,
    gradient_map: np.ndarray,
    quota: int,
    rng: np.random.Generator,
) -> List[AnchorPoint]:
    """Phase 2+: Flat-textured maria region anchor extraction."""
    h, w = image.shape[:2]
    grad_threshold = float(np.percentile(gradient_map, 40))
    maria_mask = gradient_map <= grad_threshold

    ys, xs = np.where(maria_mask)
    if len(xs) == 0:
        return []

    n_sample = min(quota * 5, len(xs))
    indices = rng.choice(len(xs), size=n_sample, replace=False)
    sample_xs, sample_ys = xs[indices], ys[indices]

    anchors: List[AnchorPoint] = []
    used_positions = []
    min_spacing = 25.0
    for x, y in zip(sample_xs.tolist(), sample_ys.tolist()):
        if len(anchors) >= quota:
            break
        too_close = any(np.hypot(x - px, y - py) < min_spacing for px, py in used_positions)
        if too_close:
            continue
        grad = float(gradient_map[int(y), int(x)])
        anchors.append(AnchorPoint(
            id=-1, src_x=float(x), src_y=float(y),
            feature_class="maria", gradient_magnitude=grad,
        ))
        used_positions.append((x, y))

    return anchors


def _extract_shadow_boundaries(
    image: np.ndarray,
    gradient_map: np.ndarray,
    quota: int,
) -> List[AnchorPoint]:
    """Phase 2+: Solar terminator / shadow boundary anchor extraction."""
    h, w = image.shape[:2]
    dark_thresh = float(np.percentile(image, 20))
    dark_mask = image < dark_thresh

    kernel = np.ones((5, 5), np.uint8)
    boundary_mask = cv2.dilate(dark_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary_mask = boundary_mask & (~dark_mask)

    high_grad_thresh = float(np.percentile(gradient_map[boundary_mask], 70)) if boundary_mask.any() else 0
    final_mask = boundary_mask & (gradient_map > high_grad_thresh)

    ys, xs = np.where(final_mask)
    if len(xs) == 0:
        grad_thresh_fallback = float(np.percentile(gradient_map, 80))
        ys, xs = np.where(gradient_map > grad_thresh_fallback)
        if len(xs) == 0:
            return []

    grads = gradient_map[ys, xs]
    order = np.argsort(grads)[::-1]
    xs, ys = xs[order], ys[order]

    anchors: List[AnchorPoint] = []
    used_positions = []
    min_spacing = 20.0
    for x, y in zip(xs.tolist(), ys.tolist()):
        if len(anchors) >= quota:
            break
        too_close = any(np.hypot(x - px, y - py) < min_spacing for px, py in used_positions)
        if too_close:
            continue
        grad = float(gradient_map[int(y), int(x)])
        anchors.append(AnchorPoint(
            id=-1, src_x=float(x), src_y=float(y),
            feature_class="shadow_boundary", gradient_magnitude=grad,
        ))
        used_positions.append((x, y))

    return anchors


def _extract_polar_terrain(
    image: np.ndarray,
    gradient_map: np.ndarray,
    quota: int,
    rng: np.random.Generator,
    n_stripes: int = 4,
) -> List[AnchorPoint]:
    """Phase 2+: High-incidence-angle polar terrain anchor extraction."""
    h, w = image.shape[:2]
    stripe_h = h // n_stripes

    all_candidates: List[Tuple[float, float, float]] = []

    for i in range(n_stripes):
        y0, y1 = i * stripe_h, min((i + 1) * stripe_h, h)
        stripe_grad = gradient_map[y0:y1, :]
        thresh = float(np.percentile(stripe_grad, 75))
        local_ys, local_xs = np.where(stripe_grad > thresh)
        for lx, ly in zip(local_xs.tolist(), local_ys.tolist()):
            g = float(gradient_map[y0 + ly, lx])
            all_candidates.append((g, float(lx), float(y0 + ly)))

    if not all_candidates:
        return []

    all_candidates.sort(key=lambda t: t[0], reverse=True)

    anchors: List[AnchorPoint] = []
    used_positions = []
    min_spacing = 20.0
    for grad, x, y in all_candidates:
        if len(anchors) >= quota:
            break
        too_close = any(np.hypot(x - px, y - py) < min_spacing for px, py in used_positions)
        if too_close:
            continue
        anchors.append(AnchorPoint(
            id=-1, src_x=x, src_y=y,
            feature_class="polar", gradient_magnitude=grad,
        ))
        used_positions.append((x, y))

    return anchors


def _deduplicate_anchors(
    anchors: List[AnchorPoint],
    min_spacing_px: float,
) -> List[AnchorPoint]:
    """Remove anchors closer than min_spacing_px using greedy distance pruning."""
    if not anchors:
        return anchors
    anchors = sorted(anchors, key=lambda a: a.gradient_magnitude, reverse=True)
    pts = np.array([[a.src_x, a.src_y] for a in anchors], dtype=np.float64)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(pts)):
            if keep[j]:
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < min_spacing_px:
                    keep[j] = False
    return [a for a, k in zip(anchors, keep) if k]


def extract_anchors(
    image: np.ndarray,
    pair_id: str,
    config: dict,
    rng: Optional[np.random.Generator] = None,
) -> AnchorSet:
    """Extract GT anchor points from a source image."""
    assert image.ndim == 2, "extract_anchors expects a single-channel 2D image."
    assert image.size > 0, "Image must be non-empty."

    if rng is None:
        rng = np.random.default_rng(42)

    cfg_a = config.get("anchors", {})
    target_count: int = cfg_a.get("target_count", 80)
    min_count: int = cfg_a.get("min_count", 60)
    max_count: int = cfg_a.get("max_count", 100)
    grid_cells_cfg = cfg_a.get("grid_cells", [8, 8])
    grid_cells = tuple(grid_cells_cfg)
    phase: int = cfg_a.get("phase", 1)
    st_cfg = cfg_a.get("shi_tomasi", {})
    buckets_cfg = cfg_a.get("stratification_buckets", {})

    gradient_map = _build_gradient_map(image)
    anchors: List[AnchorPoint] = []

    if phase >= 1:
        st_anchors = _extract_shi_tomasi_grid(
            image=image,
            grid_cells=grid_cells,
            max_corners_per_cell=st_cfg.get("max_corners_per_cell", 20),
            quality_level=st_cfg.get("quality_level", 0.01),
            min_distance_px=st_cfg.get("min_distance_px", 15),
            block_size=st_cfg.get("block_size", 7),
            gradient_map=gradient_map,
        )
        anchors.extend(st_anchors)

    if phase >= 2:
        hg_frac = float(buckets_cfg.get("high_gradient_terrain", 0.35))
        crater_frac = float(buckets_cfg.get("craters", 0.20))
        ridge_frac = float(buckets_cfg.get("ridges", 0.10))
        maria_frac = float(buckets_cfg.get("maria", 0.15))
        shadow_frac = float(buckets_cfg.get("shadow_boundaries", 0.10))
        polar_frac = float(buckets_cfg.get("polar_terrain", 0.10))

        hg_quota = max(1, int(np.ceil(target_count * hg_frac)))
        crater_quota = max(1, int(np.ceil(target_count * crater_frac)))
        ridge_quota = max(1, int(np.ceil(target_count * ridge_frac)))
        maria_quota = max(1, int(np.ceil(target_count * maria_frac)))
        shadow_quota = max(1, int(np.ceil(target_count * shadow_frac)))
        polar_quota = max(1, int(np.ceil(target_count * polar_frac)))

        hg_anchors = sorted(
            st_anchors, key=lambda a: a.gradient_magnitude, reverse=True
        )[:hg_quota]

        crater_anchors = _extract_craters(image, gradient_map, crater_quota, rng)
        ridge_anchors = _extract_ridges(image, gradient_map, ridge_quota)
        maria_anchors = _extract_maria(image, gradient_map, maria_quota, rng)
        shadow_anchors = _extract_shadow_boundaries(image, gradient_map, shadow_quota)
        polar_anchors = _extract_polar_terrain(image, gradient_map, polar_quota, rng)

        anchors = (
            hg_anchors
            + crater_anchors
            + ridge_anchors
            + maria_anchors
            + shadow_anchors
            + polar_anchors
        )

        if len(anchors) < min_count:
            remaining = [a for a in st_anchors if a not in hg_anchors]
            anchors += remaining
            logger.warning(
                "Phase 2+ stratified extraction yielded only %d anchors for pair '%s'; padded with %d Shi-Tomasi anchors.",
                len(anchors) - len(remaining), pair_id, len(remaining),
            )

    min_spacing = st_cfg.get("min_distance_px", 15)
    anchors = _deduplicate_anchors(anchors, min_spacing_px=min_spacing)

    if len(anchors) > max_count:
        anchors = anchors[:max_count]

    if len(anchors) < min_count:
        raise RuntimeError(
            f"extract_anchors: Only found {len(anchors)} anchors for pair '{pair_id}' "
            f"(minimum required: {min_count}). Check image quality and Shi-Tomasi quality_level parameter."
        )

    for i, a in enumerate(anchors):
        a.id = i + 1

    return AnchorSet(
        pair_id=pair_id,
        image_shape=image.shape[:2],
        anchors=anchors,
        extraction_phase=phase,
        n_grid_cells=grid_cells,
    )

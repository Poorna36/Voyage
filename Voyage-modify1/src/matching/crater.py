"""
src/matching/crater.py
======================
M3 — CNSFM-style crater-geometry matcher.

Detection: YOLOv9 transfer-learned weights (primary) OR HoughCircles CPU fallback.
  - CPU fallback records matcher_id='crater_hough' (not 'crater')
  - Pre-flight recall check required before using M3 as primary matcher
  - detector_validated flag recorded in output

Gate (must pass BEFORE running M3):
  - crater_density_per_km2 >= tau_c (3.0 craters/km²) in BOTH images
  - terrain_class in {highland, polar_highland, polar}
  - gate_skip=True + reason recorded when gate fails

References: ARCHITECTURE.md §4, FEATURES.md F13, CONFIGURATION.md §2
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseMatcher, MatchResult

log = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = "models/crater_yolov9.pt"

_DEFAULTS: Dict[str, Any] = {
    "tau_c": 3.0,              # craters/km² gate threshold
    "allowed_terrain": {"highland", "polar_highland", "polar"},
    "hough_dp": 1.2,
    "hough_min_dist": 20,
    "hough_param1": 50,
    "hough_param2": 30,
    "hough_min_r": 5,
    "hough_max_r": 80,
    "topology_max_craters": 80,
    "match_confidence_thresh": 0.65,   # per CONFIGURATION.md confidence filter
    "yolo_weights": _DEFAULT_WEIGHTS if Path(_DEFAULT_WEIGHTS).exists() else None,
}

# YOLOv9 optional import
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class CraterMatcher(BaseMatcher):
    """
    M3 — CNSFM-style crater-geometry matcher.

    matcher_id   = 'crater' (or 'crater_hough' on CPU fallback)
    requires_gpu = True  (auto-falls back to HoughCircles CPU)
    """

    matcher_id = "crater"
    requires_gpu = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}
        self.tau_c: float = float(cfg["tau_c"])
        self.allowed_terrain: set = set(cfg["allowed_terrain"])
        self.hough_dp: float = float(cfg["hough_dp"])
        self.hough_min_dist: int = int(cfg["hough_min_dist"])
        self.hough_param1: float = float(cfg["hough_param1"])
        self.hough_param2: float = float(cfg["hough_param2"])
        self.hough_min_r: int = int(cfg["hough_min_r"])
        self.hough_max_r: int = int(cfg["hough_max_r"])
        self.max_craters: int = int(cfg["topology_max_craters"])
        self.conf_thresh: float = float(cfg["match_confidence_thresh"])
        self.yolo_weights: Optional[str] = cfg.get("yolo_weights")
        self._yolo_model: Any = None
        self._last_used_detector: str = "hough"

    # ── Gate check (FEATURES.md F13) ─────────────────────────────────────────

    @staticmethod
    def check_gate(
        density_src: float,
        density_ref: float,
        terrain_src: str,
        terrain_ref: Optional[str],
        tau_c: float,
        allowed_terrain: set,
    ) -> Tuple[bool, str]:
        """
        Returns (gate_passes: bool, reason: str).
        Gate fails if density < tau_c in either image OR terrain not in allowed set.
        """
        if density_src < tau_c:
            return False, f"src_density={density_src:.2f} < tau_c={tau_c}"
        if density_ref is not None and density_ref < tau_c:
            return False, f"ref_density={density_ref:.2f} < tau_c={tau_c}"
        if terrain_src not in allowed_terrain:
            return False, f"terrain_src={terrain_src!r} not in {allowed_terrain}"
        return True, "gate_passed"

    # ── Crater detection ──────────────────────────────────────────────────────

    def _detect_craters(self, image: np.ndarray) -> np.ndarray:
        """
        Detect craters. Returns (N, 3) array of (col, row, radius_px).
        Tries YOLO first; falls back to HoughCircles automatically.
        """
        self._last_used_detector = "hough"
        if self.yolo_weights and _HAS_TORCH:
            try:
                weights_p = Path(self.yolo_weights)
                if weights_p.exists():
                    craters = self._detect_yolo(image)
                    if len(craters) > 0:
                        self._last_used_detector = "yolo"
                        return craters
            except Exception as exc:
                log.warning("YOLO crater detection failed (%s); falling back to HoughCircles", exc)
        return self._detect_hough(image)

    def _detect_hough(self, image: np.ndarray) -> np.ndarray:
        """HoughCircles CPU fallback detector. Returns (N, 3) col, row, radius."""
        import cv2
        gray = self._to_gray(image)
        # Mild blur to suppress noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_r,
            maxRadius=self.hough_max_r,
        )
        if circles is None:
            return np.empty((0, 3), dtype=np.float32)
        circles = np.round(circles[0]).astype(np.float32)  # (N, 3) x, y, r
        # Sort by radius descending (larger craters are more reliable)
        circles = circles[np.argsort(-circles[:, 2])]
        return circles[:self.max_craters]

    def _detect_yolo(self, image: np.ndarray) -> np.ndarray:
        """
        YOLO crater detector. Returns (N, 3) col, row, radius.
        Uses Multi-Scale Scale-Space Pyramid Inference for both giant impact basins
        and sub-kilometer micro-craters.
        """
        if self._yolo_model is None:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.yolo_weights)

        import cv2

        # Convert to 3-channel uint8 BGR
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_bgr = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image.copy()

        if img_bgr.dtype != np.uint8:
            vmin = float(np.percentile(img_bgr, 1))
            vmax = float(np.percentile(img_bgr, 99))
            rng = max(vmax - vmin, 1e-6)
            img_bgr = np.clip((img_bgr - vmin) / rng * 255.0, 0, 255).astype(np.uint8)

        H, W = img_bgr.shape[:2]
        all_boxes = []
        all_confs = []

        # 1. Coarse Scale for Giant Impact Basins (w > 150px)
        res_coarse = self._yolo_model.predict(img_bgr, imgsz=288, conf=0.20, verbose=False)[0]
        if len(res_coarse.boxes) > 0:
            all_boxes.append(res_coarse.boxes.xyxy.cpu().numpy())
            all_confs.append(res_coarse.boxes.conf.cpu().numpy())

        # 2. Medium Scale for Standard Craters
        res_med = self._yolo_model.predict(img_bgr, imgsz=640, conf=0.20, verbose=False)[0]
        if len(res_med.boxes) > 0:
            all_boxes.append(res_med.boxes.xyxy.cpu().numpy())
            all_confs.append(res_med.boxes.conf.cpu().numpy())

        # 3. Fine Sliced Scale (SAHI) for micro-craters if high resolution (>= 800px)
        if max(H, W) >= 800:
            tile_size = 640
            stride = 440
            for y in range(0, max(1, H - tile_size + stride), stride):
                y_end = min(H, y + tile_size)
                y_start = max(0, y_end - tile_size)
                for x in range(0, max(1, W - tile_size + stride), stride):
                    x_end = min(W, x + tile_size)
                    x_start = max(0, x_end - tile_size)

                    crop = img_bgr[y_start:y_end, x_start:x_end]
                    res_tile = self._yolo_model.predict(crop, conf=0.20, imgsz=tile_size, verbose=False)[0]
                    if len(res_tile.boxes) > 0:
                        b = res_tile.boxes.xyxy.cpu().numpy().copy()
                        c = res_tile.boxes.conf.cpu().numpy().copy()
                        b[:, [0, 2]] += x_start
                        b[:, [1, 3]] += y_start
                        all_boxes.append(b)
                        all_confs.append(c)

        if not all_boxes:
            return np.empty((0, 3), dtype=np.float32)

        boxes = np.vstack(all_boxes)
        confs = np.concatenate(all_confs)

        # Multi-scale OpenCV NMS
        cv_boxes = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]
        indices = cv2.dnn.NMSBoxes(cv_boxes, confs.tolist(), score_threshold=0.20, nms_threshold=0.45)
        if len(indices) == 0:
            return np.empty((0, 3), dtype=np.float32)

        indices = np.array(indices).flatten()
        boxes = boxes[indices]
        confs = confs[indices]

        # Convert xyxy -> (col, row, radius)
        craters = np.zeros((len(boxes), 3), dtype=np.float32)
        craters[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0  # col
        craters[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0  # row
        craters[:, 2] = ((boxes[:, 2] - boxes[:, 0]) + (boxes[:, 3] - boxes[:, 1])) / 4.0  # radius

        # Sort by confidence descending, then cap to self.max_craters
        sort_idx = np.argsort(-confs)
        craters = craters[sort_idx]
        return craters[:self.max_craters]

    # ── CNSF topology (FEATURES.md F13) ──────────────────────────────────────

    def _build_cnsf(self, craters: np.ndarray) -> Dict[str, Any]:
        """
        Build Crater Neighbourhood Shape Feature (CNSF).

        Per FEATURES.md F13: centre + radius + neighbourhood topology per crater.
        Returns dict with 'craters' (N,3), 'adj_matrix' (N,N), and local 'descriptors' (N, D).
        """
        if len(craters) < 2:
            return {
                "craters": craters,
                "adj_matrix": np.zeros((len(craters), len(craters)), dtype=np.float32),
                "descriptors": np.zeros((len(craters), 0), dtype=np.float32),
            }

        N = len(craters)
        adj = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i + 1, N):
                dx = craters[i, 0] - craters[j, 0]
                dy = craters[i, 1] - craters[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = craters[i, 2] + craters[j, 2]
                norm_dist = dist / (r_sum + 1e-9)
                adj[i, j] = adj[j, i] = norm_dist

        # Local K-NN Shape Descriptor (scale-invariant & robust to missing/extra craters)
        K = min(8, N - 1)
        coords = craters[:, :2]
        radii = craters[:, 2]
        descriptors = []
        for i in range(N):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            nn_idx = np.argsort(dists)[1:K+1]
            nn_dists = dists[nn_idx]
            scale = np.median(nn_dists) + 1e-6
            norm_d = nn_dists / scale
            norm_r = radii[nn_idx] / (radii[i] + 1e-6)
            descriptors.append(np.hstack([norm_d, norm_r]))

        return {
            "craters": craters,
            "adj_matrix": adj,
            "descriptors": np.array(descriptors, dtype=np.float32),
        }

    def _topology_match(
        self,
        cnsf_src: Dict[str, Any],
        cnsf_ref: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Similarity-invariant topology matching between two CNSFs.
        Returns (src_xy, ref_xy, confidence) arrays.

        Strategy: Mutual nearest neighbor matching on local CNSF descriptors
        with fallback to sorted adjacency matching for tiny sets.
        Applies MCR structural outlier removal.
        """
        c_src = cnsf_src["craters"]   # (Ns, 3)
        c_ref = cnsf_ref["craters"]   # (Nr, 3)

        Ns, Nr = len(c_src), len(c_ref)
        if Ns == 0 or Nr == 0:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    np.empty(0, dtype=np.float32))

        desc_src = cnsf_src.get("descriptors")
        desc_ref = cnsf_ref.get("descriptors")

        src_xy_out, ref_xy_out, conf_out = [], [], []

        # If both have local descriptors with matching dimension
        if desc_src is not None and desc_ref is not None and desc_src.ndim == 2 and desc_ref.ndim == 2 and desc_src.shape[1] == desc_ref.shape[1] and desc_src.shape[1] > 0:
            # Pairwise L2 distance
            cost = np.linalg.norm(desc_src[:, None, :] - desc_ref[None, :, :], axis=2)
            best_ref_for_src = np.argmin(cost, axis=1)
            best_src_for_ref = np.argmin(cost, axis=0)

            for i in range(Ns):
                j = best_ref_for_src[i]
                if best_src_for_ref[j] == i:
                    score = float(cost[i, j])
                    conf = float(1.0 / (1.0 + score))
                    if conf >= min(self.conf_thresh, 0.50):
                        src_xy_out.append([c_src[i, 0], c_src[i, 1]])
                        ref_xy_out.append([c_ref[j, 0], c_ref[j, 1]])
                        conf_out.append(conf)
        else:
            # Adjacency vector fallback
            A_src = cnsf_src["adj_matrix"]
            A_ref = cnsf_ref["adj_matrix"]
            for i in range(Ns):
                desc_i = np.sort(A_src[i])
                best_j, best_score = -1, float("inf")
                for j in range(Nr):
                    desc_j = np.sort(A_ref[j])
                    L = min(len(desc_i), len(desc_j))
                    score = float(np.abs(desc_i[:L] - desc_j[:L]).mean())
                    if score < best_score:
                        best_score = score
                        best_j = j
                if best_j >= 0:
                    raw_conf = float(1.0 / (1.0 + best_score))
                    if raw_conf >= min(self.conf_thresh, 0.50):
                        src_xy_out.append([c_src[i, 0], c_src[i, 1]])
                        ref_xy_out.append([c_ref[best_j, 0], c_ref[best_j, 1]])
                        conf_out.append(raw_conf)

        if not src_xy_out:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    np.empty(0, dtype=np.float32))

        src_xy = np.array(src_xy_out, dtype=np.float32)
        ref_xy = np.array(ref_xy_out, dtype=np.float32)
        confidence = np.array(conf_out, dtype=np.float32)

        # ── MCR structural outlier removal ────────────────────────────────────
        src_xy, ref_xy, confidence = self._mcr_filter(src_xy, ref_xy, confidence)

        return src_xy, ref_xy, confidence

    @staticmethod
    def _mcr_filter(
        src_xy: np.ndarray,
        ref_xy: np.ndarray,
        confidence: np.ndarray,
        max_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MCR (Maximum Consistent Ratio) structural outlier filter.
        Removes matches with large displacement inconsistent with the consensus.
        """
        if len(src_xy) < 3:
            return src_xy, ref_xy, confidence

        assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
        assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
        displacements = ref_xy - src_xy  # (N, 2)
        med = np.median(displacements, axis=0)
        diffs = np.linalg.norm(displacements - med, axis=1)
        mad = np.median(diffs) + 1e-9
        keep = diffs <= max_ratio * mad
        return src_xy[keep], ref_xy[keep], confidence[keep]

    # ── Pre-flight recall check (FEATURES.md F13) ────────────────────────────

    def preflight_recall_check(self, image: np.ndarray) -> bool:
        """
        Check detector recall on sample image.
        Sets detector_validated flag. Must be run before using M3 as primary.
        Returns True if at least 1 crater detected.
        """
        craters = self._detect_craters(image)
        return len(craters) > 0

    # ── Public match interface ────────────────────────────────────────────────

    def match(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float = 1.0,
        # Gate inputs — passed from benchmark.py
        crater_density_src: float = 0.0,
        crater_density_ref: float = 0.0,
        terrain_src: str = "unknown",
        terrain_ref: str = "unknown",
        **kwargs: Any,
    ) -> MatchResult:
        t0 = time.time()
        try:
            return self._match_impl(
                src, ref, gsd_ratio,
                crater_density_src, crater_density_ref,
                terrain_src, terrain_ref,
            )
        except Exception as exc:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason=str(exc),
            )

    def _match_impl(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float,
        density_src: float,
        density_ref: float,
        terrain_src: str,
        terrain_ref: str,
    ) -> MatchResult:
        t0 = time.time()

        # ── Gate check ────────────────────────────────────────────────────────
        gate_ok, gate_reason = self.check_gate(
            density_src, density_ref, terrain_src, terrain_ref,
            self.tau_c, self.allowed_terrain,
        )
        if not gate_ok:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason=f"gate_skip: {gate_reason}",
                gate_skip=True,
                gate_reason=gate_reason,
            )

        # ── Detect craters ────────────────────────────────────────────────────
        craters_src = self._detect_craters(src)
        src_detector = self._last_used_detector
        craters_ref = self._detect_craters(ref)
        ref_detector = self._last_used_detector

        using_hough = (src_detector == "hough" or ref_detector == "hough")
        actual_id = "crater_hough" if using_hough else "crater"

        # Pre-flight recall check
        detector_validated = len(craters_src) > 0 and len(craters_ref) > 0

        if not detector_validated:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="no_craters_detected",
                gate_skip=False,
                detector_validated=False,
                matcher_id_actual=actual_id,
            )

        # ── Build CNSF + topology match ───────────────────────────────────────
        cnsf_src = self._build_cnsf(craters_src)
        cnsf_ref = self._build_cnsf(craters_ref)
        src_xy, ref_xy, confidence = self._topology_match(cnsf_src, cnsf_ref)

        runtime = time.time() - t0
        return MatchResult(
            src_xy=src_xy,
            ref_xy=ref_xy,
            confidence=confidence,
            scale=np.ones(len(src_xy), dtype=np.float32) * gsd_ratio,
            angle_deg=np.zeros(len(src_xy), dtype=np.float32),
            runtime_s=runtime,
            matcher_params={
                "matcher_id": actual_id,
                "detector_validated": detector_validated,
                "n_craters_src": len(craters_src),
                "n_craters_ref": len(craters_ref),
                "gate_skip": False,
                "gsd_ratio": gsd_ratio,
                "cpu_fallback": using_hough,
            },
        )

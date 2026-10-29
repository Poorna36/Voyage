"""
src/matching/sift.py
====================
M0 — SIFT matcher (always-on baseline).

Pipeline: detect -> ANMS SSC -> re-describe -> FLANN + Lowe ratio test.
Runs on 100% of pairs, never crashes, always returns a MatchResult.

Configuration keys (configs/matchers.yaml, sift block):
  num_keypoints : 2048     (ANMS budget)
  ratio_thresh  : 0.75     (Lowe ratio)
  flann_trees   : 5
  flann_checks  : 50
  max_log_scale_deviation : 0.3   (scale-consistency filter threshold)

References: ARCHITECTURE.md §4, FEATURES.md F09, CONFIGURATION.md §2
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMatcher, MatchResult
from ..selection.anms import anms_ssc

# Default config — overridden by configs/matchers.yaml at runtime
_DEFAULTS: Dict[str, Any] = {
    "num_keypoints": 2048,
    "ratio_thresh": 0.75,
    "flann_trees": 5,
    "flann_checks": 50,
    "max_log_scale_deviation": 0.3,
}


class SIFTMatcher(BaseMatcher):
    """
    M0 — SIFT-based correspondence matcher.

    Always runs, even at polar scenes or when other matchers fail.
    Returns count >= 0; inlier_count=0 is a valid outcome.
    """

    matcher_id = "sift"
    requires_gpu = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}
        self.num_keypoints: int = int(cfg["num_keypoints"])
        self.ratio_thresh: float = float(cfg["ratio_thresh"])
        self.flann_trees: int = int(cfg["flann_trees"])
        self.flann_checks: int = int(cfg["flann_checks"])
        self.max_log_scale_dev: float = float(cfg["max_log_scale_deviation"])

    def match(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float = 1.0,
        mask_src: Optional[np.ndarray] = None,
        mask_ref: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> MatchResult:
        """
        Detect, select (ANMS SSC), describe, and match SIFT features.

        Returns MatchResult with (col, row) coordinates and confidence
        derived from the Lowe ratio (1 - ratio for intuitive direction).
        """
        t0 = time.time()
        try:
            return self._match_impl(src, ref, gsd_ratio, mask_src=mask_src, mask_ref=mask_ref)
        except Exception as exc:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason=str(exc),
            )

    # ── Private implementation ────────────────────────────────────────────────

    def _match_impl(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float,
        mask_src: Optional[np.ndarray] = None,
        mask_ref: Optional[np.ndarray] = None,
    ) -> MatchResult:
        import cv2

        t0 = time.time()

        src_gray = self._to_gray(src)
        ref_gray = self._to_gray(ref)

        sift = cv2.SIFT_create(nfeatures=0)   # detect all, ANMS will cull

        # Prepare uint8 masks (255 = valid, 0 = invalid)
        m_src = (mask_src > 0).astype(np.uint8) * 255 if mask_src is not None else None
        m_ref = (mask_ref > 0).astype(np.uint8) * 255 if mask_ref is not None else None

        # ── Detect ────────────────────────────────────────────────────────────
        kp_src = sift.detect(src_gray, m_src)
        kp_ref = sift.detect(ref_gray, m_ref)

        if not kp_src or not kp_ref:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="no_keypoints_detected",
            )

        # ── ANMS SSC — spatial uniformity (FEATURES.md F09) ─────────────────
        kp_src = anms_ssc(kp_src, self.num_keypoints, src_gray.shape[:2])
        kp_ref = anms_ssc(kp_ref, self.num_keypoints, ref_gray.shape[:2])

        # ── Describe ──────────────────────────────────────────────────────────
        kp_src, des_src = sift.compute(src_gray, kp_src)
        kp_ref, des_ref = sift.compute(ref_gray, kp_ref)

        if des_src is None or des_ref is None or len(des_src) < 2 or len(des_ref) < 2:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="descriptor_computation_failed",
            )

        # ── FLANN matching ────────────────────────────────────────────────────
        flann_params = dict(algorithm=1, trees=self.flann_trees)   # FLANN_INDEX_KDTREE=1
        search_params = dict(checks=self.flann_checks)
        flann = cv2.FlannBasedMatcher(flann_params, search_params)

        matches = flann.knnMatch(des_src, des_ref, k=2)

        # ── Lowe ratio test ───────────────────────────────────────────────────
        good_src_xy, good_ref_xy, conf = [], [], []
        scales_src, scales_ref, angles = [], [], []

        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.ratio_thresh * n.distance:
                continue   # fails ratio test
            # Coordinates in (col, row) = (x, y)
            s = kp_src[m.queryIdx]
            r = kp_ref[m.trainIdx]
            good_src_xy.append([s.pt[0], s.pt[1]])
            good_ref_xy.append([r.pt[0], r.pt[1]])
            conf.append(1.0 - m.distance / (n.distance + 1e-9))
            scales_src.append(s.size)
            scales_ref.append(r.size)
            # angle in degrees (orientation difference)
            angles.append(s.angle - r.angle)

        if not good_src_xy:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="lowe_ratio_filtered_all",
            )

        src_xy = np.array(good_src_xy, dtype=np.float32)
        ref_xy = np.array(good_ref_xy, dtype=np.float32)
        confidence = np.clip(np.array(conf, dtype=np.float32), 0.0, 1.0)
        scales_src_a = np.array(scales_src, dtype=np.float32)
        scales_ref_a = np.array(scales_ref, dtype=np.float32)
        angle_arr = np.array(angles, dtype=np.float32)
        scale_ratio = scales_src_a / (scales_ref_a + 1e-9)

        # ── Scale-consistency filter (ARCHITECTURE.md §4) ─────────────────────
        # If gsd_ratio is nominal (1.0), allow up to 3x scale variation (log(3) ~ 1.1)
        tol = self.max_log_scale_dev if abs(gsd_ratio - 1.0) > 1e-3 else max(self.max_log_scale_dev, 1.2)
        keep = self._scale_consistency_mask(
            scales_src_a, scales_ref_a, gsd_ratio, tol
        )
        src_xy = src_xy[keep]
        ref_xy = ref_xy[keep]
        confidence = confidence[keep]
        scale_ratio = scale_ratio[keep]
        angle_arr = angle_arr[keep]

        runtime = time.time() - t0
        return MatchResult(
            src_xy=src_xy,
            ref_xy=ref_xy,
            confidence=confidence,
            scale=scale_ratio,
            angle_deg=angle_arr,
            runtime_s=runtime,
            matcher_params={
                "matcher_id": self.matcher_id,
                "num_keypoints": self.num_keypoints,
                "ratio_thresh": self.ratio_thresh,
                "gsd_ratio": gsd_ratio,
                "n_before_scale_filter": int(keep.shape[0]),
                "n_after_scale_filter": int(keep.sum()),
            },
        )

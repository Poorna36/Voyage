"""
src/matching/lnift.py
=====================
M1b — LNIFT matcher (Light Non-linear Intensity Feature Transform).

Pilot benchmark alongside RIFT2 — same scale-consistency filter applied.
Simpler/faster than RIFT2: 2 scales, 4 orientations, Gabor-based descriptors,
no multi-octave extension. Used purely for comparative benchmarking.

Decision outcome documented in DECISIONS.md D14.

References: ARCHITECTURE.md §4, FEATURES.md F11, CONFIGURATION.md §2
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseMatcher, MatchResult
from ..selection.anms import anms_ssc

_DEFAULTS: Dict[str, Any] = {
    "n_scales": 2,
    "n_orientations": 4,
    "num_keypoints": 2048,
    "max_log_scale_deviation": 0.3,   # same as RIFT2 per FEATURES.md F11
    "match_ratio_thresh": 0.8,
    "patch_size": 64,
    "descriptor_grid": 4,             # 4x4 spatial bins (lighter than RIFT2's 6x6)
}


class LNIFTMatcher(BaseMatcher):
    """
    M1b — LNIFT (Light NIFT) pilot benchmark matcher.

    matcher_id  = 'lnift'
    requires_gpu = False

    Uses Gabor filter responses at 2 scales x 4 orientations to build a
    histogram-of-gradients-style descriptor weighted by filter amplitude.
    Same ANMS SSC and scale-consistency filter as RIFT2.
    """

    matcher_id = "lnift"
    requires_gpu = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}
        self.n_scales: int = int(cfg["n_scales"])
        self.n_orient: int = int(cfg["n_orientations"])
        self.num_kp: int = int(cfg["num_keypoints"])
        self.max_log_dev: float = float(cfg["max_log_scale_deviation"])
        self.ratio_thresh: float = float(cfg["match_ratio_thresh"])
        self.patch_size: int = int(cfg["patch_size"])
        self.grid: int = int(cfg["descriptor_grid"])

    def match(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float = 1.0,
        **kwargs: Any,
    ) -> MatchResult:
        t0 = time.time()
        try:
            return self._match_impl(src, ref, gsd_ratio)
        except Exception as exc:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason=str(exc),
            )

    # ── Gabor filter bank (simpler than RIFT2 log-Gabor) ─────────────────────

    def _gabor_bank(
        self,
        image: np.ndarray,
        n_scales: int,
        n_orientations: int,
    ) -> np.ndarray:
        """
        Apply Gabor filter bank; return amplitude responses.

        Returns (n_scales, n_orient, H, W) float32 amplitude array.
        """
        import cv2

        H, W = image.shape
        responses = np.zeros((n_scales, n_orientations, H, W), dtype=np.float32)

        for s in range(n_scales):
            wavelength = 4.0 * (2.0 ** s)   # 4px, 8px, ...
            for o in range(n_orientations):
                theta = o * np.pi / n_orientations
                kern = cv2.getGaborKernel(
                    ksize=(int(wavelength * 3) | 1, int(wavelength * 3) | 1),
                    sigma=wavelength * 0.5,
                    theta=theta,
                    lambd=wavelength,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F,
                )
                real = cv2.filter2D(image, cv2.CV_32F, kern)
                kern_imag = cv2.getGaborKernel(
                    ksize=(int(wavelength * 3) | 1, int(wavelength * 3) | 1),
                    sigma=wavelength * 0.5,
                    theta=theta,
                    lambd=wavelength,
                    gamma=0.5,
                    psi=np.pi / 2,
                    ktype=cv2.CV_32F,
                )
                imag = cv2.filter2D(image, cv2.CV_32F, kern_imag)
                responses[s, o] = np.sqrt(real**2 + imag**2)

        return responses

    # ── Keypoint detection via Gabor energy ───────────────────────────────────

    def _detect_keypoints(
        self,
        responses: np.ndarray,
        image_shape: tuple,
    ) -> np.ndarray:
        """
        Detect keypoints from Gabor energy maxima.
        Returns (N, 3) array: (col, row, strength) — (x, y) convention.
        """
        from scipy.ndimage import maximum_filter

        energy = responses.max(axis=(0, 1))   # (H, W) — max across s, o
        win = 7
        local_max = maximum_filter(energy, size=win)
        mask = (energy == local_max) & (energy > energy.mean() + energy.std() * 0.5)
        ys, xs = np.where(mask)
        strengths = energy[ys, xs]
        return np.column_stack([xs, ys, strengths]).astype(np.float32)

    # ── LNIFT descriptor ──────────────────────────────────────────────────────

    def _lnift_descriptor(
        self,
        responses: np.ndarray,   # (n_s, n_o, H, W)
        kp_x: float,
        kp_y: float,
    ) -> Optional[np.ndarray]:
        """
        Build descriptor: (grid x grid x n_orient) histogram of Gabor energies.
        """
        n_s, n_o, H, W = responses.shape
        g = self.grid
        half = self.patch_size // 2
        x0, y0 = int(round(kp_x)), int(round(kp_y))

        if x0 - half < 0 or x0 + half >= W or y0 - half < 0 or y0 + half >= H:
            return None

        # Use max across scales per orientation: (n_o, P, P)
        patch = responses[:, :, y0-half:y0+half, x0-half:x0+half].max(axis=0)
        P = patch.shape[1]
        cell_h = max(1, P // g)
        cell_w = max(1, P // g)

        desc = np.zeros((g, g, n_o), dtype=np.float32)
        for gi in range(g):
            for gj in range(g):
                r0, r1 = gi * cell_h, min((gi + 1) * cell_h, P)
                c0, c1 = gj * cell_w, min((gj + 1) * cell_w, P)
                desc[gi, gj, :] = patch[:, r0:r1, c0:c1].mean(axis=(1, 2))

        flat = desc.flatten()
        norm = np.linalg.norm(flat) + 1e-9
        return (flat / norm).astype(np.float32)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def _match_impl(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float,
    ) -> MatchResult:
        import cv2

        t0 = time.time()

        src_gray = self._to_gray(src).astype(np.float32)
        ref_gray = self._to_gray(ref).astype(np.float32)

        resp_src = self._gabor_bank(src_gray, self.n_scales, self.n_orient)
        resp_ref = self._gabor_bank(ref_gray, self.n_scales, self.n_orient)

        kps_src = self._detect_keypoints(resp_src, src_gray.shape)
        kps_ref = self._detect_keypoints(resp_ref, ref_gray.shape)

        if len(kps_src) == 0 or len(kps_ref) == 0:
            return self._empty_result(runtime_s=time.time()-t0, reason="no_keypoints")

        # ANMS SSC — same as RIFT2 per FEATURES.md F11
        kps_src = anms_ssc(kps_src, self.num_kp, src_gray.shape[:2])
        kps_ref = anms_ssc(kps_ref, self.num_kp, ref_gray.shape[:2])
        kps_src = np.array(kps_src, dtype=np.float32)
        kps_ref = np.array(kps_ref, dtype=np.float32)

        # Build descriptors
        descs_src, valid_src = [], []
        for kp in kps_src:
            d = self._lnift_descriptor(resp_src, float(kp[0]), float(kp[1]))
            if d is not None:
                descs_src.append(d)
                valid_src.append(kp)

        descs_ref, valid_ref = [], []
        for kp in kps_ref:
            d = self._lnift_descriptor(resp_ref, float(kp[0]), float(kp[1]))
            if d is not None:
                descs_ref.append(d)
                valid_ref.append(kp)

        if not descs_src or not descs_ref:
            return self._empty_result(runtime_s=time.time()-t0,
                                      reason="descriptor_computation_failed")

        des_s = np.vstack(descs_src).astype(np.float32)
        des_r = np.vstack(descs_ref).astype(np.float32)
        kp_s = np.vstack(valid_src)
        kp_r = np.vstack(valid_ref)

        # Ratio test matching
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des_s, des_r, k=2)

        good_src, good_ref, conf = [], [], []
        scales_s, scales_r = [], []

        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.ratio_thresh * n.distance:
                continue
            sp, rp = kp_s[m.queryIdx], kp_r[m.trainIdx]
            good_src.append([sp[0], sp[1]])   # col, row
            good_ref.append([rp[0], rp[1]])
            conf.append(1.0 - m.distance / (n.distance + 1e-9))
            scales_s.append(sp[2])
            scales_r.append(rp[2])

        if not good_src:
            return self._empty_result(runtime_s=time.time()-t0,
                                      reason="ratio_test_filtered_all")

        src_xy = np.array(good_src, dtype=np.float32)
        ref_xy = np.array(good_ref, dtype=np.float32)
        confidence = np.clip(np.array(conf, dtype=np.float32), 0.0, 1.0)
        sc_s = np.array(scales_s, dtype=np.float32)
        sc_r = np.array(scales_r, dtype=np.float32)

        # Same scale-consistency filter as RIFT2 (FEATURES.md F11)
        keep = self._scale_consistency_mask(sc_s, sc_r, gsd_ratio, self.max_log_dev)
        src_xy = src_xy[keep]
        ref_xy = ref_xy[keep]
        confidence = confidence[keep]
        sc_s = sc_s[keep]
        sc_r = sc_r[keep]

        runtime = time.time() - t0
        return MatchResult(
            src_xy=src_xy,
            ref_xy=ref_xy,
            confidence=confidence,
            scale=sc_s / (sc_r + 1e-9),
            angle_deg=np.zeros(len(src_xy), dtype=np.float32),
            runtime_s=runtime,
            matcher_params={
                "matcher_id": self.matcher_id,
                "n_scales": self.n_scales,
                "n_orientations": self.n_orient,
                "gsd_ratio": gsd_ratio,
                "n_before_scale_filter": int(keep.shape[0]),
                "n_after_scale_filter": int(keep.sum()),
            },
        )

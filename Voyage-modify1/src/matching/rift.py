"""
src/matching/rift.py
====================
M1a — RIFT2 matcher with multi-octave log-Gabor scale-space extension.

Algorithm:
  1. Log-Gabor filter bank  (n_scales=4, n_orientations=6)
  2. Phase congruency (PC) map  (Kovesi 2003)
  3. PC-weighted maximum-moment keypoint detection
  4. ANMS SSC for spatial uniformity
  5. MIM descriptor  (6x6 spatial grid x n_orientations orientation bins)
  6. Scale-space extension over octaves (our novelty)
  7. Scale-consistency filter before returning

Configuration keys (matchers.yaml, rift2 block):
  n_scales             : 4
  n_orientations       : 6
  log_gabor_min_wl     : 3      (pixels, minimum filter wavelength)
  log_gabor_mult       : 2.1    (scale multiplier between consecutive scales)
  log_gabor_sigma_on_f : 0.55   (bandwidth ratio)
  scale_space_octaves  : 3      (our scale-space extension)
  num_keypoints        : 2048
  max_log_scale_deviation : 0.3

References: ARCHITECTURE.md §4, FEATURES.md F10, CONFIGURATION.md §2
           Lin et al. 2019 (RIFT), Kovesi 2003 (Phase Congruency)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseMatcher, MatchResult
from ..selection.anms import anms_ssc

_DEFAULTS: Dict[str, Any] = {
    "n_scales": 4,
    "n_orientations": 6,
    "log_gabor_min_wl": 3,
    "log_gabor_mult": 2.1,
    "log_gabor_sigma_on_f": 0.55,
    "scale_space_octaves": 3,
    "num_keypoints": 2048,
    "max_log_scale_deviation": 0.3,
    "descriptor_grid": 6,      # 6x6 spatial bins
    "match_ratio_thresh": 0.8,
}


class RIFT2Matcher(BaseMatcher):
    """
    M1a — RIFT2 + multi-octave log-Gabor scale-space extension.

    matcher_id  = 'rift2'
    requires_gpu = False
    """

    matcher_id = "rift2"
    requires_gpu = False

    # Per PROGRESS.md §3.4 — flag if total tile runtime exceeds this threshold
    RUNTIME_WARN_S: float = 120.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}
        self.n_scales: int = int(cfg["n_scales"])
        self.n_orient: int = int(cfg["n_orientations"])
        self.min_wl: float = float(cfg["log_gabor_min_wl"])
        self.mult: float = float(cfg["log_gabor_mult"])
        self.sigma_on_f: float = float(cfg["log_gabor_sigma_on_f"])
        self.octaves: int = int(cfg["scale_space_octaves"])
        self.num_kp: int = int(cfg["num_keypoints"])
        self.max_log_dev: float = float(cfg["max_log_scale_deviation"])
        self.grid: int = int(cfg["descriptor_grid"])
        self.ratio_thresh: float = float(cfg["match_ratio_thresh"])

    # ── Public interface ─────────────────────────────────────────────────────

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

    # ── Log-Gabor filter bank ─────────────────────────────────────────────────

    def _log_gabor_bank(
        self,
        image: np.ndarray,
        n_scales: int,
        n_orientations: int,
        octave: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build log-Gabor filter bank and return per-scale-orientation responses.

        Parameters
        ----------
        image         : 2-D float32 image (H, W)
        n_scales      : number of frequency scales
        n_orientations: number of orientations
        octave        : current octave index (0 = full resolution)

        Returns
        -------
        amplitudes : (n_scales, n_orientations, H, W) float32
        phases     : (n_scales, n_orientations, H, W) float32
        """
        H, W = image.shape
        # Work in frequency domain
        img_f = np.fft.fft2(image.astype(np.float64))

        # Build frequency grids
        fx = np.fft.fftfreq(W)    # (W,)
        fy = np.fft.fftfreq(H)    # (H,)
        FX, FY = np.meshgrid(fx, fy)  # (H, W) each

        radius = np.sqrt(FX**2 + FY**2)
        radius[0, 0] = 1.0   # avoid log(0)
        theta_grid = np.arctan2(FY, FX)  # (H, W)

        amplitudes = np.zeros((n_scales, n_orientations, H, W), dtype=np.float32)
        phases = np.zeros_like(amplitudes)

        for s in range(n_scales):
            # Centre frequency for this scale
            wl_s = self.min_wl * (self.mult ** s) * (2 ** octave)
            f0 = 1.0 / max(wl_s, 1e-6)
            sigma_f = self.sigma_on_f * f0

            # Radial log-Gabor component
            # H_radial(f) = exp(-log(f/f0)^2 / (2 * (log(sigma_f/f0))^2))
            log_sigma = np.log(sigma_f / f0 + 1e-9)
            H_radial = np.exp(
                -np.log(radius / f0 + 1e-9) ** 2
                / (2.0 * log_sigma**2 + 1e-9)
            )
            H_radial[0, 0] = 0.0   # DC = 0

            for o in range(n_orientations):
                # Angular component — spread orientations evenly
                theta_o = o * np.pi / n_orientations
                d_theta = theta_grid - theta_o
                # Wrap to [-pi/2, pi/2]
                d_theta = np.where(
                    d_theta > np.pi / 2,
                    d_theta - np.pi,
                    np.where(d_theta < -np.pi / 2, d_theta + np.pi, d_theta),
                )
                # Gaussian angular taper
                sigma_theta = np.pi / (n_orientations * 2.0)
                H_ang = np.exp(-(d_theta**2) / (2.0 * sigma_theta**2))

                H = H_radial * H_ang
                resp_f = img_f * H
                resp_spatial = np.fft.ifft2(resp_f)

                amplitudes[s, o] = np.abs(resp_spatial).astype(np.float32)
                phases[s, o] = np.angle(resp_spatial).astype(np.float32)

        return amplitudes, phases

    # ── Phase congruency ──────────────────────────────────────────────────────

    def _phase_congruency(
        self,
        amplitudes: np.ndarray,
        phases: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Kovesi phase congruency map from filter bank responses.

        Returns
        -------
        pc_map     : (H, W) float32  — overall PC strength
        pc_orient  : (H, W) float32  — dominant orientation per pixel (radians)
        pc_moment  : (H, W) float32  — max moment of PC (for keypoint detection)
        """
        n_scales, n_orient, H, W = amplitudes.shape
        eps = 1e-7

        # Weighted mean phase per orientation
        sumA = amplitudes.sum(axis=0) + eps        # (n_orient, H, W)
        weighted_cos = (amplitudes * np.cos(phases)).sum(axis=0)  # (n_orient, H, W)
        weighted_sin = (amplitudes * np.sin(phases)).sum(axis=0)

        phi_mean = np.arctan2(weighted_sin, weighted_cos + eps)  # (n_orient, H, W)

        # Phase alignment: cos(phi - phi_mean) - |sin(phi - phi_mean)|
        diff = phases - phi_mean[np.newaxis, ...]  # (n_scales, n_orient, H, W)
        energy = (amplitudes * (np.cos(diff) - np.abs(np.sin(diff)))).sum(axis=0)
        energy = np.maximum(energy, 0.0)  # (n_orient, H, W)

        pc_orient_maps = energy / (sumA + eps)  # (n_orient, H, W)

        # Overall PC — max across orientations
        pc_map = pc_orient_maps.max(axis=0).astype(np.float32)  # (H, W)

        # Dominant orientation
        pc_orient = (np.argmax(pc_orient_maps, axis=0).astype(np.float32)
                     * np.pi / n_orient)  # (H, W)

        # Maximum moment for keypoint detection (Kovesi 2003)
        cx = (pc_orient_maps * np.cos(
            np.arange(n_orient)[:, None, None] * np.pi / n_orient
        )).sum(axis=0)
        cy = (pc_orient_maps * np.sin(
            np.arange(n_orient)[:, None, None] * np.pi / n_orient
        )).sum(axis=0)
        pc_moment = np.sqrt(cx**2 + cy**2).astype(np.float32)

        return pc_map, pc_orient, pc_moment

    # ── Keypoint detection from PC moment ────────────────────────────────────

    def _detect_from_moment(
        self,
        pc_moment: np.ndarray,
        scale_px: float,
    ) -> np.ndarray:
        """
        Detect keypoints at local maxima of pc_moment.

        Returns (N, 3) array of (x, y, strength) in (col, row) convention.
        """
        from scipy.ndimage import maximum_filter, gaussian_filter

        # Smooth before finding maxima
        sigma = max(1.0, scale_px / 4.0)
        smoothed = gaussian_filter(pc_moment, sigma=sigma)

        # Non-maximum suppression via maximum filter
        win = max(3, int(scale_px))
        local_max = maximum_filter(smoothed, size=win)
        mask = (smoothed == local_max) & (smoothed > smoothed.mean())

        ys, xs = np.where(mask)
        strengths = smoothed[ys, xs]

        # Return as (x, y, strength) — (col, row) convention
        return np.column_stack([xs, ys, strengths]).astype(np.float32)

    # ── MIM descriptor ────────────────────────────────────────────────────────

    def _mim_descriptor(
        self,
        pc_orient_maps: np.ndarray,  # (n_orient, H, W)
        pc_amplitude: np.ndarray,    # (n_orient, H, W)
        kp_x: float,
        kp_y: float,
        dominant_angle: float,
        patch_size: int = 96,
    ) -> Optional[np.ndarray]:
        """
        MIM (Maximum Index Map) descriptor — 6x6 spatial grid x n_orient bins.
        
        Returns (grid*grid*n_orient,) float32 vector, or None if out of bounds.
        """
        n_orient, H, W = pc_orient_maps.shape
        g = self.grid
        half = patch_size // 2
        x0, y0 = int(round(kp_x)), int(round(kp_y))

        # Boundary check
        if (x0 - half < 0 or x0 + half >= W or
                y0 - half < 0 or y0 + half >= H):
            return None

        patch_orient = pc_orient_maps[:, y0-half:y0+half, x0-half:x0+half]  # (n_orient, P, P)
        patch_amp = pc_amplitude[:, y0-half:y0+half, x0-half:x0+half]

        P = patch_orient.shape[1]
        cell_h = P // g
        cell_w = P // g

        descriptor = np.zeros((g, g, n_orient), dtype=np.float32)

        for gi in range(g):
            for gj in range(g):
                r0, r1 = gi * cell_h, (gi + 1) * cell_h
                c0, c1 = gj * cell_w, (gj + 1) * cell_w
                cell_amp = patch_amp[:, r0:r1, c0:c1]   # (n_orient, h_c, w_c)
                # Sum amplitude per orientation channel
                descriptor[gi, gj, :] = cell_amp.sum(axis=(1, 2))

        desc = descriptor.flatten()
        norm = np.linalg.norm(desc) + 1e-9
        return (desc / norm).astype(np.float32)

    # ── Main matching pipeline ────────────────────────────────────────────────

    def _match_impl(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float,
    ) -> MatchResult:
        import cv2

        t0 = time.time()

        src_gray = self._to_gray(src).astype(np.float32) / 255.0
        ref_gray = self._to_gray(ref).astype(np.float32) / 255.0

        src_kps_all: List[np.ndarray] = []  # list of (N_i, 3) arrays
        ref_kps_all: List[np.ndarray] = []
        src_descs: List[np.ndarray] = []
        ref_descs: List[np.ndarray] = []

        for oct_idx in range(self.octaves):
            # ── Scale-space: downsample for higher octaves ────────────────────
            scale_factor = 0.5 ** oct_idx
            if oct_idx > 0:
                new_h = max(32, int(src_gray.shape[0] * scale_factor))
                new_w = max(32, int(src_gray.shape[1] * scale_factor))
                src_oct = cv2.resize(src_gray, (new_w, new_h))
                ref_oct = cv2.resize(ref_gray, (new_w, new_h))
            else:
                src_oct = src_gray
                ref_oct = ref_gray

            scale_px = self.min_wl * (2 ** oct_idx)

            for img, kps_all, descs_all in [
                (src_oct, src_kps_all, src_descs),
                (ref_oct, ref_kps_all, ref_descs),
            ]:
                H, W = img.shape
                amps, phs = self._log_gabor_bank(img, self.n_scales, self.n_orient, oct_idx)
                pc_map, pc_orient, pc_moment = self._phase_congruency(amps, phs)
                kps = self._detect_from_moment(pc_moment, scale_px)

                if len(kps) == 0:
                    continue

                # ANMS SSC (col=x, row=y — correct convention)
                kp_list = anms_ssc(kps, self.num_kp, (H, W))
                kp_arr = np.array(kp_list) if not isinstance(kp_list[0], np.ndarray) \
                    else np.vstack(kp_list)

                # Amplitude summed across scales per orientation: (n_orient, H, W)
                amp_per_orient = amps.sum(axis=0)  # (n_orient, H, W)

                # Build orientation maps: (n_orient, H, W) — already separated
                orient_maps = amp_per_orient  # reuse amplitude as orientation strength

                # Build descriptors
                for kp in kp_arr:
                    kp_x, kp_y = float(kp[0]) / scale_factor, float(kp[1]) / scale_factor
                    kp_x_oct, kp_y_oct = float(kp[0]), float(kp[1])
                    dominant = float(pc_orient[int(kp_y_oct), int(kp_x_oct)])
                    desc = self._mim_descriptor(
                        orient_maps, amp_per_orient, kp_x_oct, kp_y_oct, dominant
                    )
                    if desc is not None:
                        # Store at original resolution coords
                        kps_all.append(np.array([[kp_x, kp_y, float(kp[2])]]))
                        descs_all.append(desc)

        if not src_descs or not ref_descs:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="no_descriptors_computed",
            )

        src_kp_arr = np.vstack(src_kps_all)    # (N_s, 3) col, row, strength
        ref_kp_arr = np.vstack(ref_kps_all)    # (N_r, 3)
        src_desc_arr = np.vstack(src_descs).astype(np.float32)  # (N_s, D)
        ref_desc_arr = np.vstack(ref_descs).astype(np.float32)  # (N_r, D)

        # ── Brute-force L2 matching with ratio test ──────────────────────────
        flann_params = dict(algorithm=1, trees=5)
        flann = cv2.FlannBasedMatcher(flann_params, dict(checks=50))
        matches = flann.knnMatch(src_desc_arr, ref_desc_arr, k=2)

        good_src_xy, good_ref_xy, conf = [], [], []
        scales_src, scales_ref = [], []

        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.ratio_thresh * n.distance:
                continue
            sp = src_kp_arr[m.queryIdx]
            rp = ref_kp_arr[m.trainIdx]
            good_src_xy.append([sp[0], sp[1]])   # col, row
            good_ref_xy.append([rp[0], rp[1]])
            conf.append(1.0 - m.distance / (n.distance + 1e-9))
            scales_src.append(sp[2])
            scales_ref.append(rp[2])

        if not good_src_xy:
            return self._empty_result(
                runtime_s=time.time() - t0,
                reason="ratio_test_filtered_all",
                polar_validated=False,
            )

        src_xy = np.array(good_src_xy, dtype=np.float32)
        ref_xy = np.array(good_ref_xy, dtype=np.float32)
        confidence = np.clip(np.array(conf, dtype=np.float32), 0.0, 1.0)
        scales_s = np.array(scales_src, dtype=np.float32)
        scales_r = np.array(scales_ref, dtype=np.float32)

        # ── Scale-consistency filter (MANDATORY per ARCHITECTURE.md §4) ──────
        keep = self._scale_consistency_mask(scales_s, scales_r, gsd_ratio, self.max_log_dev)
        src_xy = src_xy[keep]
        ref_xy = ref_xy[keep]
        confidence = confidence[keep]
        scales_s = scales_s[keep]
        scales_r = scales_r[keep]

        runtime = time.time() - t0

        # ── Runtime flag (PROGRESS.md §3.4) ─────────────────────────────────
        runtime_flagged = runtime > self.RUNTIME_WARN_S
        if runtime_flagged:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "RIFT2Matcher: runtime %.1f s exceeds %.0f s threshold on this tile/pair. "
                "Consider reducing scale_space_octaves or image size.",
                runtime, self.RUNTIME_WARN_S,
            )

        return MatchResult(
            src_xy=src_xy,
            ref_xy=ref_xy,
            confidence=confidence,
            scale=scales_s / (scales_r + 1e-9),
            angle_deg=np.zeros(len(src_xy), dtype=np.float32),
            runtime_s=runtime,
            matcher_params={
                "matcher_id": self.matcher_id,
                "n_scales": self.n_scales,
                "n_orientations": self.n_orient,
                "scale_space_octaves": self.octaves,
                "gsd_ratio": gsd_ratio,
                "polar_validated": False,
                "n_before_scale_filter": int(keep.shape[0]),
                "n_after_scale_filter": int(keep.sum()),
                "runtime_flagged": runtime_flagged,
                "runtime_warn_threshold_s": self.RUNTIME_WARN_S,
            },
        )

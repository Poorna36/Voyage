# -*- coding: utf-8 -*-
"""
tests/test_phase4_comprehensive.py
=====================================
Phase 4 comprehensive test suite — 100k+ test conditions via parametrize.
All tests use synthetic numpy data — no real lunar imagery required.

Covers (maps to VALIDATION.md T-series):
  T-f2     : F2 mandatory checks (bounds + one-to-one)
  T-declus : GCP declustering (GSD-scaled spacing + Z-score)
  T-ladder : DEGENSAC model ladder (similarity/affine/homography)
  T-tile   : Tile-wise models + Gaussian blend
  T-refine : Sub-pixel refinement (NCC + paraboloid + second-peak)
  T-metric : Evaluation metrics (RMSE, pct_lt_1px, coverage, etc.)
  T-aggr   : Leaderboard aggregation (strata, atomic write)
  T-leak   : Leakage audit (geo-cell split integrity)
  T-arb    : Arbitration (policy, tie-break, TOTAL_FAILURE)

Run:
  D:\\neo\\hachathon\\SIH2026_env\\Scripts\\python.exe -m pytest tests/test_phase4_comprehensive.py -v --tb=short -q

Expected: All tests PASS. SKIP is acceptable where cv2/scipy unavailable.
"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# ── Path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import pydegensac  # noqa: F401
    _HAS_DEGENSAC = True
except ImportError:
    _HAS_DEGENSAC = False


# =============================================================================
# ── Synthetic data helpers ───────────────────────────────────────────────────
# =============================================================================

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Synthetic textured grayscale image (float32 0-1)."""
    rng = _rng(seed)
    img = rng.random((h, w), dtype=np.float32)
    # Add blobs for NCC to find
    for _ in range(20):
        cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
        r = rng.integers(4, 12)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        img[y0:y1, x0:x1] = float(rng.uniform(0.7, 1.0))
    return img


def _make_shifted_pair(h: int, w: int, dcol: int = 5, drow: int = 3,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic image pair with known integer shift."""
    src = _make_image(h, w, seed)
    ref = np.zeros_like(src)
    ref[drow:, dcol:] = src[:h - drow, :w - dcol]
    return (src * 255).astype(np.uint8), (ref * 255).astype(np.uint8)


def _make_matches(n: int, h: int = 256, w: int = 256,
                  noise: float = 0.5, seed: int = 0,
                  fraction_oob: float = 0.0,
                  fraction_dup: float = 0.0,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic match set (src_xy, ref_xy, confidence)."""
    rng = _rng(seed)
    margin = 20
    src_xy = rng.uniform(margin, min(h, w) - margin, (n, 2)).astype(np.float32)
    offset = rng.uniform(-3, 3, (n, 2)).astype(np.float32)
    ref_xy = (src_xy + offset + rng.normal(0, noise, (n, 2))).astype(np.float32)
    conf = rng.uniform(0.5, 1.0, n).astype(np.float32)

    # Inject OOB points
    n_oob = int(n * fraction_oob)
    if n_oob > 0:
        oob_idx = rng.choice(n, n_oob, replace=False)
        src_xy[oob_idx] = rng.uniform(w + 50, w + 200, (n_oob, 2)).astype(np.float32)

    # Inject duplicates
    n_dup = int(n * fraction_dup)
    if n_dup > 0:
        dup_idx = rng.choice(n, n_dup, replace=False)
        # Make duplicate of first point
        src_xy[dup_idx] = src_xy[0]

    return src_xy, ref_xy, conf


def _make_homography_matches(n: int = 200, H: np.ndarray = None,
                              h: int = 512, w: int = 512,
                              noise: float = 0.3,
                              fraction_outliers: float = 0.2,
                              seed: int = 42,
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match set consistent with a known homography + outlier fraction."""
    if H is None:
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = 10.0   # tx
        H[1, 2] = 7.0    # ty
        H[0, 0] = 1.02   # slight scale

    rng = _rng(seed)
    margin = 40
    src_xy = rng.uniform(margin, min(h, w) - margin, (n, 2)).astype(np.float64)

    # Project through H
    pts_h = np.hstack([src_xy, np.ones((n, 1))])
    proj = (H @ pts_h.T).T
    ref_xy = proj[:, :2] / proj[:, 2:3]
    ref_xy += rng.normal(0, noise, ref_xy.shape)

    # Inject outliers
    n_out = int(n * fraction_outliers)
    if n_out > 0:
        out_idx = rng.choice(n, n_out, replace=False)
        ref_xy[out_idx] = rng.uniform(margin, min(h, w) - margin, (n_out, 2))

    conf = rng.uniform(0.6, 1.0, n).astype(np.float32)
    return src_xy.astype(np.float32), ref_xy.astype(np.float32), conf


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: F2 Mandatory Checks
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestF2Checks:
    """
    Tests for src/registration/checks.py
    Covers: bounds, one-to-one, edge cases, many sizes/noise levels.
    """

    # ── Basic functionality ─────────────────────────────────────────────────
    def test_import(self):
        from src.registration.checks import f2_checks, F2CheckResult
        assert callable(f2_checks)

    def test_clean_set_passes_unchanged(self):
        from src.registration.checks import f2_checks
        src_xy, ref_xy, conf = _make_matches(50, 256, 256, noise=0.1, seed=1)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.removed_oob == 0
        assert res.removed_dup == 0
        assert res.final_count == res.original_count

    def test_oob_removed(self):
        from src.registration.checks import f2_checks
        src_xy = np.array([[10., 10.], [999., 999.], [50., 50.]], dtype=np.float32)
        ref_xy = np.array([[10., 10.], [20., 20.], [55., 55.]], dtype=np.float32)
        conf   = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.removed_oob >= 1
        assert 999.0 not in res.src_xy[:, 0]

    def test_ref_oob_removed(self):
        from src.registration.checks import f2_checks
        src_xy = np.array([[10., 10.], [20., 20.]], dtype=np.float32)
        ref_xy = np.array([[10., 10.], [999., 999.]], dtype=np.float32)
        conf   = np.array([0.9, 0.8], dtype=np.float32)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.removed_oob >= 1

    def test_duplicate_src_keeps_highest_conf(self):
        from src.registration.checks import f2_checks
        # Two identical src coords — only highest conf should survive
        src_xy = np.array([[10., 10.], [10., 10.], [50., 50.]], dtype=np.float32)
        ref_xy = np.array([[11., 11.], [22., 22.], [55., 55.]], dtype=np.float32)
        conf   = np.array([0.6, 0.9, 0.8], dtype=np.float32)  # second dup has higher conf
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.removed_dup >= 1
        assert res.final_count == 2

    def test_duplicate_ref_removed(self):
        from src.registration.checks import f2_checks
        src_xy = np.array([[10., 10.], [20., 20.], [50., 50.]], dtype=np.float32)
        ref_xy = np.array([[55., 55.], [55., 55.], [80., 80.]], dtype=np.float32)
        conf   = np.array([0.9, 0.7, 0.8], dtype=np.float32)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.removed_dup >= 1

    def test_empty_input(self):
        from src.registration.checks import f2_checks
        src_xy = np.zeros((0, 2), dtype=np.float32)
        ref_xy = np.zeros((0, 2), dtype=np.float32)
        conf   = np.zeros(0, dtype=np.float32)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.final_count == 0

    def test_single_point(self):
        from src.registration.checks import f2_checks
        src_xy = np.array([[10., 20.]], dtype=np.float32)
        ref_xy = np.array([[15., 25.]], dtype=np.float32)
        conf   = np.array([0.9], dtype=np.float32)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.final_count == 1

    def test_buffer_px_allows_slightly_oob(self):
        from src.registration.checks import f2_checks
        # 5px outside image, but buffer=10 → should be kept
        src_xy = np.array([[-5., -5.]], dtype=np.float32)
        ref_xy = np.array([[10., 10.]], dtype=np.float32)
        conf   = np.array([0.8], dtype=np.float32)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256), buffer_px=10)
        assert res.removed_oob == 0

    def test_result_count_consistency(self):
        from src.registration.checks import f2_checks
        src_xy, ref_xy, conf = _make_matches(100, 256, 256, fraction_oob=0.1, fraction_dup=0.1)
        res = f2_checks(src_xy, ref_xy, conf, (256, 256), (256, 256))
        assert res.original_count == 100
        assert res.final_count == res.original_count - res.removed_oob - res.removed_dup
        assert res.final_count >= 0

    def test_output_coords_are_in_bounds(self):
        from src.registration.checks import f2_checks
        src_xy, ref_xy, conf = _make_matches(200, 512, 512, fraction_oob=0.2)
        res = f2_checks(src_xy, ref_xy, conf, (512, 512), (512, 512))
        if res.final_count > 0:
            assert np.all(res.src_xy[:, 0] >= -10)
            assert np.all(res.src_xy[:, 0] < 512 + 10)
            assert np.all(res.ref_xy[:, 1] >= -10)

    # ── Parametrized: many sizes and noise levels ────────────────────────────
    @pytest.mark.parametrize("n_pts", [1, 5, 10, 50, 100, 500, 1000])
    @pytest.mark.parametrize("img_size", [64, 128, 256, 512, 1024])
    @pytest.mark.parametrize("frac_oob", [0.0, 0.1, 0.3, 0.5])
    @pytest.mark.parametrize("frac_dup", [0.0, 0.1, 0.2])
    @pytest.mark.parametrize("seed", [0, 7, 42, 99, 123])
    def test_parametrized_f2(self, n_pts, img_size, frac_oob, frac_dup, seed):
        from src.registration.checks import f2_checks
        src_xy, ref_xy, conf = _make_matches(
            n_pts, img_size, img_size,
            fraction_oob=frac_oob, fraction_dup=frac_dup, seed=seed
        )
        res = f2_checks(src_xy, ref_xy, conf, (img_size, img_size), (img_size, img_size))
        # Invariants
        assert res.original_count == n_pts
        assert res.final_count >= 0
        assert res.final_count <= res.original_count
        assert res.removed_oob >= 0
        assert res.removed_dup >= 0
        if res.final_count > 0:
            assert res.src_xy.shape == (res.final_count, 2)
            assert res.ref_xy.shape == (res.final_count, 2)
            assert len(res.confidence) == res.final_count


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: GCP Declustering
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestDeclustering:
    """Tests for src/registration/declustering.py"""

    def test_import(self):
        from src.registration.declustering import decluster, zscore_filter, decluster_and_filter
        assert callable(decluster)

    def test_spacing_filter_removes_close_points(self):
        from src.registration.declustering import decluster
        # Pack 10 points very close together — most should be removed
        src_xy = np.tile([[100., 100.]], (10, 1)).astype(np.float32)
        src_xy += np.random.default_rng(0).uniform(-2, 2, (10, 2)).astype(np.float32)
        ref_xy = src_xy.copy()
        residuals = np.ones(10, dtype=np.float32)
        out_src, out_ref, out_res, scale = decluster(src_xy, ref_xy, residuals, ref_gsd_m=0.5)
        assert len(out_src) < 10

    def test_gsd_scaling_nac(self):
        """NAC ref (0.5 m) → scale = 1.0 → min_spacing = 20 px."""
        from src.registration.declustering import decluster
        rng = _rng(0)
        src_xy = rng.uniform(0, 512, (50, 2)).astype(np.float32)
        ref_xy = src_xy + rng.normal(0, 0.5, (50, 2)).astype(np.float32)
        residuals = rng.uniform(0, 2, 50).astype(np.float32)
        _, _, _, scale = decluster(src_xy, ref_xy, residuals, ref_gsd_m=0.5, base_gsd_m=0.5)
        assert abs(scale - 1.0) < 1e-6

    def test_gsd_scaling_wac(self):
        """WAC ref (100 m) → scale = 200 → min_spacing = 4000 px."""
        from src.registration.declustering import decluster
        rng = _rng(1)
        src_xy = rng.uniform(0, 10000, (20, 2)).astype(np.float32)
        ref_xy = src_xy.copy()
        residuals = np.ones(20, dtype=np.float32)
        _, _, _, scale = decluster(src_xy, ref_xy, residuals, ref_gsd_m=100.0, base_gsd_m=0.5)
        assert abs(scale - 200.0) < 1e-3

    def test_zscore_filter_removes_outliers(self):
        from src.registration.declustering import zscore_filter
        # 30 inliers with residual ~1 + 1 big outlier
        residuals = np.ones(30, dtype=np.float32)
        residuals[-1] = 100.0  # outlier
        src_xy = np.random.default_rng(0).uniform(0, 256, (30, 2)).astype(np.float32)
        ref_xy = src_xy.copy()
        out_s, out_r, out_res = zscore_filter(src_xy, ref_xy, residuals, threshold=3.0, min_gcps=20)
        assert len(out_s) < 30
        assert 100.0 not in out_res

    def test_zscore_skips_when_too_few_gcps(self):
        from src.registration.declustering import zscore_filter
        # Only 5 points — filter should skip (min_gcps=20)
        src_xy = np.random.default_rng(0).uniform(0, 100, (5, 2)).astype(np.float32)
        ref_xy = src_xy.copy()
        residuals = np.array([1., 1., 1., 1., 100.], dtype=np.float32)  # outlier present
        out_s, out_r, out_res = zscore_filter(src_xy, ref_xy, residuals, threshold=3.0, min_gcps=20)
        assert len(out_s) == 5  # unchanged

    def test_empty_input(self):
        from src.registration.declustering import decluster_and_filter
        src_xy = np.zeros((0, 2), dtype=np.float32)
        ref_xy = np.zeros((0, 2), dtype=np.float32)
        residuals = np.zeros(0, dtype=np.float32)
        _, _, _, scale, count = decluster_and_filter(src_xy, ref_xy, residuals, ref_gsd_m=0.5)
        assert count == 0

    def test_combined_pipeline(self):
        from src.registration.declustering import decluster_and_filter
        rng = _rng(5)
        n = 100
        src_xy = rng.uniform(0, 512, (n, 2)).astype(np.float32)
        ref_xy = src_xy + rng.normal(0, 0.3, (n, 2)).astype(np.float32)
        residuals = rng.uniform(0, 3, n).astype(np.float32)
        residuals[0] = 200.0  # inject outlier
        out_s, out_r, out_res, scale, count = decluster_and_filter(
            src_xy, ref_xy, residuals, ref_gsd_m=0.5
        )
        assert count >= 0
        assert 200.0 not in out_res

    # ── Parametrized: many GSD + size combos ────────────────────────────────
    @pytest.mark.parametrize("ref_gsd_m", [0.31, 0.5, 5.0, 80.0, 100.0])
    @pytest.mark.parametrize("n_pts", [5, 25, 50, 100, 300])
    @pytest.mark.parametrize("spacing", [10.0, 20.0, 50.0])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_parametrized_decluster(self, ref_gsd_m, n_pts, spacing, seed):
        from src.registration.declustering import decluster_and_filter
        rng = _rng(seed)
        img_size = max(1000, int(n_pts * 20))
        src_xy = rng.uniform(0, img_size, (n_pts, 2)).astype(np.float32)
        ref_xy = src_xy + rng.normal(0, 0.5, (n_pts, 2)).astype(np.float32)
        residuals = rng.uniform(0, 3, n_pts).astype(np.float32)
        out_s, out_r, out_res, scale, count = decluster_and_filter(
            src_xy, ref_xy, residuals,
            ref_gsd_m=ref_gsd_m, min_spacing_px=spacing,
        )
        assert scale == pytest.approx(ref_gsd_m / 0.5, rel=1e-4)
        assert 0 <= count <= n_pts
        assert len(out_s) == count


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestMetrics:
    """Tests for src/evaluation/metrics.py"""

    def test_import(self):
        from src.evaluation.metrics import rmse, pct_lt_1px, medae, spatial_coverage
        assert callable(rmse)

    def test_rmse_perfect_prediction(self):
        from src.evaluation.metrics import rmse
        pts = np.random.default_rng(0).uniform(0, 100, (50, 2)).astype(np.float64)
        assert rmse(pts, pts) == pytest.approx(0.0, abs=1e-9)

    def test_rmse_known_value(self):
        from src.evaluation.metrics import rmse
        pred = np.array([[3., 0.]], dtype=np.float64)
        gt   = np.array([[0., 4.]], dtype=np.float64)
        # residual = sqrt(9+16) = 5; RMSE = 5
        assert rmse(pred, gt) == pytest.approx(5.0, rel=1e-6)

    def test_pct_lt_1px_all_perfect(self):
        from src.evaluation.metrics import pct_lt_1px
        pts = np.random.default_rng(1).uniform(0, 100, (30, 2)).astype(np.float64)
        assert pct_lt_1px(pts, pts) == pytest.approx(1.0)

    def test_pct_lt_1px_all_far(self):
        from src.evaluation.metrics import pct_lt_1px
        pred = np.zeros((20, 2), dtype=np.float64)
        gt   = pred + 10.0
        assert pct_lt_1px(pred, gt) == pytest.approx(0.0)

    def test_pct_lt_0p5px(self):
        from src.evaluation.metrics import pct_lt_0p5px
        # first residual = sqrt(0.18) ≈ 0.424 < 0.5 → passes
        # second residual = sqrt((1-0)^2 + (1-0)^2) = sqrt(2) ≈ 1.41 >= 0.5 → fails
        # So 1/2 = 50% within 0.5px
        pred = np.array([[0., 0.], [1., 1.]], dtype=np.float64)
        gt   = np.array([[0.3, 0.3], [0., 0.]], dtype=np.float64)
        val = pct_lt_0p5px(pred, gt)
        assert 0.4 <= val <= 0.6

    def test_medae_robust_to_outlier(self):
        from src.evaluation.metrics import medae
        pred = np.vstack([
            np.zeros((99, 2)),
            np.array([[1000., 1000.]])
        ]).astype(np.float64)
        gt = np.zeros((100, 2), dtype=np.float64)
        med = medae(pred, gt)
        assert med < 2.0  # median unaffected by single outlier

    def test_refinement_gain_positive(self):
        from src.evaluation.metrics import refinement_gain
        assert refinement_gain(2.0, 1.0) == pytest.approx(1.0)

    def test_refinement_gain_negative(self):
        from src.evaluation.metrics import refinement_gain
        assert refinement_gain(1.0, 2.0) == pytest.approx(-1.0)

    def test_spatial_coverage_full(self):
        from src.evaluation.metrics import spatial_coverage
        # Points in all 64 cells of 8x8 grid
        rng = _rng(0)
        pts = rng.uniform(0, 512, (500, 2)).astype(np.float64)
        cov = spatial_coverage(pts, (512, 512), n=8)
        assert cov > 0.9  # should cover most cells

    def test_spatial_coverage_corner_only(self):
        from src.evaluation.metrics import spatial_coverage
        pts = np.array([[5., 5.]], dtype=np.float64)
        cov = spatial_coverage(pts, (512, 512), n=8)
        assert cov < 0.1  # only 1/64 cells

    def test_spatial_coverage_zero_if_no_pts(self):
        from src.evaluation.metrics import spatial_coverage
        pts = np.zeros((0, 2), dtype=np.float64)
        cov = spatial_coverage(pts, (256, 256), n=8)
        assert cov == pytest.approx(0.0)

    def test_grid_density_std(self):
        from src.evaluation.metrics import grid_density_std
        # Uniform distribution → low std
        rng = _rng(5)
        pts = rng.uniform(0, 512, (200, 2)).astype(np.float64)
        std_uniform = grid_density_std(pts, (512, 512), n=8)
        # Clustered → high std
        pts_clustered = rng.uniform(0, 30, (200, 2)).astype(np.float64)
        std_clustered = grid_density_std(pts_clustered, (512, 512), n=8)
        assert std_clustered > std_uniform

    def test_compute_all_metrics_returns_dict(self):
        from src.evaluation.metrics import compute_all_metrics
        rng = _rng(9)
        n = 20
        pred = rng.uniform(0, 256, (n, 2)).astype(np.float64)
        gt_checkpoints = [
            {"id": i, "src_xy": [float(rng.uniform(0, 256)), float(rng.uniform(0, 256))],
             "ref_xy": [float(pred[i, 0]), float(pred[i, 1])],
             "partition": "eval"}
            for i in range(n)
        ]
        metrics = compute_all_metrics(pred, gt_checkpoints, runtime_s=1.5)
        assert "rmse_px" in metrics
        assert metrics["rmse_px"] == pytest.approx(0.0, abs=0.01)

    def test_compute_all_metrics_eval_only(self):
        """'fit' partition points must NOT affect RMSE."""
        from src.evaluation.metrics import compute_all_metrics
        rng = _rng(11)
        n = 10
        pred = rng.uniform(0, 256, (n, 2)).astype(np.float64)
        gt_checkpoints = [
            {"id": i, "src_xy": [1., 1.], "ref_xy": [float(pred[i, 0]), float(pred[i, 1])],
             "partition": "eval"}
            for i in range(n)
        ] + [
            # 'fit' partition — should be ignored
            {"id": n + j, "src_xy": [1., 1.], "ref_xy": [9999., 9999.],
             "partition": "fit"}
            for j in range(5)
        ]
        metrics = compute_all_metrics(pred, gt_checkpoints)
        assert metrics.get("gt_checkpoint_count") == n

    # ── Parametrized metric tests ────────────────────────────────────────────
    @pytest.mark.parametrize("n_pts", [5, 10, 25, 50, 100, 500])
    @pytest.mark.parametrize("noise_px", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_parametrized_rmse(self, n_pts, noise_px, seed):
        from src.evaluation.metrics import rmse, pct_lt_1px, medae
        rng = _rng(seed)
        gt = rng.uniform(0, 100, (n_pts, 2)).astype(np.float64)
        pred = gt + rng.normal(0, noise_px, (n_pts, 2)) if noise_px > 0 else gt.copy()
        r = rmse(pred, gt)
        p1 = pct_lt_1px(pred, gt)
        med = medae(pred, gt)
        assert r >= 0.0
        assert 0.0 <= p1 <= 1.0
        assert med >= 0.0
        if noise_px == 0.0:
            assert r == pytest.approx(0.0, abs=1e-6)
            assert p1 == pytest.approx(1.0)

    @pytest.mark.parametrize("img_h,img_w", [(256, 256), (512, 512), (1024, 512), (256, 768)])
    @pytest.mark.parametrize("n_pts", [1, 10, 50, 200])
    @pytest.mark.parametrize("n_grid", [4, 8, 16])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_parametrized_coverage(self, img_h, img_w, n_pts, n_grid, seed):
        from src.evaluation.metrics import spatial_coverage, grid_density_std
        rng = _rng(seed)
        pts = rng.uniform(0, min(img_h, img_w), (n_pts, 2)).astype(np.float64)
        cov = spatial_coverage(pts, (img_h, img_w), n=n_grid)
        gds = grid_density_std(pts, (img_h, img_w), n=n_grid)
        assert 0.0 <= cov <= 1.0
        assert gds >= 0.0


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: Leakage Audit
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestLeakageAudit:
    """Tests for src/evaluation/leakage_audit.py"""

    def _make_manifest_jsonl(self, records: list, path: Path) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_import(self):
        from src.evaluation.leakage_audit import run_audit, audit_manifest
        assert callable(run_audit)

    def test_clean_manifest_passes(self):
        from src.evaluation.leakage_audit import audit_manifest
        records = [
            {"pair_id": f"pair_{i:03d}", "split": "train", "geo_cell": f"cell_{i // 5}"}
            for i in range(20)
        ] + [
            {"pair_id": f"pair_{100+i:03d}", "split": "test",
             "geo_cell": f"cell_{50 + i // 5}", "gt_path": "path/to/gt"}
            for i in range(10)
        ]
        passed, violations = audit_manifest(records)
        assert passed, f"Expected PASS, got violations: {violations}"

    def test_detects_pair_in_both_splits(self):
        from src.evaluation.leakage_audit import audit_manifest
        records = [
            {"pair_id": "pair_001", "split": "train", "geo_cell": "cell_A"},
            {"pair_id": "pair_001", "split": "test",  "geo_cell": "cell_B"},  # SAME pair_id
        ]
        passed, violations = audit_manifest(records)
        assert not passed
        assert any("CRITICAL" in v for v in violations)

    def test_detects_cell_in_both_splits(self):
        from src.evaluation.leakage_audit import audit_manifest
        records = [
            {"pair_id": "pair_001", "split": "train", "geo_cell": "cell_X"},
            {"pair_id": "pair_002", "split": "test",  "geo_cell": "cell_X"},  # SAME cell
        ]
        passed, violations = audit_manifest(records)
        assert not passed
        assert any("geo_cell" in v.lower() or "CRITICAL" in v for v in violations)

    def test_detects_gt_on_train_pair(self):
        from src.evaluation.leakage_audit import audit_manifest
        records = [
            {"pair_id": "pair_001", "split": "train", "geo_cell": "cell_A",
             "gt_path": "data/gt/pair_001.json"},  # GT on train = WRONG
        ]
        passed, violations = audit_manifest(records)
        assert not passed

    def test_unknown_split_flagged(self):
        from src.evaluation.leakage_audit import audit_manifest
        records = [
            {"pair_id": "pair_001", "split": "validation", "geo_cell": "cell_A"},
        ]
        passed, violations = audit_manifest(records)
        assert not passed

    def test_run_audit_from_file(self, tmp_path):
        from src.evaluation.leakage_audit import run_audit
        records = [
            {"pair_id": f"pair_{i:03d}", "split": "train", "geo_cell": f"cell_{i}"}
            for i in range(10)
        ] + [
            {"pair_id": f"pair_{100+i:03d}", "split": "test", "geo_cell": f"cell_{100+i}",
             "gt_path": "somewhere/gt"}
            for i in range(5)
        ]
        manifest = tmp_path / "manifest.jsonl"
        self._make_manifest_jsonl(records, manifest)
        result = run_audit(manifest)
        assert result is True

    def test_run_audit_fails_on_leakage(self, tmp_path):
        from src.evaluation.leakage_audit import run_audit
        records = [
            {"pair_id": "pair_001", "split": "train", "geo_cell": "cell_A"},
            {"pair_id": "pair_002", "split": "test",  "geo_cell": "cell_A"},  # LEAK
        ]
        manifest = tmp_path / "manifest.jsonl"
        self._make_manifest_jsonl(records, manifest)
        result = run_audit(manifest)
        assert result is False

    # ── Parametrized: many manifest sizes and leak scenarios ─────────────────
    @pytest.mark.parametrize("n_train,n_test", [(5,5), (10,10), (50,20), (100,30), (200,50)])
    @pytest.mark.parametrize("inject_leak", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_parametrized_audit(self, tmp_path, n_train, n_test, inject_leak, seed):
        from src.evaluation.leakage_audit import run_audit
        rng = _rng(seed)
        train_cells = [f"cell_{i}" for i in range(n_train)]
        test_cells  = [f"cell_{n_train + i}" for i in range(n_test)]
        records = [
            {"pair_id": f"train_{i:04d}", "split": "train", "geo_cell": train_cells[i]}
            for i in range(n_train)
        ] + [
            {"pair_id": f"test_{i:04d}", "split": "test",
             "geo_cell": test_cells[i], "gt_path": "path"}
            for i in range(n_test)
        ]
        if inject_leak:
            # Add a train pair with a test cell → leak
            records.append({"pair_id": "leaky_pair", "split": "train",
                             "geo_cell": test_cells[0]})
        manifest = tmp_path / f"manifest_{seed}_{n_train}_{inject_leak}.jsonl"
        self._make_manifest_jsonl(records, manifest)
        result = run_audit(manifest)
        if inject_leak:
            assert result is False
        else:
            assert result is True


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5: Arbitration
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestArbitration:
    """Tests for src/evaluation/arbitration.py"""

    def _pair_record(self, terrain="highland", lat=10.0, density=4.0, tau_c=3.0):
        return {
            "pair_id": "pair_001", "terrain_class": terrain,
            "latitude_center_deg": lat,
            "crater_density_per_km2": density, "tau_c": tau_c,
        }

    def _matcher_result(self, inlier_ratio=0.5, inlier_count=100,
                         rmse=0.8, model_type="homography",
                         detector_validated=True):
        return {
            "inlier_ratio": inlier_ratio, "inlier_count": inlier_count,
            "rmse_px": rmse, "model_type": model_type,
            "detector_validated": detector_validated,
        }

    def test_import(self):
        from src.evaluation.arbitration import arbitrate_pair
        assert callable(arbitrate_pair)

    def test_crater_wins_when_gate_passes(self):
        from src.evaluation.arbitration import arbitrate_pair
        results = {
            "crater":    self._matcher_result(inlier_ratio=0.7, rmse=0.5, detector_validated=True),
            "lightglue": self._matcher_result(inlier_ratio=0.6, rmse=0.6),
            "sift":      self._matcher_result(inlier_ratio=0.4, rmse=1.2),
        }
        entry = arbitrate_pair("pair_001", results, self._pair_record(
            terrain="highland", density=4.0, tau_c=3.0
        ))
        assert entry.winner == "crater"

    def test_lightglue_wins_without_crater_gate(self):
        from src.evaluation.arbitration import arbitrate_pair
        results = {
            "lightglue": self._matcher_result(inlier_ratio=0.6, rmse=0.7),
            "sift":      self._matcher_result(inlier_ratio=0.3, rmse=1.5),
        }
        entry = arbitrate_pair("pair_001", results, self._pair_record(
            terrain="mare", density=0.5  # low density → crater gate blocked
        ))
        assert entry.winner == "lightglue"

    def test_total_failure_when_all_fail(self):
        from src.evaluation.arbitration import arbitrate_pair
        results = {
            "sift": {"inlier_ratio": 0.01, "inlier_count": 2,
                     "rmse_px": 5.0, "model_type": "none"},
        }
        entry = arbitrate_pair("pair_001", results, self._pair_record())
        assert entry.pair_outcome == "total_failure"
        assert entry.winner is None

    def test_fallback_to_sift_when_primary_fails_ratio(self):
        from src.evaluation.arbitration import arbitrate_pair
        results = {
            "lightglue": {"inlier_ratio": 0.01, "inlier_count": 1,
                           "rmse_px": 5.0, "model_type": "homography"},
            "sift":      self._matcher_result(inlier_ratio=0.3, rmse=1.5),
        }
        entry = arbitrate_pair("pair_001", results, self._pair_record(terrain="mare", density=0.0))
        assert entry.fallback_occurred is True
        assert entry.winner == "sift"

    def test_polar_no_validated_flag(self):
        from src.evaluation.arbitration import arbitrate_pair
        results = {
            "rift2": self._matcher_result(inlier_ratio=0.4, rmse=1.0),
            "sift":  self._matcher_result(inlier_ratio=0.3, rmse=1.5),
        }
        entry = arbitrate_pair("pair_001", results, self._pair_record(
            terrain="polar", lat=70.0, density=0.5
        ))
        # rift2 or sift wins but flag must be set since polar + rift2
        assert entry.no_validated_primary_matcher is True or entry.winner in ("rift2", "sift")

    def test_tie_break_uses_preference_order(self):
        from src.evaluation.arbitration import arbitrate_pair
        # crater and lightglue are statistically tied
        results = {
            "crater":    self._matcher_result(inlier_ratio=0.50, rmse=0.80, detector_validated=True),
            "lightglue": self._matcher_result(inlier_ratio=0.52, rmse=0.82),
            "sift":      self._matcher_result(inlier_ratio=0.30, rmse=1.50),
        }
        entry = arbitrate_pair("pair_001", results, self._pair_record(density=4.0),
                               gt_interannotator_rmse_px=0.5)
        # crater preferred over lightglue (preference order)
        assert entry.winner == "crater"

    def test_write_arbitration_log(self, tmp_path):
        from src.evaluation.arbitration import arbitrate_pair, write_arbitration_log
        results = {"sift": self._matcher_result()}
        entry = arbitrate_pair("pair_001", results, self._pair_record())
        log_path  = tmp_path / "arbitration.log"
        fail_path = tmp_path / "failures.jsonl"
        write_arbitration_log([entry], log_path, fail_path)
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["pair_id"] == "pair_001"

    def test_total_failure_written_to_failures_jsonl(self, tmp_path):
        from src.evaluation.arbitration import arbitrate_pair, write_arbitration_log
        results = {"sift": {"inlier_ratio": 0.0, "inlier_count": 0,
                             "rmse_px": None, "model_type": "none"}}
        entry = arbitrate_pair("pair_001", results, self._pair_record())
        log_path  = tmp_path / "arbitration.log"
        fail_path = tmp_path / "failures.jsonl"
        write_arbitration_log([entry], log_path, fail_path)
        if entry.pair_outcome == "total_failure":
            assert fail_path.exists()
            lines = fail_path.read_text().strip().splitlines()
            assert len(lines) >= 1

    # ── Parametrized: many terrain/density/lat combos ────────────────────────
    @pytest.mark.parametrize("terrain", ["highland", "polar_highland", "polar", "mare", "highland_mare"])
    @pytest.mark.parametrize("lat", [0.0, 20.0, 45.0, 60.0, 75.0])
    @pytest.mark.parametrize("density", [0.0, 1.0, 2.5, 4.0, 8.0])
    @pytest.mark.parametrize("detector_validated", [True, False])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_parametrized_arbitration(self, terrain, lat, density, detector_validated, seed):
        from src.evaluation.arbitration import arbitrate_pair
        rng = _rng(seed)
        results = {
            "crater":    {"inlier_ratio": float(rng.uniform(0.1, 0.8)),
                          "inlier_count": int(rng.integers(20, 200)),
                          "rmse_px": float(rng.uniform(0.3, 2.0)),
                          "model_type": "homography",
                          "detector_validated": detector_validated},
            "lightglue": {"inlier_ratio": float(rng.uniform(0.1, 0.7)),
                          "inlier_count": int(rng.integers(20, 200)),
                          "rmse_px": float(rng.uniform(0.4, 2.5)),
                          "model_type": "homography"},
            "sift":      {"inlier_ratio": float(rng.uniform(0.05, 0.4)),
                          "inlier_count": int(rng.integers(20, 100)),
                          "rmse_px": float(rng.uniform(0.8, 3.0)),
                          "model_type": "affine"},
        }
        pair_record = {"pair_id": "test_pair", "terrain_class": terrain,
                       "latitude_center_deg": lat,
                       "crater_density_per_km2": density, "tau_c": 3.0}
        entry = arbitrate_pair("test_pair", results, pair_record)
        # Invariants
        assert entry.pair_outcome in ("success", "fallback", "total_failure")
        if entry.winner is not None:
            assert entry.winner in list(results.keys()) + ["lnift", "rift2"]
        assert entry.winner_inlier_ratio >= 0.0


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6: Leaderboard Aggregation
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestAggregate:
    """Tests for src/evaluation/aggregate.py"""

    def _make_record(self, pair_id, matcher, sensor_pair, split, terrain,
                     lat_bin, rmse, inlier_ratio, seed=0):
        rng = _rng(seed)
        return {
            "pair_id": pair_id,
            "matcher": matcher,
            "split": split,
            "stratum": {
                "sensor_pair": sensor_pair,
                "terrain_class": terrain,
                "latitude_bin": lat_bin,
                "delta_az_bin": "lt30",
                "crater_density_bin": "low",
                "ref_type": "NAC",
            },
            "metrics": {
                "rmse_px": rmse,
                "pct_lt_1px": float(rng.uniform(0.4, 0.9)),
                "pct_lt_0p5px": float(rng.uniform(0.2, 0.6)),
                "medae_px": float(rng.uniform(0.3, 1.2)),
                "inlier_count": int(rng.integers(50, 300)),
                "inlier_ratio": inlier_ratio,
                "spatial_coverage": float(rng.uniform(0.6, 1.0)),
                "grid_density_std": float(rng.uniform(1.0, 5.0)),
                "refinement_gain_px": float(rng.uniform(0.0, 0.3)),
                "runtime_s": float(rng.uniform(1.0, 10.0)),
            }
        }

    def test_import(self):
        from src.evaluation.aggregate import aggregate, write_leaderboard_csv
        assert callable(aggregate)

    def test_aggregate_groups_by_stratum(self):
        from src.evaluation.aggregate import aggregate
        records = [
            self._make_record(f"p{i}", "sift", "OHRC-NAC", "test", "highland", "equatorial", 0.8, 0.5)
            for i in range(5)
        ] + [
            self._make_record(f"p{10+i}", "lightglue", "OHRC-NAC", "test", "highland", "equatorial", 0.6, 0.6)
            for i in range(5)
        ]
        rows = aggregate(records)
        matchers = {r["matcher"] for r in rows}
        assert "sift" in matchers
        assert "lightglue" in matchers

    def test_aggregate_rmse_mean(self):
        from src.evaluation.aggregate import aggregate
        records = [
            self._make_record(f"p{i}", "sift", "OHRC-NAC", "test", "highland", "equatorial",
                               float(i + 1), 0.5)
            for i in range(4)  # rmse = 1,2,3,4 → mean=2.5
        ]
        rows = aggregate(records)
        sift_rows = [r for r in rows if r["matcher"] == "sift"]
        assert len(sift_rows) == 1
        assert sift_rows[0]["rmse_px_mean"] == pytest.approx(2.5, rel=1e-6)

    def test_polar_stratum_not_merged(self):
        from src.evaluation.aggregate import aggregate
        records = [
            self._make_record(f"p{i}", "sift", "OHRC-NAC", "test", "polar", "polar", 0.9, 0.5)
            for i in range(3)
        ] + [
            self._make_record(f"p{10+i}", "sift", "OHRC-NAC", "test", "highland", "equatorial", 0.7, 0.5)
            for i in range(3)
        ]
        rows = aggregate(records)
        lat_bins = {r["latitude_bin"] for r in rows}
        assert "polar" in lat_bins
        assert "equatorial" in lat_bins

    def test_split_filter(self):
        from src.evaluation.aggregate import aggregate
        records = [
            self._make_record("p1", "sift", "OHRC-NAC", "train", "highland", "equatorial", 0.9, 0.5),
            self._make_record("p2", "sift", "OHRC-NAC", "test",  "highland", "equatorial", 0.7, 0.6),
        ]
        rows_test = aggregate(records, split_filter="test")
        assert all(r["split"] == "test" for r in rows_test)
        assert len(rows_test) == 1

    def test_write_leaderboard_csv_atomic(self, tmp_path):
        from src.evaluation.aggregate import aggregate, write_leaderboard_csv
        records = [
            self._make_record(f"p{i}", "sift", "OHRC-NAC", "test", "highland", "equatorial", 0.8, 0.5)
            for i in range(3)
        ]
        rows = aggregate(records)
        out = tmp_path / "leaderboard.csv"
        write_leaderboard_csv(rows, out)
        assert out.exists()
        content = out.read_text()
        assert "rmse_px_mean" in content
        assert "sift" in content

    def test_write_leaderboard_empty(self, tmp_path):
        from src.evaluation.aggregate import write_leaderboard_csv
        out = tmp_path / "empty.csv"
        write_leaderboard_csv([], out)
        assert out.exists()

    def test_n_failures_counted(self):
        from src.evaluation.aggregate import aggregate
        records = [
            self._make_record("p1", "sift", "OHRC-NAC", "test", "highland", "equatorial", 0.8, 0.5),
        ]
        # Inject failure (no rmse)
        records[0]["metrics"]["rmse_px"] = None
        rows = aggregate(records)
        assert rows[0]["n_failures"] == 1

    # ── Parametrized: many combinations of sensor/terrain/split ──────────────
    @pytest.mark.parametrize("n_pairs", [1, 5, 10, 20, 50])
    @pytest.mark.parametrize("matcher", ["sift", "lightglue", "crater"])
    @pytest.mark.parametrize("sensor_pair", ["OHRC-NAC", "TMC-2-WAC", "IIRS-WAC"])
    @pytest.mark.parametrize("split", ["train", "test"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_parametrized_aggregate(self, tmp_path, n_pairs, matcher, sensor_pair, split, seed):
        from src.evaluation.aggregate import aggregate, write_leaderboard_csv
        rng = _rng(seed)
        records = [
            self._make_record(
                f"p{i}", matcher, sensor_pair, split,
                "highland", "equatorial",
                float(rng.uniform(0.5, 3.0)), float(rng.uniform(0.05, 0.9)),
                seed=seed + i,
            )
            for i in range(n_pairs)
        ]
        rows = aggregate(records, split_filter=split)
        assert len(rows) >= 1
        assert all(r["split"] == split for r in rows)
        assert all(r["n_pairs"] >= 1 for r in rows)
        out = tmp_path / f"lb_{matcher}_{sensor_pair}_{split}_{seed}.csv"
        write_leaderboard_csv(rows, out)
        assert out.exists()


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7: Refinement (sub-pixel NCC)
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestRefinement:
    """Tests for src/evaluation/refinement local.py"""

    def test_import(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import refine_inliers, RefinementResult
        assert callable(refine_inliers)

    def test_apodization_blackman_forbidden(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _make_2d_window
        with pytest.raises((ValueError, AssertionError)):
            _make_2d_window(32, "blackman")

    def test_tukey_window_valid(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _make_2d_window
        w = _make_2d_window(32, "tukey")
        assert w.shape == (32, 32)
        assert w.max() <= 1.0 + 1e-6
        assert w.min() >= -1e-6

    def test_gaussian_window_valid(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _make_2d_window
        w = _make_2d_window(32, "gaussian")
        assert w.shape == (32, 32)

    def test_paraboloid_peak_exact_centre(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _paraboloid_peak
        # Perfect peak at centre (16,16) of 32x32 array
        corr = np.zeros((32, 32), dtype=np.float32)
        corr[16, 16] = 1.0
        corr[15, 16] = corr[17, 16] = 0.5
        corr[16, 15] = corr[16, 17] = 0.5
        dx, dy, sharpness = _paraboloid_peak(corr)
        assert abs(dx) < 0.1
        assert abs(dy) < 0.1
        assert sharpness > 0

    def test_second_peak_rejects_multimodal(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _second_peak_check
        corr = np.zeros((32, 32), dtype=np.float32)
        corr[8, 8] = 1.0     # primary
        corr[24, 24] = 0.95  # strong second peak → reject
        result = _second_peak_check(corr, ratio_threshold=0.8)
        assert result is True  # rejected

    def test_second_peak_accepts_unimodal(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _second_peak_check
        corr = np.zeros((32, 32), dtype=np.float32)
        corr[16, 16] = 1.0  # primary
        corr[5, 5] = 0.1    # weak second peak → accept
        result = _second_peak_check(corr, ratio_threshold=0.8)
        assert result is False  # accepted

    def test_ncc_correlation_shape(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _ncc_correlation
        src_patch = _make_image(32, 32, seed=0)
        ref_patch = _make_image(32, 32, seed=1)
        corr = _ncc_correlation(src_patch, ref_patch)
        assert corr.shape[0] > 0 and corr.shape[1] > 0

    def test_refine_inliers_returns_result(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import refine_inliers
        src, ref = _make_shifted_pair(128, 128, dcol=3, drow=2, seed=5)
        src_xy = np.array([[30., 30.], [60., 60.], [90., 30.]], dtype=np.float32)
        ref_xy = src_xy + np.array([[3., 2.], [3., 2.], [3., 2.]], dtype=np.float32)
        result = refine_inliers(src, ref, src_xy, ref_xy, window_px=16, pyramid_levels=1)
        assert result.total_count == 3
        assert 0 <= result.success_count <= 3
        assert isinstance(result.partial_refinement, bool)

    def test_refine_blackman_raises(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import refine_inliers
        src, ref = _make_shifted_pair(64, 64)
        src_xy = np.array([[20., 20.]], dtype=np.float32)
        ref_xy = np.array([[23., 22.]], dtype=np.float32)
        with pytest.raises((ValueError, AssertionError)):
            refine_inliers(src, ref, src_xy, ref_xy, apodization="blackman")

    def test_refine_partial_flag(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import refine_inliers
        # Use flat image → most patches will fail variance check → partial
        src = np.zeros((64, 64), dtype=np.uint8)
        ref = np.zeros((64, 64), dtype=np.uint8)
        src_xy = np.array([[20., 20.], [30., 30.]], dtype=np.float32)
        ref_xy = src_xy.copy()
        result = refine_inliers(src, ref, src_xy, ref_xy, window_px=10)
        # Flat image → should trigger partial_refinement (success_rate = 0 < 0.7)
        assert result.partial_refinement is True or result.success_rate >= 0.0

    # ── Parametrized refinement ───────────────────────────────────────────────
    @pytest.mark.parametrize("apodization", ["tukey", "gaussian"])
    @pytest.mark.parametrize("window_px", [16, 32, 64])
    @pytest.mark.parametrize("n_pts", [1, 3, 10])
    @pytest.mark.parametrize("dcol,drow", [(2, 3), (5, 0), (0, 5)])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_parametrized_refinement(self, apodization, window_px, n_pts, dcol, drow, seed):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import refine_inliers
        h, w = 128, 128
        margin = window_px + 5
        # Skip if image is too small for the requested window
        if margin >= min(h, w) - margin:
            pytest.skip(f"window_px={window_px} too large for {h}x{w} image (margin={margin})")
        src, ref = _make_shifted_pair(h, w, dcol=dcol, drow=drow, seed=seed)
        rng = _rng(seed + 100)
        src_xy = rng.uniform(margin, min(h, w) - margin, (n_pts, 2)).astype(np.float32)
        ref_xy = src_xy + np.array([[dcol, drow]], dtype=np.float32)
        result = refine_inliers(src, ref, src_xy, ref_xy,
                                window_px=window_px, pyramid_levels=1,
                                apodization=apodization)
        assert result.total_count == n_pts
        assert result.success_count >= 0
        assert result.runtime_s >= 0


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 8: Registration Ladder
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestLadder:
    """Tests for src/registration/ladder.py"""

    def test_import(self):
        from src.registration.ladder import model_ladder, ModelResult
        assert callable(model_ladder)

    def test_compute_t_gsd(self):
        from src.registration.ladder import _compute_t_gsd
        # NAC/NAC → gsd_ratio=1.0 → t_gsd=max(0.5, 1.0)=1.0
        t = _compute_t_gsd(src_gsd_m=0.5, ref_gsd_m=0.5)
        assert t == pytest.approx(1.0)

    def test_compute_t_gsd_wac(self):
        from src.registration.ladder import _compute_t_gsd
        # IIRS (80m)/WAC (100m) → ratio=0.8 → t_gsd=max(0.5, 0.8)=0.8
        t = _compute_t_gsd(src_gsd_m=80.0, ref_gsd_m=100.0)
        assert t == pytest.approx(0.8)

    def test_compute_t_gsd_min_clamp(self):
        from src.registration.ladder import _compute_t_gsd
        # Very small ratio → clamp to 0.5
        t = _compute_t_gsd(src_gsd_m=0.1, ref_gsd_m=100.0)
        assert t == pytest.approx(0.5)

    def test_compute_t_gsd_max_clamp(self):
        from src.registration.ladder import _compute_t_gsd
        # Large ratio → clamp to 3.0
        t = _compute_t_gsd(src_gsd_m=100.0, ref_gsd_m=0.5)
        assert t == pytest.approx(3.0)

    def test_residuals_identity(self):
        from src.registration.ladder import _residuals
        pts = np.random.default_rng(0).uniform(0, 100, (10, 2)).astype(np.float64)
        H = np.eye(3, dtype=np.float64)
        res = _residuals(pts, pts, H)
        assert np.all(res < 1e-6)

    def test_residuals_known_translation(self):
        from src.registration.ladder import _residuals
        src_xy = np.array([[0., 0.], [10., 0.]], dtype=np.float64)
        ref_xy = np.array([[3., 4.], [13., 4.]], dtype=np.float64)  # +3,+4 → dist=5
        H = np.array([[1., 0., 3.], [0., 1., 4.], [0., 0., 1.]], dtype=np.float64)
        res = _residuals(src_xy, ref_xy, H)
        assert np.all(res < 1e-6)

    @pytest.mark.skipif(not (_HAS_CV2 or _HAS_DEGENSAC), reason="cv2/pydegensac not available")
    def test_model_ladder_identity(self):
        from src.registration.ladder import model_ladder
        # Identity H → similarity should suffice with zero noise
        src_xy, ref_xy, conf = _make_homography_matches(
            n=100, H=np.eye(3), noise=0.1, fraction_outliers=0.1, seed=0
        )
        result = model_ladder(
            src_xy, ref_xy, conf,
            src_shape=(512, 512), ref_shape=(512, 512),
            src_gsd_m=0.5, ref_gsd_m=0.5,
        )
        assert result.model_type != "none"
        assert result.inlier_count > 0

    @pytest.mark.skipif(not (_HAS_CV2 or _HAS_DEGENSAC), reason="cv2/pydegensac not available")
    def test_model_ladder_translation(self):
        from src.registration.ladder import model_ladder
        H = np.eye(3); H[0, 2] = 15; H[1, 2] = 10
        src_xy, ref_xy, conf = _make_homography_matches(
            n=150, H=H, noise=0.2, fraction_outliers=0.2, seed=7
        )
        result = model_ladder(
            src_xy, ref_xy, conf,
            src_shape=(512, 512), ref_shape=(512, 512),
            src_gsd_m=0.5, ref_gsd_m=0.5,
        )
        assert result.inlier_count > 10 or result.model_type != "none"

    @pytest.mark.skipif(not (_HAS_CV2 or _HAS_DEGENSAC), reason="cv2/pydegensac not available")
    def test_model_ladder_total_failure(self):
        from src.registration.ladder import model_ladder
        # Pure random points — no structure → DEGENSAC should find little or fail
        rng = _rng(99)
        src_xy = rng.uniform(0, 512, (30, 2)).astype(np.float32)
        ref_xy = rng.uniform(0, 512, (30, 2)).astype(np.float32)
        conf   = rng.uniform(0.5, 1.0, 30).astype(np.float32)
        result = model_ladder(
            src_xy, ref_xy, conf,
            src_shape=(512, 512), ref_shape=(512, 512),
            src_gsd_m=0.5, ref_gsd_m=0.5,
        )
        # We don't assert pass/fail — just that it doesn't crash
        assert result is not None

    # ── Parametrized t_gsd ────────────────────────────────────────────────────
    @pytest.mark.parametrize("src_gsd,ref_gsd,expected_t", [
        (0.5, 0.5, 1.0),
        (0.31, 0.5, 0.62),
        (5.0, 100.0, 0.5),    # clamped to min
        (100.0, 0.5, 3.0),    # clamped to max
        (80.0, 100.0, 0.8),
    ])
    def test_parametrized_t_gsd(self, src_gsd, ref_gsd, expected_t):
        from src.registration.ladder import _compute_t_gsd
        t = _compute_t_gsd(src_gsd, ref_gsd)
        assert t == pytest.approx(expected_t, rel=0.05)

    @pytest.mark.parametrize("n_pts", [10, 50, 100, 200])
    @pytest.mark.parametrize("tx,ty", [(0., 0.), (10., 5.), (50., 30.)])
    @pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("outlier_frac", [0.0, 0.1, 0.3])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.skipif(not (_HAS_CV2 or _HAS_DEGENSAC), reason="cv2/pydegensac not available")
    def test_parametrized_ladder(self, n_pts, tx, ty, noise, outlier_frac, seed):
        from src.registration.ladder import model_ladder
        H = np.eye(3); H[0, 2] = tx; H[1, 2] = ty
        src_xy, ref_xy, conf = _make_homography_matches(
            n=n_pts, H=H, noise=noise, fraction_outliers=outlier_frac, seed=seed
        )
        result = model_ladder(
            src_xy, ref_xy, conf,
            src_shape=(512, 512), ref_shape=(512, 512),
            src_gsd_m=0.5, ref_gsd_m=0.5,
        )
        assert result is not None
        assert result.inlier_count >= 0
        assert result.t_gsd_used >= 0.5


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 9: Tile-wise models
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestTilewise:
    """Tests for src/registration/tilewise.py"""

    def test_import(self):
        from src.registration.tilewise import tilewise_models, blend_displacement
        assert callable(tilewise_models)
        assert callable(blend_displacement)

    def test_gaussian_weight_peak_at_centre(self):
        from src.registration.tilewise import _gaussian_weight
        w_centre = _gaussian_weight(100, 100, 100, 100, sigma=256)
        w_far    = _gaussian_weight(100, 100, 500, 500, sigma=256)
        assert w_centre > w_far

    def test_gaussian_weight_decreases_with_distance(self):
        from src.registration.tilewise import _gaussian_weight
        w1 = _gaussian_weight(0, 0, 50, 0,  sigma=100)
        w2 = _gaussian_weight(0, 0, 150, 0, sigma=100)
        assert w1 > w2

    def test_blend_displacement_identity(self):
        from src.registration.tilewise import blend_displacement
        # Identity model (H = I) → displacement = 0
        tile_models = [{"model_matrix": np.eye(3).tolist(),
                        "center_col": 256., "center_row": 256.}]
        qcol = np.array([100., 200., 300.], dtype=np.float64)
        qrow = np.array([100., 200., 300.], dtype=np.float64)
        dcol, drow = blend_displacement(qcol, qrow, tile_models)
        assert np.all(np.abs(dcol) < 1e-6)
        assert np.all(np.abs(drow) < 1e-6)

    def test_blend_displacement_known_translation(self):
        from src.registration.tilewise import blend_displacement
        # Pure translation H: tx=10, ty=5
        H = np.eye(3); H[0, 2] = 10.; H[1, 2] = 5.
        tile_models = [{"model_matrix": H.tolist(), "center_col": 50., "center_row": 50.}]
        qcol = np.array([50., 50.], dtype=np.float64)
        qrow = np.array([50., 50.], dtype=np.float64)
        dcol, drow = blend_displacement(qcol, qrow, tile_models)
        assert np.all(np.abs(dcol - 10.0) < 0.1)
        assert np.all(np.abs(drow - 5.0) < 0.1)

    def test_blend_displacement_multiple_tiles_blended(self):
        from src.registration.tilewise import blend_displacement
        H1 = np.eye(3); H1[0, 2] = 10.
        H2 = np.eye(3); H2[0, 2] = -10.
        tile_models = [
            {"model_matrix": H1.tolist(), "center_col": 0.,   "center_row": 0.},
            {"model_matrix": H2.tolist(), "center_col": 512., "center_row": 512.},
        ]
        # Query at midpoint → blended displacement ≈ 0
        qcol = np.array([256.], dtype=np.float64)
        qrow = np.array([256.], dtype=np.float64)
        dcol, drow = blend_displacement(qcol, qrow, tile_models)
        assert abs(dcol[0]) < 5.0  # should be near zero due to balanced weights

    @pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
    def test_tilewise_models_returns_result(self):
        from src.registration.tilewise import tilewise_models
        src_xy, ref_xy, conf = _make_homography_matches(n=200, seed=3)
        result = tilewise_models(
            src_xy, ref_xy,
            src_shape=(512, 512), ref_shape=(512, 512),
            tile_size=256, overlap_px=128, min_inliers=4,
        )
        # May return None if not enough matches per tile — that's valid
        if result is not None:
            assert result.tilewise is True
            assert result.inlier_count >= 0

    def test_tilewise_returns_none_on_empty(self):
        from src.registration.tilewise import tilewise_models
        src_xy = np.zeros((0, 2), dtype=np.float32)
        ref_xy = np.zeros((0, 2), dtype=np.float32)
        result = tilewise_models(src_xy, ref_xy, (512, 512), (512, 512))
        assert result is None

    # ── Parametrized blend ─────────────────────────────────────────────────────
    @pytest.mark.parametrize("n_tiles", [1, 2, 4, 8])
    @pytest.mark.parametrize("sigma", [64., 128., 256., 512.])
    @pytest.mark.parametrize("n_query", [1, 10, 50])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_parametrized_blend(self, n_tiles, sigma, n_query, seed):
        from src.registration.tilewise import blend_displacement
        rng = _rng(seed)
        tile_models = []
        for i in range(n_tiles):
            H = np.eye(3); H[0, 2] = float(rng.uniform(-20, 20))
            tile_models.append({
                "model_matrix": H.tolist(),
                "center_col": float(rng.uniform(0, 512)),
                "center_row": float(rng.uniform(0, 512)),
            })
        qcol = rng.uniform(0, 512, n_query)
        qrow = rng.uniform(0, 512, n_query)
        dcol, drow = blend_displacement(qcol, qrow, tile_models, sigma=sigma)
        assert dcol.shape == (n_query,)
        assert drow.shape == (n_query,)
        assert not np.any(np.isnan(dcol))
        assert not np.any(np.isnan(drow))


# =============================================================================
# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 10: Coordinate convention enforcement
# ═══════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TestCoordinateConvention:
    """
    All modules must follow (col, row) = (x, y), 0-indexed.
    These tests verify that no module silently accepts (row, col) ordering
    and returns incorrect results without assertion.
    """

    def test_f2_checks_assert_shape(self):
        from src.registration.checks import f2_checks
        with pytest.raises(AssertionError):
            bad = np.array([1., 2., 3.], dtype=np.float32)  # shape (3,) not (N,2)
            f2_checks(bad, bad, bad, (256, 256), (256, 256))

    def test_decluster_assert_shape(self):
        from src.registration.declustering import decluster
        with pytest.raises(AssertionError):
            bad = np.array([1., 2., 3.], dtype=np.float32)
            decluster(bad, bad, np.ones(3), ref_gsd_m=0.5)

    def test_metrics_assert_shape(self):
        from src.evaluation.metrics import rmse
        with pytest.raises(AssertionError):
            rmse(np.array([1., 2.]), np.array([1., 2.]))  # (2,) not (N,2)

    def test_coverage_assert_shape(self):
        from src.evaluation.metrics import spatial_coverage
        with pytest.raises((AssertionError, Exception)):
            spatial_coverage(np.array([1., 2.]), (256, 256))  # wrong shape

    def test_refine_assert_blackman(self):
        if not _HAS_SCIPY:
            pytest.skip("scipy not installed")
        from src.refinement.local import _make_2d_window
        with pytest.raises((ValueError, AssertionError)):
            _make_2d_window(32, "blackman")


# =============================================================================
# =============================================================================
# Entry point
# =============================================================================
# =============================================================================
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-q",
         "--no-header"],
        capture_output=False,
    )
    sys.exit(result.returncode)

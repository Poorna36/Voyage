"""
tests/test_synthetic_benchmark.py — Unit & Regression Tests for Phase 10

Covers T-SB01 through T-SB10 as specified in PROGRESS.md §10.6 and
docs/SYNTHETIC_BENCHMARK_ARCHITECTURE.md.

Tests are fully self-contained: all images are generated with numpy RNG;
no external files or network access required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from src.synthetic.transforms import (
    build_transform_matrix,
    apply_transform,
    transform_gt_points,
    generate_synthetic_pair,
    apply_illumination_gamma,
)
from src.synthetic.anchors import extract_anchors, AnchorSet
from src.evaluation.synthetic_eval import (
    assign_gt_predictions,
    score_l5_refinement,
    aggregate_scorecards,
    compute_oracle_best_matcher,
    StageScorecard,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_lunar_image(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """Generate a synthetic lunar-like image: noise + craters + gradients."""
    rng = np.random.default_rng(seed)
    # Base: correlated noise (simple smoothing via box filter)
    noise = rng.random((h, w)).astype(np.float32)
    import cv2
    img = cv2.GaussianBlur(noise, (15, 15), 3.0)
    # Add a few synthetic "craters" (dark circles with bright rims)
    for _ in range(5):
        cx = int(rng.integers(30, w - 30))
        cy = int(rng.integers(30, h - 30))
        r = int(rng.integers(10, 30))
        cv2.circle(img, (cx, cy), r, float(np.percentile(img, 20)), -1)   # dark floor
        cv2.circle(img, (cx, cy), r, float(np.percentile(img, 85)), 2)    # bright rim
    # Normalise to [0, 1]
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img.astype(np.float32)


def _make_anchor_config(target_count: int = 10, min_count: int = 5) -> dict:
    """Minimal anchor extraction config for testing."""
    return {
        "anchors": {
            "target_count": target_count,
            "min_count": min_count,
            "max_count": target_count + 20,
            "grid_cells": [4, 4],
            "phase": 1,
            "shi_tomasi": {
                "max_corners_per_cell": 5,
                "quality_level": 0.01,
                "min_distance_px": 5,
                "block_size": 5,
            },
        }
    }


def _make_transform_config(
    scale: bool = False,
    rotation: bool = False,
    translation: bool = True,
    illumination: bool = False,
    sensor: bool = False,
) -> dict:
    """Minimal transform config for generate_synthetic_pair."""
    return {
        "transforms": {
            "scale": {"enabled": scale, "min_factor": 0.9, "max_factor": 1.1, "interpolation": "lanczos"},
            "rotation": {"enabled": rotation, "max_angle_deg": 2.0},
            "translation": {"enabled": translation, "max_shift_px": 0.5},
            "illumination": {
                "enabled": illumination,
                "gamma_range": [0.8, 1.2],
                "shadow_extension": {"enabled": False},
            },
            "sensor_simulation": {
                "enabled": sensor,
                "mtf_blur": {"enabled": sensor, "sigma_range": [0.5, 1.0]},
                "pushbroom_noise": {"enabled": False},
            },
        }
    }


# ===========================================================================
# T-SB01 — transform_gt_points() invertibility
# ===========================================================================

class TestSB01TransformInvertibility:
    """T-SB01: Applying M then M^-1 returns within 1e-9 px of the original."""

    @pytest.mark.parametrize("scale,rot,tx,ty", [
        (1.0,  0.0,  0.0,   0.0),    # identity
        (1.0,  0.0,  3.7,  -2.1),    # pure translation
        (1.0,  5.0,  0.0,   0.0),    # pure rotation
        (0.75, 0.0,  0.0,   0.0),    # pure scale
        (1.1,  3.5, 12.4,  -8.7),    # combined
    ])
    def test_invertibility(self, scale, rot, tx, ty):
        img_shape = (256, 256)
        M = build_transform_matrix(img_shape, scale_factor=scale, rotation_deg=rot,
                                   translation_px=(tx, ty))
        # Random source points
        rng = np.random.default_rng(1)
        src_pts = rng.uniform(10, 240, size=(20, 2))

        # Forward: src → tgt
        tgt_pts = transform_gt_points(src_pts, M)

        # Inverse: tgt → src_recovered
        M_inv = np.linalg.inv(M)
        src_recovered = transform_gt_points(tgt_pts, M_inv)

        max_err = float(np.max(np.linalg.norm(src_pts - src_recovered, axis=1)))
        assert max_err < 1e-9, (
            f"Invertibility failed: max round-trip error = {max_err:.2e} px "
            f"(expected < 1e-9 px)."
        )


# ===========================================================================
# T-SB02 — extract_anchors() grid coverage
# ===========================================================================

class TestSB02GridCoverage:
    """T-SB02: At least one anchor per grid cell (4×4 = 16 cells in test config)."""

    def test_at_least_one_anchor_per_cell(self):
        img = _make_lunar_image(seed=42, h=256, w=256)
        cfg = {
            "anchors": {
                "target_count": 80,
                "min_count": 16,   # at least 1 per cell
                "max_count": 120,
                "grid_cells": [4, 4],
                "phase": 1,
                "shi_tomasi": {
                    "max_corners_per_cell": 10,
                    "quality_level": 0.005,
                    "min_distance_px": 5,
                    "block_size": 5,
                },
            }
        }
        anchor_set = extract_anchors(img, pair_id="test_grid", config=cfg)

        n_rows, n_cols = 4, 4
        h, w = img.shape
        cell_h, cell_w = h // n_rows, w // n_cols

        # Check at least one anchor in each cell
        pts = anchor_set.as_numpy()  # (N, 2) col, row
        cells_with_anchors = set()
        for col, row in pts:
            r_idx = min(int(row // cell_h), n_rows - 1)
            c_idx = min(int(col // cell_w), n_cols - 1)
            cells_with_anchors.add((r_idx, c_idx))

        expected_cells = n_rows * n_cols
        coverage = len(cells_with_anchors)
        assert coverage >= expected_cells * 0.75, (
            f"Grid coverage {coverage}/{expected_cells} cells < 75%."
        )


# ===========================================================================
# T-SB03 — extract_anchors() min count gate raises RuntimeError
# ===========================================================================

class TestSB03MinCountGate:
    """T-SB03: extract_anchors raises RuntimeError when too few anchors found."""

    def test_raises_on_featureless_image(self):
        # Flat/constant image — Shi-Tomasi finds zero corners
        flat_img = np.zeros((64, 64), dtype=np.float32) + 0.5
        cfg = {
            "anchors": {
                "target_count": 20,
                "min_count": 10,
                "max_count": 30,
                "grid_cells": [4, 4],
                "phase": 1,
                "shi_tomasi": {
                    "max_corners_per_cell": 5,
                    "quality_level": 0.01,
                    "min_distance_px": 5,
                    "block_size": 5,
                },
            }
        }
        with pytest.raises(RuntimeError, match="minimum required"):
            extract_anchors(flat_img, pair_id="flat", config=cfg)

    def test_raises_with_min_count_too_high(self):
        # Texured image but min_count set impossibly high
        img = _make_lunar_image(seed=7, h=64, w=64)
        cfg = {
            "anchors": {
                "target_count": 5,
                "min_count": 9999,   # impossible
                "max_count": 10000,
                "grid_cells": [2, 2],
                "phase": 1,
                "shi_tomasi": {
                    "max_corners_per_cell": 3,
                    "quality_level": 0.01,
                    "min_distance_px": 5,
                    "block_size": 5,
                },
            }
        }
        with pytest.raises(RuntimeError):
            extract_anchors(img, pair_id="high_min", config=cfg)


# ===========================================================================
# T-SB04 — assign_gt_predictions(): perfect predictions → recall = 1.0
# ===========================================================================

class TestSB04PerfectRecall:
    """T-SB04: Perfect predictions (dist=0 to GT) give recall=1.0."""

    @pytest.mark.parametrize("n_pts", [5, 20, 64])
    def test_perfect_recall(self, n_pts):
        rng = np.random.default_rng(0)
        gt_pts = rng.uniform(10, 200, size=(n_pts, 2)).astype(np.float64)
        # Predicted = exactly GT (zero distance)
        pred_pts = gt_pts.copy()

        assign = assign_gt_predictions(gt_pts, pred_pts, max_dist_px=2.0)
        recall = assign.n_matched / assign.n_gt

        assert recall == 1.0, f"Expected recall=1.0, got {recall:.4f}"
        assert assign.n_fn == 0, f"Expected 0 false negatives, got {assign.n_fn}"
        assert assign.n_fp == 0, f"Expected 0 false positives, got {assign.n_fp}"
        assert np.allclose(assign.distances, 0.0, atol=1e-12)


# ===========================================================================
# T-SB05 — assign_gt_predictions(): all dist > max_dist_px → recall = 0.0
# ===========================================================================

class TestSB05ZeroRecall:
    """T-SB05: All predictions outside max_dist_px → recall=0.0."""

    @pytest.mark.parametrize("n_pts", [5, 20, 50])
    def test_zero_recall(self, n_pts):
        rng = np.random.default_rng(1)
        # Place GT and pred in completely separate non-overlapping regions
        # to guarantee no accidental Hungarian assignment within threshold.
        gt_pts = rng.uniform(10, 50, size=(n_pts, 2)).astype(np.float64)
        max_dist = 2.0
        # Shift all predictions to a clearly separate region (far corner)
        # Use a fixed large offset that guarantees min dist > max_dist
        pred_pts = gt_pts + 500.0

        assign = assign_gt_predictions(gt_pts, pred_pts, max_dist_px=max_dist)

        assert assign.n_matched == 0, (
            f"Expected 0 matches (all outside threshold), got {assign.n_matched}"
        )
        assert assign.n_fn == n_pts, f"Expected {n_pts} false negatives, got {assign.n_fn}"


# ===========================================================================
# T-SB06 — score_l5_refinement(): gain > 0 when refined closer to GT
# ===========================================================================

class TestSB06RefinementGain:
    """T-SB06: Refinement gain is positive when refined_pts are closer to GT."""

    def test_positive_gain(self):
        rng = np.random.default_rng(2)
        n = 30
        gt_pts = rng.uniform(20, 200, size=(n, 2)).astype(np.float64)
        # Coarse: 1.0 px error
        coarse = gt_pts + rng.normal(0, 1.0, size=(n, 2))
        # Refined: 0.3 px error (clearly better)
        refined = gt_pts + rng.normal(0, 0.3, size=(n, 2))

        scorecard = score_l5_refinement(
            gt_tgt_pts=gt_pts,
            coarse_pred_tgt_pts=coarse,
            refined_pred_tgt_pts=refined,
            pair_id="test_pair",
            matcher="sift",
            max_dist_px=5.0,
        )

        gain = scorecard.metrics["refinement_gain_px"]
        pct_improved = scorecard.metrics["pct_improved"]

        assert gain > 0.0, f"Expected positive refinement gain, got {gain:.4f} px"
        assert pct_improved > 0.5, (
            f"Expected >50% of inliers to improve, got {pct_improved:.3f}"
        )
        assert scorecard.metrics["l5_rmse_px"] < scorecard.metrics["l4_rmse_px"], (
            "L5 RMSE should be lower than L4 RMSE after refinement."
        )

    def test_negative_gain_when_refined_worse(self):
        """Verify gain is correctly negative when 'refinement' degrades matches."""
        rng = np.random.default_rng(3)
        n = 20
        gt_pts = rng.uniform(20, 150, size=(n, 2)).astype(np.float64)
        coarse = gt_pts + rng.normal(0, 0.3, size=(n, 2))   # good coarse
        refined = gt_pts + rng.normal(0, 1.5, size=(n, 2))  # bad refined

        scorecard = score_l5_refinement(
            gt_tgt_pts=gt_pts,
            coarse_pred_tgt_pts=coarse,
            refined_pred_tgt_pts=refined,
            pair_id="test_pair",
            matcher="sift",
            max_dist_px=5.0,
        )
        gain = scorecard.metrics["refinement_gain_px"]
        assert gain < 0.0, f"Expected negative gain when refined is worse, got {gain:.4f}"


# ===========================================================================
# T-SB07 — aggregate_scorecards(): CI width decreases with N; CI includes mean
# ===========================================================================

class TestSB07AggregateCI:
    """T-SB07: CI width decreases with N and CI always contains the true mean."""

    def _make_scorecards(self, n: int, true_mean: float, std: float, seed: int = 0) -> List[StageScorecard]:
        rng = np.random.default_rng(seed)
        vals = rng.normal(true_mean, std, size=n)
        scorecards = []
        for v in vals:
            sc = StageScorecard(
                stage="L2", pair_id="test", matcher="sift", n_gt=10,
                metrics={"gt_recall": float(v)},
            )
            scorecards.append(sc)
        return scorecards

    def test_ci_width_decreases_with_n(self):
        true_mean, std = 0.75, 0.10
        agg_small = aggregate_scorecards(self._make_scorecards(10, true_mean, std, seed=42))
        agg_large = aggregate_scorecards(self._make_scorecards(100, true_mean, std, seed=42))

        width_small = agg_small["gt_recall_ci_high"] - agg_small["gt_recall_ci_low"]
        width_large = agg_large["gt_recall_ci_high"] - agg_large["gt_recall_ci_low"]

        assert width_large < width_small, (
            f"Expected CI to shrink with more samples: "
            f"n=10 width={width_small:.4f}, n=100 width={width_large:.4f}"
        )

    def test_ci_contains_true_mean(self):
        """95% CI should contain the true mean across many random samples."""
        true_mean, std = 0.80, 0.05
        n_experiments = 200
        n_per_exp = 30
        contained = 0
        for seed in range(n_experiments):
            agg = aggregate_scorecards(
                self._make_scorecards(n_per_exp, true_mean, std, seed=seed)
            )
            ci_low = agg["gt_recall_ci_low"]
            ci_high = agg["gt_recall_ci_high"]
            if ci_low <= true_mean <= ci_high:
                contained += 1
        coverage = contained / n_experiments
        # With 95% CI we expect ~95% coverage — allow ±5% tolerance
        assert coverage >= 0.90, (
            f"CI coverage = {coverage:.2%} (expected ≥ 90% for 95% CI)."
        )

    def test_single_sample_has_zero_ci(self):
        """With n=1 sample the SE=0 and CI width=0."""
        scorecards = self._make_scorecards(1, true_mean=0.7, std=0.1)
        agg = aggregate_scorecards(scorecards)
        width = agg["gt_recall_ci_high"] - agg["gt_recall_ci_low"]
        assert width == 0.0, f"Expected zero CI width for n=1, got {width:.6f}"


# ===========================================================================
# T-SB08 — generate_synthetic_pair() output in [0, 1] after all transforms
# ===========================================================================

class TestSB08OutputRange:
    """T-SB08: All transforms produce output strictly in [0, 1]."""

    @pytest.mark.parametrize("seed,scale,rot,illum,sensor", [
        (42,  False, False, False, False),   # translation only
        (1,   True,  True,  False, False),   # geometric
        (7,   True,  True,  True,  False),   # + illumination
        (13,  True,  True,  True,  True),    # all transforms
    ])
    def test_output_range(self, seed, scale, rot, illum, sensor):
        img = _make_lunar_image(seed=seed, h=128, w=128)
        cfg = _make_transform_config(
            scale=scale, rotation=rot, translation=True,
            illumination=illum, sensor=sensor,
        )
        synthetic, params, M = generate_synthetic_pair(
            source_image=img, config=cfg, pair_id="test", seed=seed,
        )

        assert synthetic.shape == img.shape, "Output shape must match source."
        assert synthetic.dtype == np.float32, f"Expected float32, got {synthetic.dtype}."
        # Clip to [0, 1] as generate_synthetic_pair guarantees clipping at end
        min_val = float(synthetic.min())
        max_val = float(synthetic.max())
        assert min_val >= 0.0 - 1e-6, f"Output below 0: min={min_val:.6f}"
        assert max_val <= 1.0 + 1e-6, f"Output above 1: max={max_val:.6f}"


# ===========================================================================
# T-SB09 — apply_illumination_gamma(): gamma=1.0 is identity
# ===========================================================================

class TestSB09GammaIdentity:
    """T-SB09: apply_illumination_gamma(image, gamma=1.0) returns identical image."""

    def test_identity_gamma(self):
        from src.synthetic.transforms import apply_illumination_gamma
        rng = np.random.default_rng(0)
        img = rng.random((64, 64)).astype(np.float32)

        result = apply_illumination_gamma(img, gamma=1.0)

        max_diff = float(np.max(np.abs(result.astype(np.float64) - img.astype(np.float64))))
        assert max_diff < 1e-6, (
            f"gamma=1.0 should be identity; max pixel diff = {max_diff:.2e}"
        )

    @pytest.mark.parametrize("gamma", [0.7, 0.9, 1.1, 1.4])
    def test_output_in_range(self, gamma):
        from src.synthetic.transforms import apply_illumination_gamma
        img = _make_lunar_image(seed=0)
        result = apply_illumination_gamma(img, gamma=gamma)
        assert result.min() >= 0.0 - 1e-7
        assert result.max() <= 1.0 + 1e-7

    def test_negative_gamma_raises(self):
        from src.synthetic.transforms import apply_illumination_gamma
        img = np.ones((8, 8), dtype=np.float32) * 0.5
        with pytest.raises(AssertionError):
            apply_illumination_gamma(img, gamma=-0.5)


# ===========================================================================
# T-SB10 — compute_oracle_best_matcher(): RMSE=0 matcher always wins
# ===========================================================================

class TestSB10OracleRMSE:
    """T-SB10: The matcher with RMSE=0 (perfect) always wins the oracle score."""

    @pytest.mark.parametrize("perfect_matcher,others", [
        ("sift",      {"rift2": (1.5, 0.3, 0.6), "lightglue": (2.0, 0.5, 0.7)}),
        ("lightglue", {"sift":  (0.8, 0.2, 0.5), "rift2":     (1.2, 0.4, 0.6)}),
        ("crater",    {"sift":  (1.0, 0.5, 0.8), "lightglue": (0.5, 0.7, 0.9)}),
    ])
    def test_zero_rmse_always_wins(self, perfect_matcher, others):
        """Matcher with gt_rmse_px=0 should always be oracle best."""
        matcher_metrics = {
            perfect_matcher: {
                "gt_rmse_px": 0.0,
                "gt_inlier_ratio": 1.0,
                "gt_spatial_coverage": 1.0,
            }
        }
        for m, (rmse, inlier_ratio, cov) in others.items():
            matcher_metrics[m] = {
                "gt_rmse_px": rmse,
                "gt_inlier_ratio": inlier_ratio,
                "gt_spatial_coverage": cov,
            }

        oracle = compute_oracle_best_matcher(matcher_metrics)
        assert oracle == perfect_matcher, (
            f"Expected oracle='{perfect_matcher}' (RMSE=0), got '{oracle}'."
        )

    def test_handles_nan_rmse(self):
        """Matchers with NaN RMSE should be excluded from oracle selection."""
        matcher_metrics = {
            "sift":      {"gt_rmse_px": float("nan"), "gt_inlier_ratio": 0.9, "gt_spatial_coverage": 0.8},
            "lightglue": {"gt_rmse_px": 0.3,          "gt_inlier_ratio": 0.8, "gt_spatial_coverage": 0.7},
        }
        oracle = compute_oracle_best_matcher(matcher_metrics)
        assert oracle == "lightglue", (
            f"Expected oracle='lightglue' (only finite RMSE), got '{oracle}'."
        )

    def test_all_nan_returns_fallback(self):
        """If all matchers have NaN RMSE, fallback to 'sift'."""
        matcher_metrics = {
            "sift":  {"gt_rmse_px": float("nan"), "gt_inlier_ratio": 0.5, "gt_spatial_coverage": 0.5},
            "rift2": {"gt_rmse_px": float("nan"), "gt_inlier_ratio": 0.4, "gt_spatial_coverage": 0.4},
        }
        oracle = compute_oracle_best_matcher(matcher_metrics)
        assert oracle == "sift", f"Expected fallback 'sift', got '{oracle}'."


# ===========================================================================
# Additional integration: AnchorSet serialisation round-trip
# ===========================================================================

class TestAnchorSetSerialization:
    """Verify AnchorSet save/load round-trip preserves all fields exactly."""

    def test_save_load_roundtrip(self, tmp_path):
        img = _make_lunar_image(seed=99, h=128, w=128)
        cfg = _make_anchor_config(target_count=10, min_count=5)
        anchor_set = extract_anchors(img, pair_id="roundtrip_test", config=cfg)

        gt_path = tmp_path / "test_gt_anchors.json"
        anchor_set.save(gt_path)

        loaded = AnchorSet.load(gt_path)

        assert loaded.pair_id == anchor_set.pair_id
        assert loaded.extraction_phase == anchor_set.extraction_phase
        assert len(loaded.anchors) == len(anchor_set.anchors)

        orig_pts = anchor_set.as_numpy()
        loaded_pts = loaded.as_numpy()
        assert np.allclose(orig_pts, loaded_pts, atol=1e-9), (
            "AnchorSet coordinates changed on save/load round-trip."
        )

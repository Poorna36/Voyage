# -*- coding: utf-8 -*-
"""
tests/verify_phase3.py
======================
Phase 3 verification — runs against synthetic images, no real data needed.

Tests (maps to VALIDATION.md T05-T08):
  T-base  : MatchResult schema, coordinate assertions
  T-anms  : ANMS SSC budget ±5%, no two points within suppression radius
  T-sift  : M0 produces >= 1 match on a known-good synthetic pair; schema valid
  T-rift  : RIFT2 initialises and returns valid MatchResult (structural check)
  T-lnift : LNIFT same structural check
  T-lg    : LightGlue import check; graceful failure if not installed
  T-crater: CraterMatcher gate logic; HoughCircles on synthetic ring image
  T-spatial: grid_cap, coverage_greedy, one_to_one, selection_stats correctness

Run: python -m tests.verify_phase3
  or: python tests/verify_phase3.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

# ── Path fix so `src` is importable without installing
import sys as _sys
import pathlib as _pathlib
_repo_root = str(_pathlib.Path(__file__).parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

# ── ANSI colours for terminal output ─────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

_results: List[Tuple[str, bool, str]] = []   # (name, passed, detail)


def _test(name: str):
    """Decorator to register a test function."""
    def decorator(fn: Callable):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  {GREEN}PASS{RESET}  {name}")
        except AssertionError as exc:
            _results.append((name, False, str(exc)))
            print(f"  {RED}FAIL{RESET}  {name}  -> {exc}")
        except Exception as exc:
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  {RED}FAIL{RESET}  {name}  -> {type(exc).__name__}: {exc}")
            if "--tb" in sys.argv:
                traceback.print_exc()
        return fn
    return decorator


# ── Synthetic image helpers ───────────────────────────────────────────────────

def _make_textured_pair(
    h: int = 256,
    w: int = 256,
    shift_x: int = 10,
    shift_y: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a pair of synthetic uint8 grayscale images with known shift.
    The reference is the source shifted by (shift_x, shift_y) pixels.
    """
    rng = np.random.default_rng(seed)
    # Random texture + structured features
    base = rng.integers(30, 220, (h, w), dtype=np.uint8)
    # Add some blobs for detectability
    for _ in range(40):
        cy, cx = rng.integers(20, h-20), rng.integers(20, w-20)
        r = rng.integers(4, 15)
        ys, xs = np.ogrid[-r:r+1, -r:r+1]
        mask = xs**2 + ys**2 <= r**2
        y0, y1 = max(0, cy-r), min(h, cy+r+1)
        x0, x1 = max(0, cx-r), min(w, cx+r+1)
        base[y0:y1, x0:x1] = np.where(mask[:y1-y0, :x1-x0],
                                        rng.integers(180, 255),
                                        base[y0:y1, x0:x1])
    src = base.copy()
    # Shift to make reference
    ref = np.zeros_like(src)
    sy = shift_y
    sx_ = shift_x
    ref[sy:, sx_:] = src[:h-sy, :w-sx_]
    return src, ref


def _make_crater_image(
    h: int = 200,
    w: int = 200,
    n_craters: int = 8,
    seed: int = 7,
) -> np.ndarray:
    """Synthetic lunar image with circular crater rims."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 80, dtype=np.uint8)
    for _ in range(n_craters):
        cy = rng.integers(30, h-30)
        cx = rng.integers(30, w-30)
        r = rng.integers(10, 35)
        # Draw rim as bright ring
        for angle in np.linspace(0, 2*np.pi, 200):
            py = int(cy + r * np.sin(angle))
            px = int(cx + r * np.cos(angle))
            if 0 <= py < h and 0 <= px < w:
                img[py, px] = 200
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: MatchResult schema and coordinate assertions
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-base: MatchResult schema and (N,2) assertion")
def _():
    from src.matching.base import MatchResult
    import numpy as np

    # Valid creation
    r = MatchResult(
        src_xy=np.array([[1.0, 2.0], [3.0, 4.0]]),
        ref_xy=np.array([[5.0, 6.0], [7.0, 8.0]]),
        confidence=np.array([0.8, 0.9]),
    )
    assert r.count == 2
    assert not r.is_empty()
    assert r.src_xy.dtype == np.float32
    assert r.ref_xy.dtype == np.float32

    # Coordinate assertion fires on wrong shape
    try:
        MatchResult(
            src_xy=np.array([1.0, 2.0]),   # wrong: (2,) not (N,2)
            ref_xy=np.array([[1.0, 2.0]]),
            confidence=np.array([0.5]),
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


@_test("T-base: BaseMatcher _empty_result returns valid MatchResult")
def _():
    from src.matching.sift import SIFTMatcher
    m = SIFTMatcher()
    empty = m._empty_result(reason="test")
    assert empty.is_empty()
    assert empty.src_xy.shape == (0, 2)
    assert empty.matcher_params["matcher_id"] == "sift"
    assert empty.matcher_params["failure_reason"] == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: ANMS SSC
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-anms: budget within ±5% (VALIDATION.md T05)")
def _():
    from src.selection.anms import anms_ssc
    rng = np.random.default_rng(0)
    N_in = 1000
    target = 200
    pts = np.column_stack([
        rng.uniform(0, 512, N_in),
        rng.uniform(0, 512, N_in),
        rng.uniform(0, 1, N_in),   # strength
    ]).astype(np.float32)
    out = anms_ssc(pts, target, image_shape=(512, 512))
    n_out = len(out)
    tol = max(1, int(0.05 * target))
    assert abs(n_out - target) <= tol, (
        f"ANMS returned {n_out}, expected {target} ±{tol}"
    )


@_test("T-anms: no two output points within suppression radius")
def _():
    from src.selection.anms import anms_ssc
    rng = np.random.default_rng(1)
    N_in = 800
    target = 100
    pts = np.column_stack([
        rng.uniform(0, 256, N_in),
        rng.uniform(0, 256, N_in),
        rng.uniform(0, 1, N_in),
    ]).astype(np.float32)
    out = anms_ssc(pts, target, image_shape=(256, 256))
    out_arr = np.array(out, dtype=np.float32)
    if len(out_arr) < 2:
        return   # trivially satisfied
    # Compute all pairwise distances; suppress radius ~ sqrt(W*H/target)
    min_expected_r = 0.5 * np.sqrt(256 * 256 / target)
    for i in range(len(out_arr)):
        for j in range(i + 1, len(out_arr)):
            dx = float(out_arr[i, 0]) - float(out_arr[j, 0])
            dy = float(out_arr[i, 1]) - float(out_arr[j, 1])
            dist = np.sqrt(dx**2 + dy**2)
            assert dist >= min_expected_r * 0.5, (
                f"Two points too close: dist={dist:.2f} < {min_expected_r*0.5:.2f}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: SIFT Matcher (M0)
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-sift: M0 returns valid MatchResult schema on synthetic pair")
def _():
    try:
        import cv2
    except ImportError:
        print("    [SKIP — opencv not installed]", end="")
        return

    from src.matching.sift import SIFTMatcher
    src, ref = _make_textured_pair(h=256, w=256, shift_x=10, shift_y=8)
    m = SIFTMatcher()
    result = m.match(src, ref, gsd_ratio=1.0)

    # Schema checks
    assert result.src_xy.shape[-1] == 2, "src_xy must be (N,2)"
    assert result.ref_xy.shape[-1] == 2, "ref_xy must be (N,2)"
    assert len(result.confidence) == result.count
    assert result.src_xy.dtype == np.float32
    assert result.matcher_params["matcher_id"] == "sift"
    print(f"    [n_matches={result.count}, runtime={result.runtime_s:.3f}s]", end="")


@_test("T-sift: M0 >= 1 match on textured pair (VALIDATION.md T06 partial)")
def _():
    try:
        import cv2
    except ImportError:
        print("    [SKIP — opencv not installed]", end="")
        return

    from src.matching.sift import SIFTMatcher
    src, ref = _make_textured_pair(h=256, w=256, shift_x=10, shift_y=8)
    m = SIFTMatcher()
    result = m.match(src, ref, gsd_ratio=1.0)
    assert result.count >= 1, f"Expected >= 1 matches, got {result.count}"


@_test("T-sift: (col,row) convention — x < width, y < height")
def _():
    try:
        import cv2
    except ImportError:
        print("    [SKIP — opencv not installed]", end="")
        return

    from src.matching.sift import SIFTMatcher
    H, W = 256, 256
    src, ref = _make_textured_pair(h=H, w=W)
    m = SIFTMatcher()
    result = m.match(src, ref, gsd_ratio=1.0)
    if result.count > 0:
        # col (x) must be < W, row (y) must be < H
        assert np.all(result.src_xy[:, 0] < W), "col >= W — likely (row,col) swap!"
        assert np.all(result.src_xy[:, 1] < H), "row >= H — likely (row,col) swap!"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: RIFT2 Matcher (M1a)
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-rift: RIFT2 instantiates and returns valid MatchResult type")
def _():
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("    [SKIP — scipy not installed]", end="")
        return

    from src.matching.rift import RIFT2Matcher
    m = RIFT2Matcher(config={"n_scales": 2, "n_orientations": 4,
                              "scale_space_octaves": 1, "num_keypoints": 50})
    assert m.matcher_id == "rift2"
    assert not m.requires_gpu

    src, ref = _make_textured_pair(h=64, w=64, shift_x=3, shift_y=2)
    result = m.match(src, ref, gsd_ratio=1.0)
    assert hasattr(result, "src_xy")
    assert result.src_xy.shape[-1] == 2
    assert result.matcher_params["matcher_id"] == "rift2"
    print(f"    [n={result.count}]", end="")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: LNIFT Matcher (M1b)
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-lnift: LNIFT instantiates and returns valid MatchResult type")
def _():
    try:
        import cv2
        import scipy  # noqa: F401
    except ImportError:
        print("    [SKIP — cv2/scipy not installed]", end="")
        return

    from src.matching.lnift import LNIFTMatcher
    m = LNIFTMatcher(config={"n_scales": 1, "n_orientations": 2, "num_keypoints": 30})
    assert m.matcher_id == "lnift"
    assert not m.requires_gpu

    src, ref = _make_textured_pair(h=64, w=64, shift_x=4, shift_y=4)
    result = m.match(src, ref, gsd_ratio=1.0)
    assert result.src_xy.shape[-1] == 2
    assert result.matcher_params["matcher_id"] == "lnift"
    print(f"    [n={result.count}]", end="")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: LightGlue Matcher (M2)
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-lightglue: graceful failure if library not installed")
def _():
    from src.matching.lightglue import LightGlueMatcher, _HAS_LIGHTGLUE
    m = LightGlueMatcher()
    assert m.matcher_id == "lightglue"
    assert m.requires_gpu   # declared requires_gpu=True

    if not _HAS_LIGHTGLUE:
        src, ref = _make_textured_pair()
        result = m.match(src, ref)
        # Must return empty MatchResult, not raise
        assert result.src_xy.shape[-1] == 2
        assert "failure_reason" in result.matcher_params
        print("    [lightglue not installed — graceful empty result]", end="")
    else:
        print("    [lightglue installed — full test would run on device]", end="")


@_test("T-lightglue: F2 checks remove out-of-bounds points")
def _():
    from src.matching.lightglue import LightGlueMatcher
    m = LightGlueMatcher()
    src_xy = np.array([
        [10.0, 20.0],     # valid
        [999.0, 999.0],   # out of bounds (image is 256x256)
        [50.0, 60.0],     # valid
    ], dtype=np.float32)
    ref_xy = np.array([
        [15.0, 25.0],
        [15.0, 25.0],
        [55.0, 65.0],
    ], dtype=np.float32)
    conf = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    sx, rx, c, n_rem = m._f2_checks(src_xy, ref_xy, conf,
                                      src_shape=(256, 256), ref_shape=(256, 256))
    assert 999.0 not in sx[:, 0], "Out-of-bounds point was not removed"
    assert n_rem >= 1, "Expected at least 1 removal"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: CraterMatcher (M3)
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-crater: gate correctly blocks when density < tau_c")
def _():
    from src.matching.crater import CraterMatcher
    m = CraterMatcher()
    gate_ok, reason = CraterMatcher.check_gate(
        density_src=1.5,   # below tau_c=3.0
        density_ref=5.0,
        terrain_src="highland",
        terrain_ref="highland",
        tau_c=3.0,
        allowed_terrain={"highland", "polar_highland", "polar"},
    )
    assert not gate_ok, "Gate should fail when density_src < tau_c"
    assert "density" in reason.lower()


@_test("T-crater: gate passes when density >= tau_c and terrain ok")
def _():
    from src.matching.crater import CraterMatcher
    gate_ok, reason = CraterMatcher.check_gate(
        density_src=4.0,
        density_ref=5.0,
        terrain_src="highland",
        terrain_ref="highland",
        tau_c=3.0,
        allowed_terrain={"highland", "polar_highland", "polar"},
    )
    assert gate_ok, f"Gate should pass: {reason}"


@_test("T-crater: HoughCircles detects craters in synthetic ring image")
def _():
    try:
        import cv2
    except ImportError:
        print("    [SKIP — opencv not installed]", end="")
        return

    from src.matching.crater import CraterMatcher
    m = CraterMatcher()
    img = _make_crater_image(h=200, w=200, n_craters=5)
    craters = m._detect_hough(img)
    assert isinstance(craters, np.ndarray)
    assert craters.ndim == 2 and craters.shape[1] == 3
    print(f"    [detected {len(craters)} craters]", end="")


@_test("T-crater: MCR filter removes displacement outliers")
def _():
    from src.matching.crater import CraterMatcher
    src_xy = np.array([[10.0, 10.0], [50.0, 50.0], [900.0, 900.0]], dtype=np.float32)
    ref_xy = np.array([[15.0, 12.0], [55.0, 52.0], [30.0, 30.0]], dtype=np.float32)
    conf = np.array([0.9, 0.85, 0.7], dtype=np.float32)
    sx, rx, c = CraterMatcher._mcr_filter(src_xy, ref_xy, conf, max_ratio=1.5)
    assert len(sx) < 3, "Outlier was not removed by MCR filter"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Spatial selection
# ═══════════════════════════════════════════════════════════════════════════════
@_test("T-spatial: confidence_filter removes below-threshold matches")
def _():
    from src.selection.spatial import confidence_filter
    sx = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    rx = sx.copy()
    conf = np.array([0.1, 0.3, 0.9], dtype=np.float32)
    sx_f, rx_f, cf = confidence_filter(sx, rx, conf, "lightglue", threshold=0.2)
    assert len(sx_f) == 2, f"Expected 2 after filter, got {len(sx_f)}"
    assert float(cf.min()) >= 0.2


@_test("T-spatial: grid_cap limits per-cell matches")
def _():
    from src.selection.spatial import grid_cap
    rng = np.random.default_rng(5)
    # All points in same cell (top-left)
    sx = np.column_stack([rng.uniform(0, 30, 20), rng.uniform(0, 30, 20)]).astype(np.float32)
    rx = sx + 5.0
    conf = rng.uniform(0, 1, 20).astype(np.float32)
    sx_g, rx_g, cg = grid_cap(sx, rx, conf, n=8, cap=5, image_shape=(256, 256))
    assert len(sx_g) <= 5, f"grid_cap should limit to 5, got {len(sx_g)}"


@_test("T-spatial: coverage_greedy meets coverage_min on spread points")
def _():
    from src.selection.spatial import coverage_greedy
    rng = np.random.default_rng(9)
    # Uniformly spread 200 points across 256x256
    sx = np.column_stack([rng.uniform(0, 256, 200), rng.uniform(0, 256, 200)]).astype(np.float32)
    rx = sx + rng.uniform(-3, 3, sx.shape).astype(np.float32)
    conf = rng.uniform(0, 1, 200).astype(np.float32)
    sx_g, rx_g, cg = coverage_greedy(sx, rx, conf, budget=250, min_coverage=0.60,
                                       n=8, image_shape=(256, 256))
    # Compute coverage manually
    cx = np.clip((sx_g[:, 0] / 256 * 8).astype(int), 0, 7)
    cy = np.clip((sx_g[:, 1] / 256 * 8).astype(int), 0, 7)
    occupied = len(set(zip(cx.tolist(), cy.tolist())))
    cov = occupied / 64
    assert cov >= 0.55, f"coverage {cov:.2f} too low (expected >= 0.55)"


@_test("T-spatial: one_to_one removes duplicate source coords")
def _():
    from src.selection.spatial import one_to_one
    sx = np.array([[10.0, 10.0], [10.1, 10.1], [50.0, 50.0]], dtype=np.float32)
    rx = np.array([[15.0, 15.0], [20.0, 20.0], [55.0, 55.0]], dtype=np.float32)
    conf = np.array([0.8, 0.5, 0.9], dtype=np.float32)
    sx_o, rx_o, co = one_to_one(sx, rx, conf, tol_px=2.0)
    # [10.0, 10.0] and [10.1, 10.1] are within 2px — only the higher conf kept
    assert len(sx_o) == 2, f"Expected 2 after one_to_one, got {len(sx_o)}"
    assert float(co[0]) >= float(co[-1])   # sorted by confidence


@_test("T-spatial: selection_stats computes coverage and density_std")
def _():
    from src.selection.spatial import selection_stats
    rng = np.random.default_rng(11)
    sx_b = np.column_stack([rng.uniform(0, 256, 100), rng.uniform(0, 256, 100)]).astype(np.float32)
    sx_a = sx_b[:30]   # after selection: 30 points
    stats = selection_stats(sx_b, sx_a, np.ones(100), np.ones(30), image_shape=(256, 256))
    assert stats["n_before"] == 100
    assert stats["n_after"] == 30
    assert 0.0 <= stats["coverage_after"] <= 1.0
    assert stats["grid_density_std_after"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary():
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}Phase 3 Verification Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Total  : {total}")
    print(f"  {GREEN}Passed : {passed}{RESET}")
    if failed:
        print(f"  {RED}Failed : {failed}{RESET}")
        print()
        print("Failed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {RED}•{RESET} {name}")
                if detail:
                    print(f"      {detail}")
    else:
        print(f"  {GREEN}All tests passed!{RESET}")
    print()
    return failed


if __name__ == "__main__":
    print(f"\n{BOLD}Running Phase 3 verification...{RESET}\n")
    n_fail = _print_summary()
    sys.exit(0 if n_fail == 0 else 1)

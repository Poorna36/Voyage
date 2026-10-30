"""
quantum/benchmark.py
====================
Quantum vs. Classical Benchmarking Harness.

Runs quantum and classical solvers across multiple challenging conditions
(easy, noise, illumination, partial overlap, rotation, scale, repeated structure,
combined, real lunar crops) over multiple seeds.

Compares:
  - SIFT + Lowe ratio + RANSAC (sift_ratio_ransac)
  - SIFT without ratio test + RANSAC (sift_noratio_ransac)
  - Hungarian algorithm (qubo_v1 & qubo_v2)
  - Classical Simulated Annealing on QUBO (qubo_v1 & qubo_v2)
  - Simulated Quantum Annealing on QUBO (qubo_v1 & qubo_v2)
  - QUBO SQA solution + RANSAC filter (qubo_v1 & qubo_v2)
  - QAOA statevector scaling (M <= 16, dedicated small table)

Usage
-----
  python -m quantum.benchmark [--conditions easy,noise_10,...] [--n-seeds 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test data generator (Legacy & Advanced)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lunar_base(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic lunar regolith texture with craters."""
    base = rng.standard_normal((size, size)).astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter
        base = gaussian_filter(base, sigma=3.0)
    except ImportError:
        kernel = np.ones((5, 5), dtype=np.float32) / 25.0
        base = _convolve2d_simple(base, kernel)

    for _ in range(rng.integers(6, 18)):
        cx = rng.integers(12, size - 12)
        cy = rng.integers(12, size - 12)
        r = rng.integers(4, 24)
        Y, X = np.ogrid[:size, :size]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        base[mask] -= rng.uniform(0.6, 1.4)

    lo, hi = base.min(), base.max()
    return np.clip((base - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def _make_repeated_crater_base(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a surface with identical repeated craters on a grid (ambiguous descriptors)."""
    base = rng.standard_normal((size, size)).astype(np.float32) * 5.0 + 130.0
    grid_coords = [45, 95, 145, 195]
    crater_radius = 12
    Y, X = np.ogrid[:size, :size]
    for gx in grid_coords:
        for gy in grid_coords:
            cx = gx + int(rng.uniform(-2, 2))
            cy = gy + int(rng.uniform(-2, 2))
            dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
            # Dark floor
            mask_floor = dist_sq <= crater_radius ** 2
            base[mask_floor] -= 55.0
            # Bright rim on one side (sunlight simulation)
            mask_rim = (dist_sq <= (crater_radius + 3) ** 2) & (dist_sq > crater_radius ** 2) & (X < cx)
            base[mask_rim] += 45.0
    return np.clip(base, 0, 255).astype(np.uint8)


def _warp_image(
    img: np.ndarray, H: np.ndarray, out_shape: Tuple[int, int]
) -> np.ndarray:
    """Apply homography H to image."""
    try:
        import cv2
        return cv2.warpPerspective(
            img, H, (out_shape[1], out_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    except ImportError:
        H_inv = np.linalg.inv(H)
        H_out, W_out = out_shape
        out = np.zeros((H_out, W_out), dtype=img.dtype)
        for row in range(H_out):
            for col in range(W_out):
                src_h = H_inv @ np.array([col, row, 1.0])
                sx, sy = src_h[0] / src_h[2], src_h[1] / src_h[2]
                ix, iy = int(sx), int(sy)
                if 0 <= ix < img.shape[1] and 0 <= iy < img.shape[0]:
                    out[row, col] = img[iy, ix]
        return out


def _convolve2d_simple(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution via sliding window."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out


def _make_synthetic_pair(
    size: int = 256,
    n_true_matches: int = 30,
    rotation_deg: float = 8.0,
    scale: float = 1.15,
    tx: float = 12.0,
    ty: float = -8.0,
    noise_sigma: float = 1.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Legacy generator for backwards-compatibility with unit tests."""
    rng = np.random.default_rng(seed)
    src_img = _make_lunar_base(size, rng)

    cx, cy = size / 2.0, size / 2.0
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    H_true = np.array([
        [scale * cos_t, -scale * sin_t, cx + tx - scale * (cos_t * cx - sin_t * cy)],
        [scale * sin_t,  scale * cos_t, cy + ty - scale * (sin_t * cx + cos_t * cy)],
        [0.0,            0.0,           1.0],
    ], dtype=np.float64)

    ref_img = _warp_image(src_img, H_true, (size, size))

    margin = 20
    xs = rng.uniform(margin, size - margin, n_true_matches).astype(np.float32)
    ys = rng.uniform(margin, size - margin, n_true_matches).astype(np.float32)
    gt_src_xy = np.column_stack([xs, ys])

    ones = np.ones((n_true_matches, 1), dtype=np.float64)
    src_h = np.hstack([gt_src_xy.astype(np.float64), ones])
    ref_h = (H_true @ src_h.T).T
    gt_ref_xy = (ref_h[:, :2] / ref_h[:, 2:3]).astype(np.float32)

    in_bounds = (
        (gt_ref_xy[:, 0] >= 0) & (gt_ref_xy[:, 0] < size) &
        (gt_ref_xy[:, 1] >= 0) & (gt_ref_xy[:, 1] < size)
    )
    gt_src_xy = gt_src_xy[in_bounds]
    gt_ref_xy = gt_ref_xy[in_bounds]

    return {
        "src": src_img,
        "ref": ref_img,
        "H_true": H_true,
        "gt_src_xy": gt_src_xy,
        "gt_ref_xy": gt_ref_xy,
    }


def make_benchmark_pair(
    condition: str = "easy",
    seed: int = 42,
    size: int = 256,
) -> Dict[str, Any]:
    """
    Generate synthetic or real lunar image pairs with known ground-truth homography
    under 12 conditions designed to test matching robustness.
    """
    rng = np.random.default_rng(seed)

    if condition == "real_crops":
        crop_path = Path(__file__).resolve().parent.parent / "data" / "sample_images" / "img1_highlands.jpg"
        import cv2
        full_img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if full_img is None:
            base = _make_lunar_base(size, rng)
        else:
            h, w = full_img.shape
            offset_y = int((seed * 47) % max(1, h - size - 10))
            offset_x = int((seed * 73) % max(1, w - size - 10))
            base = full_img[offset_y : offset_y + size, offset_x : offset_x + size].copy()
    elif condition == "repeated_structure":
        base = _make_repeated_crater_base(size, rng)
    else:
        base = _make_lunar_base(size, rng)

    # Defaults
    rot_deg = 8.0
    scale = 1.15
    tx = 12.0
    ty = -8.0
    noise_sigma = 1.5
    src_gamma = 1.0
    ref_gamma = 1.0
    src_grad = False
    ref_grad = False

    if condition == "easy":
        pass
    elif condition == "noise_10":
        noise_sigma = 10.0
    elif condition == "noise_25":
        noise_sigma = 25.0
    elif condition == "illumination":
        src_grad = True
        ref_grad = True
        ref_gamma = 1.4
        src_gamma = 0.7
    elif condition == "partial_overlap":
        tx = size * 0.38
        ty = size * 0.22
        scale = 1.05
    elif condition == "rotation_45":
        rot_deg = 45.0
        scale = 1.0
        tx = 0.0
        ty = 0.0
    elif condition == "rotation_90":
        rot_deg = 90.0
        scale = 1.0
        tx = 0.0
        ty = 0.0
    elif condition == "scale_0.5":
        scale = 0.5
        rot_deg = 8.0
        tx = 10.0
        ty = -10.0
    elif condition == "scale_2.0":
        scale = 2.0
        rot_deg = 8.0
        tx = -20.0
        ty = 15.0
    elif condition == "repeated_structure":
        noise_sigma = 4.0
    elif condition == "combined":
        noise_sigma = 15.0
        src_grad = True
        ref_grad = True
        ref_gamma = 1.3
        rot_deg = 45.0
        scale = 1.1
        tx = size * 0.25
        ty = -size * 0.15
    elif condition == "real_crops":
        rot_deg = 15.0
        scale = 1.10
        tx = 10.0
        ty = -8.0
        noise_sigma = 5.0
    else:
        raise ValueError(f"Unknown condition: {condition}")

    cx, cy = size / 2.0, size / 2.0
    theta = np.radians(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    H_true = np.array([
        [scale * cos_t, -scale * sin_t, cx + tx - scale * (cos_t * cx - sin_t * cy)],
        [scale * sin_t,  scale * cos_t, cy + ty - scale * (sin_t * cx + cos_t * cy)],
        [0.0,            0.0,           1.0],
    ], dtype=np.float64)

    ref_clean = _warp_image(base, H_true, (size, size))
    src_f = base.astype(np.float32)
    ref_f = ref_clean.astype(np.float32)

    if src_grad:
        gx = np.linspace(0.6, 1.3, size, dtype=np.float32)[None, :]
        src_f = src_f * gx
    if ref_grad:
        gy = np.linspace(1.3, 0.7, size, dtype=np.float32)[:, None]
        ref_f = ref_f * gy

    if src_gamma != 1.0:
        src_f = 255.0 * np.power(np.clip(src_f / 255.0, 1e-4, 1.0), src_gamma)
    if ref_gamma != 1.0:
        ref_f = 255.0 * np.power(np.clip(ref_f / 255.0, 1e-4, 1.0), ref_gamma)

    if noise_sigma > 0:
        src_f += rng.normal(0, noise_sigma, (size, size)).astype(np.float32)
        ref_f += rng.normal(0, noise_sigma, (size, size)).astype(np.float32)

    src_img = np.clip(src_f, 0, 255).astype(np.uint8)
    ref_img = np.clip(ref_f, 0, 255).astype(np.uint8)

    # Estimate geometrically visible keypoint count
    try:
        import cv2
        sift = cv2.SIFT_create()
        kp1 = sift.detect(src_img, None)
        if kp1:
            pts = np.array([k.pt for k in kp1], dtype=np.float64)
            ones = np.ones((len(pts), 1))
            pts_h = np.hstack([pts, ones])
            w_h = (H_true @ pts_h.T).T
            w_pts = w_h[:, :2] / (w_h[:, 2:3] + 1e-12)
            in_b = (w_pts[:, 0] >= 0) & (w_pts[:, 0] < size) & (w_pts[:, 1] >= 0) & (w_pts[:, 1] < size)
            n_possible = int(in_b.sum())
        else:
            n_possible = 25
    except Exception:
        n_possible = 25

    return {
        "src": src_img,
        "ref": ref_img,
        "H_true": H_true,
        "scale": scale,
        "rotation_deg": rot_deg,
        "condition": condition,
        "seed": seed,
        "n_possible": max(1, n_possible),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(
    pred_src: np.ndarray, pred_ref: np.ndarray, gt_src: np.ndarray, gt_ref: np.ndarray
) -> Tuple[float, int, int]:
    """Legacy gated nearest-GT RMSE."""
    if len(pred_src) == 0 or len(gt_src) == 0:
        return float("inf"), 0, len(gt_src)

    r_max = 5.0
    errors = []
    for g_src, g_ref in zip(gt_src, gt_ref):
        dists_src = np.linalg.norm(pred_src - g_src, axis=1)
        nearest_k = np.argmin(dists_src)
        if dists_src[nearest_k] > r_max:
            continue
        err = np.linalg.norm(pred_ref[nearest_k] - g_ref)
        errors.append(err)

    if not errors:
        return float("inf"), 0, len(gt_src)
    return float(np.sqrt(np.mean(np.array(errors) ** 2))), len(errors), len(gt_src)


def eval_matches(
    src_xy: np.ndarray, ref_xy: np.ndarray, H_true: np.ndarray, thr: float = 3.0
) -> Dict[str, Any]:
    """Direct evaluation of correspondences against H_true."""
    if len(src_xy) == 0:
        return dict(n=0, inliers=0, precision=0.0, median_err=float("inf"))
    p = np.hstack([src_xy, np.ones((len(src_xy), 1))]) @ H_true.T
    err = np.linalg.norm(p[:, :2] / (p[:, 2:3] + 1e-12) - ref_xy, axis=1)
    inl = err < thr
    return dict(
        n=int(len(err)),
        inliers=int(inl.sum()),
        precision=float(inl.mean()),
        median_err=float(np.median(err)) if len(err) > 0 else float("inf"),
    )


def eval_matches_comprehensive(
    src_xy: np.ndarray,
    ref_xy: np.ndarray,
    H_true: np.ndarray,
    size: int = 256,
    thr: float = 3.0,
    n_possible: int = 30,
) -> Dict[str, Any]:
    """Comprehensive correspondence & homography evaluation."""
    n = len(src_xy)
    if n == 0:
        return {
            "n_matches": 0,
            "inliers": 0,
            "precision": 0.0,
            "recall": 0.0,
            "median_err": float("inf"),
            "homography_rmse": float("inf"),
            "failed": True,
        }

    ones = np.ones((n, 1), dtype=np.float64)
    src_h = np.hstack([src_xy.astype(np.float64), ones])
    true_ref_h = (H_true @ src_h.T).T
    true_ref = true_ref_h[:, :2] / (true_ref_h[:, 2:3] + 1e-12)

    errors = np.linalg.norm(true_ref - ref_xy, axis=1)
    inl = errors < thr
    n_inliers = int(inl.sum())
    precision = float(n_inliers / n)
    recall = float(n_inliers / max(1, n_possible))
    med_err = float(np.median(errors))

    # Homography RMSE after RANSAC if at least 4 inliers
    h_rmse = float("inf")
    if n_inliers >= 4:
        try:
            import cv2
            H_est, mask = cv2.findHomography(src_xy, ref_xy, cv2.RANSAC, thr)
            if H_est is not None:
                gx = np.linspace(20, size - 20, 10)
                gy = np.linspace(20, size - 20, 10)
                grid_x, grid_y = np.meshgrid(gx, gy)
                pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])

                t_h = (H_true @ pts_h.T).T
                t_ref = t_h[:, :2] / (t_h[:, 2:3] + 1e-12)

                e_h = (H_est @ pts_h.T).T
                e_ref = e_h[:, :2] / (e_h[:, 2:3] + 1e-12)

                in_ref = (
                    (t_ref[:, 0] >= 0) & (t_ref[:, 0] < size) &
                    (t_ref[:, 1] >= 0) & (t_ref[:, 1] < size)
                )
                if in_ref.sum() > 0:
                    h_err = np.linalg.norm(t_ref[in_ref] - e_ref[in_ref], axis=1)
                    h_rmse = float(np.sqrt(np.mean(h_err ** 2)))
        except Exception:
            h_rmse = float("inf")

    failed = (n_inliers < 4) or (h_rmse == float("inf"))

    return {
        "n_matches": n,
        "inliers": n_inliers,
        "precision": precision,
        "recall": recall,
        "median_err": med_err,
        "homography_rmse": h_rmse,
        "failed": failed,
    }


def _constraint_violation(solution: np.ndarray, pairs: np.ndarray) -> float:
    """Compute one-to-one constraint violation rate."""
    if solution is None or len(solution) == 0 or pairs is None or len(pairs) == 0:
        return 0.0
    selected = pairs[solution == 1]
    if len(selected) == 0:
        return 0.0

    n_dup_src = len(selected) - len(np.unique(selected[:, 0]))
    n_dup_ref = len(selected) - len(np.unique(selected[:, 1]))
    total = len(selected)
    return float((n_dup_src + n_dup_ref) / (2 * total + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-condition & Multi-seed Benchmark Harness
# ─────────────────────────────────────────────────────────────────────────────

ALL_CONDITIONS = [
    "easy",
    "noise_10",
    "noise_25",
    "illumination",
    "partial_overlap",
    "rotation_45",
    "rotation_90",
    "scale_0.5",
    "scale_2.0",
    "repeated_structure",
    "combined",
    "real_crops",
]

ALL_METHODS = [
    "sift_ratio_ransac",
    "sift_noratio_ransac",
    "hungarian_v1",
    "hungarian_v2",
    "classical_sa_v1",
    "classical_sa_v2",
    "sqa_v1",
    "sqa_v2",
    "qubo_v1_ransac",
    "qubo_v2_ransac",
]


class QuantumBenchmark:
    """
    Comprehensive benchmark runner comparing quantum and classical solvers
    across multiple conditions and seeds.
    """

    def __init__(
        self,
        image_size: int = 256,
        n_true_matches: int = 25,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.n_true_matches = n_true_matches
        self.seed = seed

    def run(
        self,
        n_kp: int = 60,
        solvers: Optional[List[str]] = None,
        rotation_deg: float = 8.0,
        scale: float = 1.15,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Legacy single-run benchmark for backward compatibility."""
        if solvers is None:
            solvers = ["sift_ransac", "classical", "classical_sa", "sqa"]

        pair = _make_synthetic_pair(
            size=self.image_size,
            n_true_matches=self.n_true_matches,
            rotation_deg=rotation_deg,
            scale=scale,
            seed=self.seed,
        )

        results = {}
        for solver_name in solvers:
            if solver_name == "sift_ransac":
                from .classical_baseline import sift_ransac_baseline
                t0 = time.time()
                sr_res = sift_ransac_baseline(pair["src"], pair["ref"])
                t_total = time.time() - t0
                src_xy = sr_res["src_xy"]
                ref_xy = sr_res["ref_xy"]
                ev = eval_matches(src_xy, ref_xy, pair["H_true"])
                rmse, n_gated, n_gt = _rmse(src_xy, ref_xy, pair["gt_src_xy"], pair["gt_ref_xy"])
                results[solver_name] = {
                    "n_matches": len(src_xy),
                    "inliers": ev["inliers"],
                    "precision": ev["precision"],
                    "median_err_px": ev["median_err"],
                    "rmse_px": rmse,
                    "rmse_gated_points": n_gated,
                    "rmse_total_gt_points": n_gt,
                    "runtime_s": t_total,
                    "qubo_energy": None,
                    "qubo_n_vars": 0,
                    "constraint_violation": 0.0,
                    "solver_string": "sift_ransac",
                }
                continue

            from .matcher_qubo import QUBOMatcher
            cfg = {
                "solver": solver_name,
                "num_keypoints": n_kp,
                "sqa_sweeps": 400,
                "sqa_reads": 2,
            }
            matcher = QUBOMatcher(config=cfg)
            t0 = time.time()
            res = matcher.match(pair["src"], pair["ref"], gsd_ratio=scale)
            t_total = time.time() - t0

            src_xy = res.src_xy
            ref_xy = res.ref_xy
            ev = eval_matches(src_xy, ref_xy, pair["H_true"])
            rmse, n_gated, n_gt = _rmse(src_xy, ref_xy, pair["gt_src_xy"], pair["gt_ref_xy"])
            params = res.matcher_params
            cv_rate = _constraint_violation(params.get("qubo_solution"), params.get("qubo_pairs"))

            results[solver_name] = {
                "n_matches": res.count,
                "inliers": ev["inliers"],
                "precision": ev["precision"],
                "median_err_px": ev["median_err"],
                "rmse_px": rmse,
                "rmse_gated_points": n_gated,
                "rmse_total_gt_points": n_gt,
                "runtime_s": t_total,
                "qubo_energy": params.get("qubo_energy"),
                "linear_cost_energy": params.get("solver_meta", {}).get("linear_cost_energy"),
                "qubo_n_vars": params.get("qubo_n_vars"),
                "constraint_violation": cv_rate,
                "solver_string": params.get("solver_meta", {}).get("solver", solver_name),
            }

        report = {
            "benchmark": {
                "image_size": self.image_size,
                "n_gt_matches": len(pair["gt_src_xy"]),
                "n_kp_per_image": n_kp,
                "rotation_deg": rotation_deg,
                "scale": scale,
                "seed": self.seed,
            },
            "results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if verbose:
            self.print_report(report)
        return report

    @staticmethod
    def run_multi_seed_condition(
        condition: str,
        n_seeds: int = 10,
        n_kp: int = 60,
        methods: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all methods across n_seeds for a single condition and compute mean ± std.
        """
        import cv2
        from .matcher_qubo import QUBOMatcher
        from .classical_baseline import sift_ransac_baseline

        if methods is None:
            methods = ALL_METHODS

        accum: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}

        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 17
            pair = make_benchmark_pair(condition=condition, seed=seed)
            src, ref, H_true = pair["src"], pair["ref"], pair["H_true"]
            scale = pair["scale"]
            n_possible = pair["n_possible"]

            # Cache keypoints & descriptors if needed for speed
            # Run each method
            for m in methods:
                t0 = time.time()
                cv_rate = 0.0

                if m == "sift_ratio_ransac":
                    sr = sift_ransac_baseline(src, ref, ratio_thresh=0.75)
                    src_xy, ref_xy = sr["src_xy"], sr["ref_xy"]
                    runtime = time.time() - t0

                elif m == "sift_noratio_ransac":
                    sr = sift_ransac_baseline(src, ref, ratio_thresh=1.0)
                    src_xy, ref_xy = sr["src_xy"], sr["ref_xy"]
                    runtime = time.time() - t0

                elif m.startswith("hungarian"):
                    v = "v2" if "v2" in m else "v1"
                    matcher = QUBOMatcher({
                        "solver": "classical",
                        "formulation_version": v,
                        "num_keypoints": n_kp,
                        "max_pairs": 60,
                    })
                    res = matcher.match(src, ref, gsd_ratio=scale)
                    src_xy, ref_xy = res.src_xy, res.ref_xy
                    runtime = time.time() - t0
                    cv_rate = _constraint_violation(
                        res.matcher_params.get("qubo_solution"),
                        res.matcher_params.get("qubo_pairs"),
                    )

                elif m.startswith("classical_sa"):
                    v = "v2" if "v2" in m else "v1"
                    matcher = QUBOMatcher({
                        "solver": "sa_classical",
                        "formulation_version": v,
                        "num_keypoints": n_kp,
                        "max_pairs": 60,
                        "sqa_sweeps": 250,
                        "sqa_reads": 2,
                    })
                    res = matcher.match(src, ref, gsd_ratio=scale)
                    src_xy, ref_xy = res.src_xy, res.ref_xy
                    runtime = time.time() - t0
                    cv_rate = _constraint_violation(
                        res.matcher_params.get("qubo_solution"),
                        res.matcher_params.get("qubo_pairs"),
                    )

                elif m.startswith("sqa"):
                    v = "v2" if "v2" in m else "v1"
                    matcher = QUBOMatcher({
                        "solver": "sqa",
                        "formulation_version": v,
                        "num_keypoints": n_kp,
                        "max_pairs": 60,
                        "sqa_sweeps": 200,
                        "sqa_reads": 2,
                        "sqa_trotter": 8,
                    })
                    res = matcher.match(src, ref, gsd_ratio=scale)
                    src_xy, ref_xy = res.src_xy, res.ref_xy
                    runtime = time.time() - t0
                    cv_rate = _constraint_violation(
                        res.matcher_params.get("qubo_solution"),
                        res.matcher_params.get("qubo_pairs"),
                    )

                elif m.startswith("qubo_v1_ransac") or m.startswith("qubo_v2_ransac"):
                    v = "v2" if "v2" in m else "v1"
                    matcher = QUBOMatcher({
                        "solver": "sqa",
                        "formulation_version": v,
                        "num_keypoints": n_kp,
                        "max_pairs": 60,
                        "sqa_sweeps": 200,
                        "sqa_reads": 2,
                        "sqa_trotter": 8,
                    })
                    res = matcher.match(src, ref, gsd_ratio=scale)
                    cv_rate = _constraint_violation(
                        res.matcher_params.get("qubo_solution"),
                        res.matcher_params.get("qubo_pairs"),
                    )
                    # Apply RANSAC filter on QUBO solution
                    if len(res.src_xy) >= 4:
                        H_est, mask = cv2.findHomography(res.src_xy, res.ref_xy, cv2.RANSAC, 3.0)
                        if mask is not None:
                            in_mask = mask.ravel() == 1
                            src_xy = res.src_xy[in_mask]
                            ref_xy = res.ref_xy[in_mask]
                        else:
                            src_xy, ref_xy = res.src_xy, res.ref_xy
                    else:
                        src_xy, ref_xy = res.src_xy, res.ref_xy
                    runtime = time.time() - t0
                else:
                    continue

                ev = eval_matches_comprehensive(
                    src_xy, ref_xy, H_true, size=256, thr=3.0, n_possible=n_possible
                )
                ev["runtime_s"] = runtime
                ev["constraint_violation"] = cv_rate
                accum[m].append(ev)

        # Aggregate across seeds: mean and std
        summary = {}
        for m, run_list in accum.items():
            if not run_list:
                continue
            n_m = [r["n_matches"] for r in run_list]
            inl = [r["inliers"] for r in run_list]
            prec = [r["precision"] * 100.0 for r in run_list]
            rec = [r["recall"] * 100.0 for r in run_list]
            med = [r["median_err"] for r in run_list if r["median_err"] != float("inf")]
            rmse_vals = [r["homography_rmse"] for r in run_list if r["homography_rmse"] != float("inf")]
            cv_vals = [r["constraint_violation"] * 100.0 for r in run_list]
            rt = [r["runtime_s"] for r in run_list]
            fails = sum(1 for r in run_list if r["failed"])

            summary[m] = {
                "matches": (float(np.mean(n_m)), float(np.std(n_m))),
                "inliers": (float(np.mean(inl)), float(np.std(inl))),
                "precision": (float(np.mean(prec)), float(np.std(prec))),
                "recall": (float(np.mean(rec)), float(np.std(rec))),
                "median_err": (float(np.mean(med)) if med else float("inf"), float(np.std(med)) if med else 0.0),
                "homography_rmse": (float(np.mean(rmse_vals)) if rmse_vals else float("inf"), float(np.std(rmse_vals)) if rmse_vals else 0.0),
                "constraint_violation": (float(np.mean(cv_vals)), float(np.std(cv_vals))),
                "runtime_s": (float(np.mean(rt)), float(np.std(rt))),
                "failures": fails,
                "n_seeds": n_seeds,
            }

        return summary

    @staticmethod
    def run_qaoa_scaling(
        qubit_sizes: Optional[List[int]] = None,
        p_layers: int = 2,
        n_restarts: int = 5,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate QAOA statevector solver scaling on M in [4, 6, 8, 10, 12, 14, 16].
        Reports approximation ratio and ground-state probability.
        """
        from .qaoa import QAOASolver

        if qubit_sizes is None:
            qubit_sizes = [4, 6, 8, 10, 12, 14, 16]

        rows = []
        for M in qubit_sizes:
            rng = np.random.default_rng(seed + M)
            Q = rng.standard_normal((M, M))
            Q = (Q + Q.T) / 2.0

            solver = QAOASolver(
                p_layers=p_layers,
                n_optimizer_restarts=n_restarts,
                max_iter=60,
                seed=seed,
            )
            res = solver.solve(Q)
            rows.append({
                "M": M,
                "p": p_layers,
                "restarts": n_restarts,
                "approx_ratio": res["approximation_ratio"],
                "p_ground_state": res["ground_state_probability"],
                "expected_energy": res["expected_energy"],
                "ground_energy": res["ground_state_energy"],
                "runtime_s": res["runtime_s"],
            })
        return rows

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print standard single-run report."""
        print("\n" + "=" * 105)
        print("  BENCHMARK RESULTS SUMMARY")
        print("=" * 105)
        header = f"  {'Solver':<18} {'Matches':>8} {'Inliers':>8} {'Prec (%)':>9} {'MedErr':>9} {'GatedRMSE':>16} {'Energy':>11} {'Viol (%)':>9} {'Runtime':>9}"
        print(header)
        print("  " + "-" * 103)
        for solver, res in report["results"].items():
            rmse = res.get("rmse_px", float("inf"))
            rmse_s = f"{rmse:.2f}" if rmse != float("inf") else "N/A"
            n_g = res.get("rmse_gated_points", 0)
            n_tot = res.get("rmse_total_gt_points", 0)
            gated_s = f"{rmse_s} ({n_g}/{n_tot})" if n_g > 0 else "N/A"
            energy = res.get("qubo_energy")
            energy_s = f"{energy:.1f}" if energy is not None else "N/A"
            prec = res.get("precision", 0.0) * 100.0
            med_err = res.get("median_err_px", float("inf"))
            med_s = f"{med_err:.2f}" if med_err != float("inf") else "inf"
            cv = res.get("constraint_violation", 0.0) * 100.0
            solver_label = res.get("solver_string", solver)
            print(
                f"  {solver_label:<18} {res['n_matches']:>8} {res.get('inliers', 0):>8} "
                f"{prec:>8.1f}% {med_s:>9} {gated_s:>16} {energy_s:>11} {cv:>8.1f}% {res['runtime_s']:>8.2f}s"
            )
        print("=" * 105 + "\n")

    @staticmethod
    def save_report(report: Dict[str, Any], path: str) -> None:
        """Save benchmark report as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        def default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.float32, np.float64, np.floating)):
                return float(o)
            if isinstance(o, (np.int32, np.int64, np.integer)):
                return int(o)
            raise TypeError(f"Not serialisable: {type(o)}")

        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=default)
        print(f"Benchmark report saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum QUBO Correspondence Benchmark (Voyage / SIH PS-26166)"
    )
    parser.add_argument(
        "--conditions", type=str, default="easy,noise_10,noise_25,illumination,partial_overlap,rotation_45,rotation_90,scale_0.5,scale_2.0,repeated_structure,combined,real_crops",
        help="Comma-separated list of conditions to benchmark",
    )
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds per condition")
    parser.add_argument("--n-kp", type=int, default=50, help="SIFT keypoints per image")
    parser.add_argument("--out", type=str, default="quantum/results/benchmark_hard.json")
    parser.add_argument("--qaoa", action="store_true", default=False, help="Run QAOA scaling benchmark")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print("\n" + "=" * 90)
    print("  VOYAGE QUANTUM CORRESPONDENCE BENCHMARK")
    print(f"  Conditions: {len(conditions)}, Seeds per condition: {args.n_seeds}")
    print("=" * 90)

    all_results = {}
    for cond in conditions:
        print(f"\n---> Running condition: [{cond}] over {args.n_seeds} seeds ...")
        t_start = time.time()
        summary = QuantumBenchmark.run_multi_seed_condition(
            condition=cond,
            n_seeds=args.n_seeds,
            n_kp=args.n_kp,
        )
        t_elapsed = time.time() - t_start
        all_results[cond] = summary

        # Print table for this condition
        print(f"\nCondition: {cond} (completed in {t_elapsed:.1f}s)")
        print(f"{'Method':<22} {'Matches':>12} {'Inliers':>12} {'Prec (%)':>14} {'Recall (%)':>14} {'MedErr (px)':>14} {'H_RMSE (px)':>14} {'Viol (%)':>12} {'Fails':>6}")
        print("-" * 125)
        for m, s in summary.items():
            m_s = f"{s['matches'][0]:.1f}±{s['matches'][1]:.1f}"
            inl_s = f"{s['inliers'][0]:.1f}±{s['inliers'][1]:.1f}"
            p_s = f"{s['precision'][0]:.1f}±{s['precision'][1]:.1f}%"
            rec_s = f"{s['recall'][0]:.1f}±{s['recall'][1]:.1f}%"
            med_s = f"{s['median_err'][0]:.2f}±{s['median_err'][1]:.2f}" if s['median_err'][0] != float("inf") else "inf"
            h_s = f"{s['homography_rmse'][0]:.2f}±{s['homography_rmse'][1]:.2f}" if s['homography_rmse'][0] != float("inf") else "inf"
            cv_s = f"{s['constraint_violation'][0]:.1f}±{s['constraint_violation'][1]:.1f}%"
            fail_s = f"{s['failures']}/{s['n_seeds']}"
            print(f"{m:<22} {m_s:>12} {inl_s:>12} {p_s:>14} {rec_s:>14} {med_s:>14} {h_s:>14} {cv_s:>12} {fail_s:>6}")

    if args.qaoa:
        print("\n---> Running QAOA statevector scaling benchmark (M <= 16) ...")
        qaoa_res = QuantumBenchmark.run_qaoa_scaling()
        all_results["qaoa_scaling"] = qaoa_res
        print("\nQAOA Scaling Table:")
        print(f"{'M (qubits)':<12} {'p':>4} {'Restarts':>10} {'Approx Ratio':>14} {'P(Ground State)':>18} {'Expected E':>12} {'GS Energy':>12} {'Time (s)':>10}")
        print("-" * 92)
        for row in qaoa_res:
            print(f"{row['M']:<12} {row['p']:>4} {row['restarts']:>10} {row['approx_ratio']:>14.4f} {row['p_ground_state']:>18.6f} {row['expected_energy']:>12.3f} {row['ground_energy']:>12.3f} {row['runtime_s']:>10.3f}")

    QuantumBenchmark.save_report(all_results, args.out)


if __name__ == "__main__":
    main()

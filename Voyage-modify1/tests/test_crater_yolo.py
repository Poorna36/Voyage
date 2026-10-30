"""
tests/test_crater_yolo.py
==========================
Unit & Integration Tests for YOLO-backed M3 CraterMatcher.

Verifies:
1. YOLO weights loading and default configuration.
2. Multi-Scale Scale-Space crater detection on real lunar imagery.
3. End-to-end CNSF graph topology matching.
4. Graceful CPU/Hough fallback when weights or GPU are missing.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.matching.crater import CraterMatcher


def test_crater_matcher_defaults():
    """Verify CraterMatcher initializes with yolo_weights when present."""
    matcher = CraterMatcher()
    weights_path = Path("models/crater_yolov9.pt")
    if weights_path.exists():
        assert matcher.yolo_weights == "models/crater_yolov9.pt"
    assert matcher.tau_c == 3.0
    assert matcher.conf_thresh == 0.65


def test_yolo_crater_detection_on_highlands():
    """Verify _detect_yolo detects craters on real highland imagery."""
    img_path = Path("data/sample_images/img1_highlands.jpg")
    if not img_path.exists():
        pytest.skip("img1_highlands.jpg not found")

    img = cv2.imread(str(img_path))
    matcher = CraterMatcher({"yolo_weights": "models/crater_yolov9.pt"})

    craters = matcher._detect_craters(img)
    assert len(craters) > 10, f"Expected >10 craters, got {len(craters)}"
    assert craters.shape[1] == 3  # (col, row, radius)
    try:
        import ultralytics
        assert matcher._last_used_detector == "yolo"
    except ImportError:
        assert matcher._last_used_detector == "hough"

    # Check bounds
    H, W = img.shape[:2]
    assert np.all(craters[:, 0] >= 0) and np.all(craters[:, 0] <= W)
    assert np.all(craters[:, 1] >= 0) and np.all(craters[:, 1] <= H)
    assert np.all(craters[:, 2] > 0)


def test_crater_matcher_end_to_end():
    """Verify end-to-end CNSF topology matching between an image and its transform."""
    img_path = Path("data/sample_images/img1_highlands.jpg")
    if not img_path.exists():
        pytest.skip("img1_highlands.jpg not found")

    src = cv2.imread(str(img_path))
    H, W = src.shape[:2]

    # Create reference image with a known translation + slight rotation
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 2.0, 1.0)
    M[0, 2] += 20.0
    M[1, 2] += 15.0
    ref = cv2.warpAffine(src, M, (W, H))

    matcher = CraterMatcher({
        "yolo_weights": "models/crater_yolov9.pt",
        "match_confidence_thresh": 0.50,
        "topology_max_craters": 60,
    })

    result = matcher.match(
        src=src,
        ref=ref,
        gsd_ratio=1.0,
        crater_density_src=5.0,
        crater_density_ref=5.0,
        terrain_src="highland",
        terrain_ref="highland",
    )

    assert result.matcher_params["gate_skip"] is False
    try:
        import ultralytics
        assert result.matcher_params["matcher_id"] == "crater"
        assert result.matcher_params["cpu_fallback"] is False
    except ImportError:
        assert result.matcher_params["matcher_id"] in ("crater", "crater_hough")
    assert result.matcher_params["n_craters_src"] > 0
    assert result.matcher_params["n_craters_ref"] > 0
    assert len(result.src_xy) > 0
    assert result.src_xy.shape[1] == 2
    assert result.ref_xy.shape[1] == 2


def test_graceful_fallback_when_weights_missing():
    """Verify CraterMatcher falls back to HoughCircles if weights file is missing."""
    matcher = CraterMatcher({"yolo_weights": "models/nonexistent_model.pt"})

    # Synthetic circle image
    img = np.full((300, 300), 128, dtype=np.uint8)
    cv2.circle(img, (100, 100), 30, 200, 2)
    cv2.circle(img, (200, 200), 25, 50, 2)

    craters = matcher._detect_craters(img)
    assert matcher._last_used_detector == "hough"

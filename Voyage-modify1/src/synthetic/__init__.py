"""
src/synthetic — Synthetic Ground-Truth Benchmark v3.0

Provides:
  anchors    — GT anchor extraction (Shi-Tomasi grid + morphological stratification)
  transforms — Physical transformation engine (scale, translation, rotation,
               illumination, sensor MTF/pushbroom simulation)

Usage:
  from src.synthetic.anchors import extract_anchors
  from src.synthetic.transforms import build_transform_matrix, apply_transform, transform_gt_points

Coordinate convention: all pixel coordinates are (col, row) = (x, y), 0-indexed.
Ground truth points are NEVER passed to the matcher — only to the evaluation engine.
"""
from src.synthetic.anchors import extract_anchors, AnchorSet
from src.synthetic.transforms import (
    build_transform_matrix,
    apply_transform,
    transform_gt_points,
    TransformParams,
)

__all__ = [
    "extract_anchors",
    "AnchorSet",
    "build_transform_matrix",
    "apply_transform",
    "transform_gt_points",
    "TransformParams",
]

"""
synthetic — Synthetic Lunar Dataset & Benchmark Generation Package
==================================================================

Standalone module for synthesizing Chandrayaan-2 vs LRO lunar imagery pairs,
extracting exact ground-truth anchors, and applying orbital sensor transformations.

Modules:
  anchors          — GT anchor extraction (Shi-Tomasi grid + morphological stratification)
  transforms       — Physical transformation engine (scale, rotation, translation, illumination, MTF, pushbroom)
  scene_generator  — Realistic multi-scale synthetic lunar terrain synthesis
  generate         — Benchmark generator CLI (Phases 1–4)
  evaluate         — Component-wise stage evaluation CLI
  runner           — End-to-end benchmark orchestrator
"""

from synthetic.anchors import extract_anchors, AnchorPoint, AnchorSet
from synthetic.transforms import (
    build_transform_matrix,
    apply_transform,
    transform_gt_points,
    generate_synthetic_pair,
    apply_illumination_gamma,
    apply_mtf_blur,
    apply_pushbroom_noise,
    apply_shadow_extension,
    TransformParams,
)
from synthetic.scene_generator import generate_synthetic_lunar_scene

__all__ = [
    "extract_anchors",
    "AnchorPoint",
    "AnchorSet",
    "build_transform_matrix",
    "apply_transform",
    "transform_gt_points",
    "generate_synthetic_pair",
    "apply_illumination_gamma",
    "apply_mtf_blur",
    "apply_pushbroom_noise",
    "apply_shadow_extension",
    "TransformParams",
    "generate_synthetic_lunar_scene",
]

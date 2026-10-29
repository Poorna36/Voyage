"""
src/selector/__init__.py
========================
Public API for the Matcher Selection Model (MSM) (L1.5 layer).

Provides:
  - MSMFeatureVector: Dataclass representation of the 13-feature vector
  - extract_features: Extractor from PairRecord + meta.json
  - vectorize_features: Array serializer
  - hash_features: Deterministic MD5 hash generator
  - SelectorResult: Dataclass representation of selection inference output
  - MatcherSelector: Inference engine with dual-threshold routing and hard gating

References:
  - FEATURES.md F26, F27
  - INTERFACES.md §10
  - ARCHITECTURE.md §3 (L1.5)
  - PROGRESS.md Phase 5.5
"""
from src.selector.features import (
    MSMFeatureVector,
    extract_features,
    vectorize_features,
    hash_features,
    FEATURE_NAMES,
    SENSOR_PAIR_MAP,
    TERRAIN_CLASS_MAP,
)
from src.selector.model import (
    SelectorResult,
    MatcherSelector,
    MATCHER_NAMES,
    INDEX_TO_MATCHER,
    MATCHER_TO_INDEX,
)

__all__ = [
    "MSMFeatureVector",
    "extract_features",
    "vectorize_features",
    "hash_features",
    "FEATURE_NAMES",
    "SENSOR_PAIR_MAP",
    "TERRAIN_CLASS_MAP",
    "SelectorResult",
    "MatcherSelector",
    "MATCHER_NAMES",
    "INDEX_TO_MATCHER",
    "MATCHER_TO_INDEX",
]

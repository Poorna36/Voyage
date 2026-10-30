"""
src/matching/base.py
====================
Base classes for the correspondence matching layer (L2).

All coordinates are (col, row) = (x, y), 0-indexed top-left origin.
NEVER (row, col). Assertions enforce this at every call boundary.

References
----------
- ARCHITECTURE.md §4  (BaseMatcher contract)
- INTERFACES.md   §9  (MatchResult schema)
- FEATURES.md     F09 (matcher plug-in system)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# MatchResult — per INTERFACES.md §9
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Output schema for all matchers (INTERFACES.md §9).

    Fields
    ------
    src_xy     : (N, 2) float32 — col, row coords in source image
    ref_xy     : (N, 2) float32 — col, row coords in reference image
    confidence : (N,)   float32 — per-match confidence in [0, 1]
    scale      : (N,)   float32 or None — scale ratio src_kp_size / ref_kp_size
    angle_deg  : (N,)   float32 or None — rotation angle of match (degrees)
    runtime_s  : wall-clock seconds for the match() call
    matcher_params : dict with at least 'matcher_id'; carries provenance
    """

    src_xy: np.ndarray
    ref_xy: np.ndarray
    confidence: np.ndarray
    scale: Optional[np.ndarray] = None
    angle_deg: Optional[np.ndarray] = None
    runtime_s: float = 0.0
    matcher_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coordinate convention guard — ARCHITECTURE.md hard constraint
        assert self.src_xy.ndim == 2 and self.src_xy.shape[-1] == 2, (
            "Expected (N,2) array: (col, row) — src_xy shape was "
            f"{self.src_xy.shape}"
        )
        assert self.ref_xy.ndim == 2 and self.ref_xy.shape[-1] == 2, (
            "Expected (N,2) array: (col, row) — ref_xy shape was "
            f"{self.ref_xy.shape}"
        )
        n = len(self.src_xy)
        assert len(self.ref_xy) == n, "src_xy and ref_xy must have the same length."
        assert len(self.confidence) == n, "confidence length must match src_xy."
        # Normalise dtype
        self.src_xy = np.asarray(self.src_xy, dtype=np.float32)
        self.ref_xy = np.asarray(self.ref_xy, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        if self.scale is not None:
            self.scale = np.asarray(self.scale, dtype=np.float32)
        if self.angle_deg is not None:
            self.angle_deg = np.asarray(self.angle_deg, dtype=np.float32)

    @property
    def count(self) -> int:
        """Number of matched point pairs."""
        return len(self.src_xy)

    def is_empty(self) -> bool:
        return self.count == 0

    def __repr__(self) -> str:
        mid = self.matcher_params.get("matcher_id", "?")
        return (
            f"MatchResult(matcher={mid!r}, n={self.count}, "
            f"runtime={self.runtime_s:.3f}s)"
        )


# ---------------------------------------------------------------------------
# BaseMatcher — abstract plug-in interface
# ---------------------------------------------------------------------------

class BaseMatcher(ABC):
    """
    Abstract base class for all correspondence matchers (M0-M3).

    Contract (ARCHITECTURE.md §4)
    ------------------------------
    - matcher_id   : unique lowercase string (e.g. 'sift', 'rift2')
    - requires_gpu : bool property; defaults False
    - match()      : MUST return valid MatchResult even on failure.
                     MUST NEVER propagate an exception to the caller.
    - All returned coordinates are (col, row) — x-first convention.
    """

    @property
    @abstractmethod
    def matcher_id(self) -> str:
        """Unique string identifier for this matcher."""
        ...

    @property
    def requires_gpu(self) -> bool:
        """Whether this matcher requires a CUDA GPU. Default False."""
        return False

    @abstractmethod
    def match(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        gsd_ratio: float = 1.0,
        **kwargs: Any,
    ) -> MatchResult:
        """
        Find correspondences between *src* and *ref*.

        Parameters
        ----------
        src, ref    : uint8 (H,W) or (H,W,3), or float32 normalised images
        gsd_ratio   : src_gsd_m / ref_gsd_m — used by scale-consistency filter.
                      Scale-consistency filter (ARCHITECTURE.md §4, FEATURES.md F10):
                        reject if |log(scale_src/scale_ref) - log(gsd_ratio)| > 0.3
        Returns
        -------
        MatchResult with (col, row) coords. Empty MatchResult on any failure.
        """
        ...

    # ── Helpers available to all subclasses ──────────────────────────────────

    def _empty_result(
        self,
        runtime_s: float = 0.0,
        reason: str = "",
        **extra: Any,
    ) -> MatchResult:
        """Return a well-typed empty MatchResult on failure."""
        params: Dict[str, Any] = {"matcher_id": self.matcher_id}
        if reason:
            params["failure_reason"] = reason
        params.update(extra)
        return MatchResult(
            src_xy=np.empty((0, 2), dtype=np.float32),
            ref_xy=np.empty((0, 2), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            scale=np.empty(0, dtype=np.float32),
            angle_deg=np.empty(0, dtype=np.float32),
            runtime_s=runtime_s,
            matcher_params=params,
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert any image array to single-channel uint8."""
        try:
            import cv2
            if img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except ImportError:
            pass
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        if img.dtype != np.uint8:
            lo, hi = float(img.min()), float(img.max())
            if hi > lo:
                img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        return img.astype(np.uint8)

    @staticmethod
    def _scale_consistency_mask(
        scales_src: np.ndarray,
        scales_ref: np.ndarray,
        gsd_ratio: float,
        max_log_dev: float = 0.3,
    ) -> np.ndarray:
        """
        Scale-consistency filter (ARCHITECTURE.md §4, FEATURES.md F10).

        Reject match i if:
            |log(scales_src[i] / scales_ref[i]) - log(gsd_ratio)| > max_log_dev

        Returns boolean mask (True = keep).
        """
        if len(scales_src) == 0:
            return np.ones(0, dtype=bool)
        ratio = np.asarray(scales_src, dtype=np.float64) / (
            np.asarray(scales_ref, dtype=np.float64) + 1e-9
        )
        deviation = np.abs(np.log(ratio + 1e-9) - np.log(max(gsd_ratio, 1e-9)))
        return deviation <= max_log_dev

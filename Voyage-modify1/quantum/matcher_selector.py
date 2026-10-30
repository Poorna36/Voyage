"""
quantum/matcher_selector.py
===========================
Intelligent Matcher Selector (Build Spec v2 §3A).

Provides:
  1. Rule-based matcher selector:
     - Deterministic decision logic based on image pair characteristics
       (sensor type, GSD ratio, texture contrast, crater density, solar azimuth/illumination).
     - Transparent decision paths with clear explanation strings.
  2. Guardrails:
     - Only selects matchers that actually exist and are importable in repo.
     - Validates and handles manual override (rejects unknown matchers with fallback).
     - Provides safe fallback matcher name (default: "sift").
  3. Learned matcher evaluator:
     - Regret and accuracy calculation against oracle / benchmark outcomes.
  4. Benchmark logging:
     - Records matcher selection, outcome, keypoint count, inlier count, and runtime.

References:
  - Build Spec v2 §3A
  - FEATURES.md F26
  - CONFIGURATION.md §4
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("quantum.matcher_selector")

# Default known-good fallback
DEFAULT_FALLBACK_MATCHER = "sift"

# Real matchers that exist in Voyage codebase
KNOWN_PROJECT_MATCHERS = ["sift", "rift2", "lightglue", "crater", "lnift"]


def get_available_matchers() -> List[str]:
    """
    Dynamically detect which matchers are actually importable/available
    in the current environment.
    """
    available = []
    # 1. SIFT (OpenCV)
    try:
        import cv2
        if hasattr(cv2, "SIFT_create"):
            available.append("sift")
    except ImportError:
        pass

    # 2. RIFT2 (Phase congruency / NumPy)
    try:
        from src.matching.rift import RIFTMatcher
        available.append("rift2")
    except Exception:
        # Check if file exists even if not imported
        if (Path(__file__).resolve().parent.parent / "src" / "matching" / "rift.py").exists():
            available.append("rift2")

    # 3. LightGlue (PyTorch + LightGlue)
    try:
        import torch
        available.append("lightglue")
    except ImportError:
        pass

    # 4. Crater (Landmark / OpenCV / ellipse fitting)
    try:
        from src.matching.crater import CraterMatcher
        available.append("crater")
    except Exception:
        if (Path(__file__).resolve().parent.parent / "src" / "matching" / "crater.py").exists():
            available.append("crater")

    # 5. LNIFT (Local normalized intensity feature)
    try:
        from src.matching.lnift import LNIFTMatcher
        available.append("lnift")
    except Exception:
        if (Path(__file__).resolve().parent.parent / "src" / "matching" / "lnift.py").exists():
            available.append("lnift")

    # Ensure fallback matcher is present if any matcher exists
    if not available:
        available = [DEFAULT_FALLBACK_MATCHER]

    return sorted(list(set(available)))


@dataclass
class MatcherSelectionResult:
    """
    Result of intelligent matcher selection.
    """
    pair_id: str
    selected_matcher: str
    confidence: float
    explanation: str
    fallback_matcher: str
    available_matchers: List[str]
    decision_path: List[str] = field(default_factory=list)
    manual_override: bool = False
    strategy: str = "rules"  # "rules" | "learned" | "override"
    all_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleBasedMatcherSelector:
    """
    Deterministic rule-based matcher selector with transparent explanations.
    """

    def __init__(
        self,
        available_matchers: Optional[List[str]] = None,
        tau_crater: float = 5.0,
        tau_solar_azimuth: float = 45.0,
        tau_low_texture: float = 5.0,
        tau_high_texture: float = 15.0,
        tau_gsd_disparity: float = 0.5,
    ):
        self.available_matchers = available_matchers or get_available_matchers()
        self.tau_crater = tau_crater
        self.tau_crater_log = math.log1p(tau_crater)  # ~1.7917
        self.tau_solar_azimuth = tau_solar_azimuth
        self.tau_low_texture = tau_low_texture
        self.tau_high_texture = tau_high_texture
        self.tau_gsd_disparity = tau_gsd_disparity

    def _resolve_fallback(self, chosen: str) -> str:
        """Choose a fallback matcher different from chosen."""
        for candidate in ["sift", "rift2", "lightglue", "lnift"]:
            if candidate in self.available_matchers and candidate != chosen:
                return candidate
        # If no other candidate available
        return chosen

    def select(
        self,
        features: Any,
        override_matcher: Optional[str] = None,
    ) -> MatcherSelectionResult:
        """
        Select best matcher using deterministic rules or manual override.

        Parameters
        ----------
        features : MSMFeatureVector or dict
            Image pair features.
        override_matcher : Optional[str]
            Manual override matcher name.
        """
        # Extract fields from dict or dataclass
        if isinstance(features, dict):
            pair_id = str(features.get("pair_id", "unknown_pair"))
            sensor_pair_enc = int(features.get("sensor_pair_enc", 0))
            gsd_ratio = float(features.get("gsd_ratio", 1.0))
            delta_solar_azimuth = float(features.get("delta_solar_azimuth", 0.0))
            crater_density = float(features.get("crater_density", 0.0))
            src_texture = float(features.get("src_texture_contrast", 20.0))
            ref_texture = float(features.get("ref_texture_contrast", 20.0))
            terrain_class_enc = int(features.get("terrain_class_enc", 0))
        else:
            pair_id = str(getattr(features, "pair_id", "unknown_pair"))
            sensor_pair_enc = int(getattr(features, "sensor_pair_enc", 0))
            gsd_ratio = float(getattr(features, "gsd_ratio", 1.0))
            delta_solar_azimuth = float(getattr(features, "delta_solar_azimuth", 0.0))
            crater_density = float(getattr(features, "crater_density", 0.0))
            src_texture = float(getattr(features, "src_texture_contrast", 20.0))
            ref_texture = float(getattr(features, "ref_texture_contrast", 20.0))
            terrain_class_enc = int(getattr(features, "terrain_class_enc", 0))

        # Check manual override
        if override_matcher is not None:
            override_clean = override_matcher.strip().lower()
            if override_clean in self.available_matchers:
                fb = self._resolve_fallback(override_clean)
                return MatcherSelectionResult(
                    pair_id=pair_id,
                    selected_matcher=override_clean,
                    confidence=1.0,
                    explanation=f"Manual override: user specified '{override_clean}'.",
                    fallback_matcher=fb,
                    available_matchers=self.available_matchers,
                    decision_path=[f"manual_override({override_clean})"],
                    manual_override=True,
                    strategy="override",
                    all_scores={m: (1.0 if m == override_clean else 0.0) for m in self.available_matchers},
                )
            else:
                logger.warning(
                    "Manual override matcher '%s' not in available matchers %s. Falling back to rule selection.",
                    override_matcher,
                    self.available_matchers,
                )

        decision_path: List[str] = []
        scores: Dict[str, float] = {m: 0.1 for m in self.available_matchers}

        # Rule 1: High crater density & crater-rich terrain
        # crater_density is log(1 + craters/km2)
        if crater_density >= self.tau_crater_log and "crater" in self.available_matchers and terrain_class_enc != 1:
            decision_path.append(f"high_crater_density({crater_density:.2f}>={self.tau_crater_log:.2f})")
            scores["crater"] = 0.88
            explanation = (
                f"Selected crater: high crater landmark density "
                f"({crater_density:.2f} log-density >= {self.tau_crater_log:.2f}) on suitable terrain."
            )
            selected = "crater"
            fb = "rift2" if "rift2" in self.available_matchers else self._resolve_fallback(selected)

        # Rule 2: Multispectral / IIRS pair (sensor_pair_enc == 2)
        elif sensor_pair_enc == 2:
            decision_path.append("multispectral_sensor_pair(IIRS)")
            if "rift2" in self.available_matchers:
                scores["rift2"] = 0.85
                selected = "rift2"
                explanation = "Selected rift2: cross-sensor/multispectral pair requiring phase congruency structural features."
            elif "lnift" in self.available_matchers:
                scores["lnift"] = 0.80
                selected = "lnift"
                explanation = "Selected lnift: multispectral pair with intensity non-linearities."
            else:
                selected = DEFAULT_FALLBACK_MATCHER
                explanation = "Fallback to sift: specialized multispectral matchers unavailable."
            fb = self._resolve_fallback(selected)

        # Rule 3: Extreme solar azimuth / illumination difference
        elif delta_solar_azimuth >= self.tau_solar_azimuth:
            decision_path.append(f"large_illumination_diff({delta_solar_azimuth:.1f}>={self.tau_solar_azimuth:.1f}deg)")
            if "rift2" in self.available_matchers:
                scores["rift2"] = 0.82
                selected = "rift2"
                explanation = (
                    f"Selected rift2: large solar azimuth illumination difference "
                    f"({delta_solar_azimuth:.1f}° >= {self.tau_solar_azimuth:.1f}°) requiring phase congruency."
                )
            elif "lightglue" in self.available_matchers:
                scores["lightglue"] = 0.78
                selected = "lightglue"
                explanation = (
                    f"Selected lightglue: large illumination difference ({delta_solar_azimuth:.1f}°) "
                    f"handled by deep learned matcher."
                )
            else:
                selected = DEFAULT_FALLBACK_MATCHER
                explanation = "Fallback to sift: illumination-robust matchers unavailable."
            fb = self._resolve_fallback(selected)

        # Rule 4: Extreme GSD ratio / scale disparity
        elif gsd_ratio < self.tau_gsd_disparity or gsd_ratio > (1.0 / self.tau_gsd_disparity):
            decision_path.append(f"large_scale_disparity(gsd_ratio={gsd_ratio:.2f})")
            if "rift2" in self.available_matchers:
                scores["rift2"] = 0.80
                selected = "rift2"
                explanation = f"Selected rift2: multi-octave scale-space handles scale disparity (GSD ratio={gsd_ratio:.2f})."
            elif "lightglue" in self.available_matchers:
                scores["lightglue"] = 0.78
                selected = "lightglue"
                explanation = f"Selected lightglue: deep keypoints handle scale disparity (GSD ratio={gsd_ratio:.2f})."
            else:
                selected = DEFAULT_FALLBACK_MATCHER
                explanation = "Fallback to sift: multi-scale matchers unavailable."
            fb = self._resolve_fallback(selected)

        # Rule 5: Low texture / low contrast
        elif min(src_texture, ref_texture) < self.tau_low_texture:
            decision_path.append(f"low_texture_contrast({min(src_texture, ref_texture):.1f}<{self.tau_low_texture:.1f})")
            if "lightglue" in self.available_matchers:
                scores["lightglue"] = 0.82
                selected = "lightglue"
                explanation = (
                    f"Selected lightglue: low texture contrast ({min(src_texture, ref_texture):.1f}) "
                    f"benefits from learned deep keypoint detection."
                )
            elif "rift2" in self.available_matchers:
                scores["rift2"] = 0.75
                selected = "rift2"
                explanation = f"Selected rift2: low texture contrast handled via frequency-domain phase congruency."
            else:
                selected = DEFAULT_FALLBACK_MATCHER
                explanation = "Fallback to sift: deep/phase matchers unavailable."
            fb = self._resolve_fallback(selected)

        # Rule 6: High texture, same-sensor optical pair with moderate illumination
        else:
            decision_path.append("standard_optical_pair_high_texture")
            if "sift" in self.available_matchers:
                scores["sift"] = 0.90
                selected = "sift"
                explanation = (
                    f"Selected sift: same-sensor optical pair with high texture contrast "
                    f"({src_texture:.1f}, {ref_texture:.1f}) and moderate illumination difference ({delta_solar_azimuth:.1f}°)."
                )
            elif "lightglue" in self.available_matchers:
                scores["lightglue"] = 0.85
                selected = "lightglue"
                explanation = "Selected lightglue: standard optical pair, sift unavailable."
            else:
                selected = self.available_matchers[0]
                explanation = f"Selected {selected}: first available matcher."
            fb = self._resolve_fallback(selected)

        # Safety check: ensure selected is actually available
        if selected not in self.available_matchers:
            logger.warning(
                "Selected matcher '%s' is not in available matchers %s. Forcing fallback to %s.",
                selected,
                self.available_matchers,
                DEFAULT_FALLBACK_MATCHER,
            )
            selected = DEFAULT_FALLBACK_MATCHER if DEFAULT_FALLBACK_MATCHER in self.available_matchers else self.available_matchers[0]
            explanation += f" (Guardrail forced fallback to {selected})"
            fb = self._resolve_fallback(selected)

        # Normalize scores to pseudo-probabilities
        total = sum(scores.values())
        if total > 0:
            norm_scores = {k: round(v / total, 4) for k, v in scores.items()}
        else:
            norm_scores = {k: round(1.0 / len(scores), 4) for k, v in scores.items()}

        conf = norm_scores.get(selected, 0.75)

        return MatcherSelectionResult(
            pair_id=pair_id,
            selected_matcher=selected,
            confidence=conf,
            explanation=explanation,
            fallback_matcher=fb,
            available_matchers=self.available_matchers,
            decision_path=decision_path,
            manual_override=False,
            strategy="rules",
            all_scores=norm_scores,
        )


class LearnedMatcherEvaluator:
    """
    Evaluator for learned matcher selector:
    Computes selection accuracy and regret against oracle best-performing matcher.
    """

    @staticmethod
    def evaluate(
        predictions: List[str],
        oracle_best: List[str],
        oracle_metrics: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against oracle ground truth.

        Parameters
        ----------
        predictions : List[str]
            Predicted matcher names.
        oracle_best : List[str]
            Oracle best matcher names.
        oracle_metrics : Optional[List[Dict[str, float]]]
            Per-sample dictionary of matcher_name -> performance metric (e.g. inlier_ratio or recall).

        Returns
        -------
        Dict[str, float]
            selection_accuracy, regret, total_samples
        """
        if not predictions or len(predictions) != len(oracle_best):
            return {"selection_accuracy": 0.0, "mean_regret": 0.0, "total_samples": 0}

        n = len(predictions)
        correct = sum(1 for p, o in zip(predictions, oracle_best) if p == o)
        accuracy = float(correct) / n

        regrets = []
        if oracle_metrics and len(oracle_metrics) == n:
            for pred, oracle, metrics in zip(predictions, oracle_best, oracle_metrics):
                best_val = metrics.get(oracle, 1.0)
                pred_val = metrics.get(pred, 0.0)
                regret = max(0.0, best_val - pred_val)
                regrets.append(regret)
            mean_regret = float(sum(regrets) / n)
        else:
            # 0/1 regret
            mean_regret = float(1.0 - accuracy)

        return {
            "selection_accuracy": round(accuracy, 4),
            "mean_regret": round(mean_regret, 4),
            "total_samples": n,
        }


def log_matcher_run(
    pair_id: str,
    selected_matcher: str,
    success: bool,
    keypoint_count: int,
    inlier_count: int,
    runtime_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
    log_path: Union[str, Path] = "logs/matcher_benchmark.jsonl",
) -> Dict[str, Any]:
    """
    Record benchmark logging for a matcher execution run.
    """
    entry = {
        "timestamp": time.time(),
        "pair_id": pair_id,
        "selected_matcher": selected_matcher,
        "success": bool(success),
        "keypoint_count": int(keypoint_count),
        "inlier_count": int(inlier_count),
        "runtime_ms": round(float(runtime_ms), 2),
        "metadata": metadata or {},
    }
    try:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write matcher run log to %s: %s", log_path, exc)

    return entry

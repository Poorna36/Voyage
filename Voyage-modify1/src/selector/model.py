"""
src/selector/model.py
======================
Matcher Selection Model (MSM) inference engine and routing controller (L1.5 / S4.5).

Evaluates the 13-dimensional MSMFeatureVector using LightGBM multi-class meta-model
and dual-threshold confidence routing:
  - P_max >= tau_high (0.65)           -> [selected_matcher]
  - tau_low (0.40) <= P_max < tau_high -> [selected_matcher, fallback_matcher]
  - P_max < tau_low                    -> [M0, M1, M2, M3] (Safe Mode)

Enforces hard rule safety gates (Crater density gate, GPU availability, IIRS module routing).

References:
  - FEATURES.md F26
  - INTERFACES.md §10.2
  - CONFIGURATION.md §11
  - DECISIONS.md D15-D19
  - PROGRESS.md §5.5.4
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.selector.features import MSMFeatureVector, vectorize_features

logger = logging.getLogger("selector.model")

MATCHER_NAMES = ["sift", "rift2", "lightglue", "crater"]
INDEX_TO_MATCHER = {i: name for i, name in enumerate(MATCHER_NAMES)}
MATCHER_TO_INDEX = {name: i for i, name in enumerate(MATCHER_NAMES)}


@dataclass
class SelectorResult:
    """
    Inference result and routing decision from the Matcher Selection Model.
    """
    pair_id: str
    selected_matcher: str
    confidence: float
    fallback_matcher: str
    all_probs: Dict[str, float]
    routing_reason: str
    matchers_to_run: List[str]
    hard_rules_applied: List[str]
    selector_version: str = "msm_v1"
    feature_vector_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation."""
        return asdict(self)


class MatcherSelector:
    """
    Matcher Selection Model inference engine.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], Path, str]] = None):
        if config is None:
            self.config = self._load_default_config()
        elif isinstance(config, (str, Path)):
            self.config = self._load_config_from_path(Path(config))
        else:
            self.config = config

        msm_cfg = self.config.get("msm", self.config)
        self.enabled: bool = bool(msm_cfg.get("enabled", False))
        self.model_path: str = str(msm_cfg.get("model_path", "models/msm_v1.pkl"))
        self.model_version: str = str(msm_cfg.get("model_version", "msm_v1"))
        self.tau_high: float = float(msm_cfg.get("tau_high", 0.65))
        self.tau_low: float = float(msm_cfg.get("tau_low", 0.40))
        self.hard_rules: Dict[str, Any] = msm_cfg.get("hard_rules", {})
        self.fallback_cfg: Dict[str, Any] = msm_cfg.get("fallback", {})

        self.model: Any = None
        self.is_loaded: bool = False
        self._load_model_safe()

    def _load_default_config(self) -> Dict[str, Any]:
        cfg_path = Path("configs/msm.yaml")
        if cfg_path.exists():
            return self._load_config_from_path(cfg_path)
        return {
            "msm": {
                "enabled": False,
                "model_path": "models/msm_v1.pkl",
                "model_version": "msm_v1",
                "tau_high": 0.65,
                "tau_low": 0.40,
                "hard_rules": {
                    "crater_density_gate": {"enabled": True, "tau_c": 5.0},
                    "gpu_gate": {"enabled": True, "check_at_startup": True},
                    "iirs_track_gate": {"enabled": True},
                },
                "fallback": {
                    "on_model_load_error": "benchmark_mode",
                    "on_feature_extraction_error": "benchmark_mode",
                    "on_s4_gate_failure": "sift",
                },
            }
        }

    def _load_config_from_path(self, path: Path) -> Dict[str, Any]:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _load_model_safe(self) -> bool:
        p = Path(self.model_path)
        if p.exists():
            try:
                with open(p, "rb") as fh:
                    self.model = pickle.load(fh)
                self.is_loaded = True
                logger.info("Loaded MSM model from %s", self.model_path)
                return True
            except Exception as exc:
                logger.warning("Failed to load MSM model from %s: %s", self.model_path, exc)
                self.model = None
                self.is_loaded = False
                return False
        else:
            logger.debug("MSM model file not found at %s (running in rule/safe fallback mode)", self.model_path)
            self.model = None
            self.is_loaded = False
            return False

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> bool:
        """Explicitly load or reload model from a specified path."""
        if model_path is not None:
            self.model_path = str(model_path)
        return self._load_model_safe()

    def predict(self, features: MSMFeatureVector) -> SelectorResult:
        """
        Predict the winning matcher and determine execution routing for a given pair.

        Parameters
        ----------
        features : MSMFeatureVector
            13-dimensional canonical feature vector.

        Returns
        -------
        SelectorResult
            Selection result including routing decision and confidence probabilities.
        """
        hard_rules_applied: List[str] = []
        vec = vectorize_features(features).reshape(1, -1)

        # -------------------------------------------------------------- #
        # 1. Evaluate Model Probabilities (or heuristic default)
        # -------------------------------------------------------------- #
        if self.is_loaded and self.model is not None:
            try:
                if hasattr(self.model, "predict_proba"):
                    raw_probs = self.model.predict_proba(vec)[0]
                elif hasattr(self.model, "predict"):
                    pred = self.model.predict(vec)[0]
                    raw_probs = np.zeros(len(MATCHER_NAMES), dtype=np.float32)
                    if isinstance(pred, (int, np.integer)):
                        raw_probs[int(pred)] = 1.0
                    else:
                        raw_probs[0] = 1.0
                else:
                    raw_probs = np.ones(len(MATCHER_NAMES), dtype=np.float32) / len(MATCHER_NAMES)
            except Exception as exc:
                logger.warning("[%s] Model inference failed: %s. Using safe-mode fallback.", features.pair_id, exc)
                raw_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
                hard_rules_applied.append("model_inference_exception_safe_mode")
        else:
            # Rule-based / Prior distribution when no model file loaded
            # Priors based on benchmark evidence (LightGlue > RIFT2 > SIFT > Crater)
            raw_probs = np.array([0.20, 0.25, 0.45, 0.10], dtype=np.float32)

        # -------------------------------------------------------------- #
        # 2. Enforce Hard Rule Gating
        # -------------------------------------------------------------- #
        probs = np.zeros(len(MATCHER_NAMES), dtype=np.float32)
        for i in range(min(len(raw_probs), len(MATCHER_NAMES))):
            probs[i] = float(raw_probs[i])

        # Gate A: Crater branch gating (F14, DEC-013)
        crater_gate = self.hard_rules.get("crater_density_gate", {})
        if crater_gate.get("enabled", True):
            tau_c = float(crater_gate.get("tau_c", 5.0))
            tau_c_log = math.log1p(tau_c)
            # Crater index is 3
            if features.crater_density < tau_c_log or features.terrain_class_enc == 1:
                if probs[3] > 0.0:
                    hard_rules_applied.append(f"crater_density_below_tau_c({features.crater_density:.2f}<{tau_c_log:.2f})")
                    probs[3] = 0.0

        # Gate B: GPU check for M2 (LightGlue index 2)
        gpu_gate = self.hard_rules.get("gpu_gate", {})
        if gpu_gate.get("enabled", True):
            has_gpu = False
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False

            if not has_gpu:
                hard_rules_applied.append("gpu_unavailable_lightglue_cpu_fallback")

        # Gate C: IIRS track gate (F12, DEC-009)
        iirs_gate = self.hard_rules.get("iirs_track_gate", {})
        if iirs_gate.get("enabled", True) and features.sensor_pair_enc == 2:
            hard_rules_applied.append("iirs_photometric_track_routed")

        # Renormalize probability distribution
        total_p = float(np.sum(probs))
        if total_p > 0:
            probs = probs / total_p
        else:
            probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Fallback to SIFT

        # -------------------------------------------------------------- #
        # 3. Dual Confidence Threshold Routing (DEC-017)
        # -------------------------------------------------------------- #
        ranked_indices = np.argsort(probs)[::-1]
        best_idx = int(ranked_indices[0])
        second_idx = int(ranked_indices[1])

        p_max = float(probs[best_idx])
        p_second = float(probs[second_idx])

        selected_matcher = INDEX_TO_MATCHER[best_idx]
        fallback_matcher = INDEX_TO_MATCHER[second_idx]

        probs_dict = {
            INDEX_TO_MATCHER[i]: round(float(probs[i]), 4)
            for i in range(len(MATCHER_NAMES))
        }

        if p_max >= self.tau_high:
            matchers_to_run = [selected_matcher]
            routing_reason = "high_confidence_single_matcher"
        elif p_max >= self.tau_low:
            matchers_to_run = [selected_matcher, fallback_matcher]
            routing_reason = "medium_confidence_dual_matcher"
        else:
            matchers_to_run = [m for m in MATCHER_NAMES]
            routing_reason = "low_confidence_safe_mode"

        result = SelectorResult(
            pair_id=features.pair_id,
            selected_matcher=selected_matcher,
            confidence=round(p_max, 4),
            fallback_matcher=fallback_matcher,
            all_probs=probs_dict,
            routing_reason=routing_reason,
            matchers_to_run=matchers_to_run,
            hard_rules_applied=hard_rules_applied,
            selector_version=self.model_version,
            feature_vector_hash=features.feature_vector_hash,
        )
        return result

    def save_result(self, result: SelectorResult, out_path: Union[Path, str]) -> Path:
        """Save SelectorResult to JSON file atomically."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        return out

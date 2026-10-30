"""
quantum/config_validator.py
===========================
Validation and Schema Enforcement for Build Spec v2 §5 Unified Configuration.

Ensures:
  1. Valid top-level execution mode in {'classical', 'hybrid', 'experimental'}.
  2. Master switch honor (when enabled=false, pure classical execution).
  3. Actionable fast-fail messages when a component is enabled but its dependencies are missing.
  4. Validation of matcher selector strategies and simulator noise parameters.
  5. Rejection of unexpected/unknown configuration keys.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

log = logging.getLogger("quantum.config_validator")

ALLOWED_MODES: Set[str] = {"classical", "hybrid", "experimental"}
ALLOWED_STRATEGIES: Set[str] = {"rules", "learned"}
ALLOWED_SIM_MODES: Set[str] = {"ideal_simulator", "noisy_simulator", "fake_ibm_backend"}

KNOWN_QUANTUM_KEYS: Set[str] = {
    "enabled",
    "mode",
    "matcher_selector",
    "qnn_reliability",
    "qcnn_features",
    "simulator",
    "qaoa_correspondence",
}


def validate_quantum_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quantum configuration dictionary against Spec v2 §5 schema.

    Raises
    ------
    ValueError
        If any configuration value is invalid or an enabled component lacks dependencies.
    """
    q_cfg = config.get("quantum", config)

    # 1. Unknown keys check
    unknown_keys = set(q_cfg.keys()) - KNOWN_QUANTUM_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown configuration key(s) in quantum section: {sorted(list(unknown_keys))}")

    # 2. Master switch & mode
    enabled = bool(q_cfg.get("enabled", False))
    mode = str(q_cfg.get("mode", "classical")).lower().strip()
    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"Invalid quantum.mode '{mode}'. Allowed modes are: {sorted(list(ALLOWED_MODES))}"
        )

    # If master switch is false, entire pipeline is pure classical - no quantum checks required
    if not enabled or mode == "classical":
        return q_cfg

    # 3. Matcher Selector check
    ms_cfg = q_cfg.get("matcher_selector", {})
    if ms_cfg.get("enabled", False):
        strat = str(ms_cfg.get("strategy", "rules")).lower().strip()
        if strat not in ALLOWED_STRATEGIES:
            raise ValueError(
                f"Invalid matcher_selector.strategy '{strat}'. Allowed strategies: {sorted(list(ALLOWED_STRATEGIES))}"
            )

    # 4. QNN Reliability check
    qnn_cfg = q_cfg.get("qnn_reliability", {})
    if qnn_cfg.get("enabled", False):
        try:
            import qiskit_machine_learning
        except ImportError:
            raise ValueError(
                "QNN enabled but qiskit-machine-learning not installed. "
                "Install via pip install qiskit-machine-learning or set quantum.qnn_reliability.enabled: false"
            )

    # 5. QCNN Features check
    qcnn_cfg = q_cfg.get("qcnn_features", {})
    if qcnn_cfg.get("enabled", False):
        try:
            import qiskit
        except ImportError:
            raise ValueError(
                "QCNN enabled but qiskit is not installed. "
                "Install via pip install qiskit or set quantum.qcnn_features.enabled: false"
            )

    # 6. Simulator check
    sim_cfg = q_cfg.get("simulator", {})
    sim_mode = str(sim_cfg.get("mode", "ideal_simulator")).lower().strip()
    if sim_mode not in ALLOWED_SIM_MODES:
        raise ValueError(
            f"Invalid simulator.mode '{sim_mode}'. Allowed modes: {sorted(list(ALLOWED_SIM_MODES))}"
        )

    # 7. QAOA Correspondence check
    qaoa_cfg = q_cfg.get("qaoa_correspondence", {})
    if qaoa_cfg.get("enabled", False):
        depth = int(qaoa_cfg.get("depth_p", 1))
        if depth < 1 or depth > 5:
            raise ValueError(f"QAOA depth_p must be between 1 and 5; got {depth}")

    return q_cfg


def load_and_validate_quantum_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configs/quantum.yaml and validate schema."""
    p = Path(config_path) if config_path else (Path(__file__).resolve().parent.parent / "configs" / "quantum.yaml")
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}

    return validate_quantum_config(raw_cfg)

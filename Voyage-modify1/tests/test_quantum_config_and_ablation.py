"""
tests/test_quantum_config_and_ablation.py
=========================================
Tests for Build Spec v2 §5 (Unified Config), §6 (Ablation Protocol), and §7 (Run Records).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.config_validator import validate_quantum_config, load_and_validate_quantum_config
from quantum.records import QuantumRunRecord, get_git_commit_hash, generate_run_id
from quantum.ablation_runner import AblationLadderRunner, AblationConfigResult


def test_config_validator_valid():
    cfg = {
        "quantum": {
            "enabled": True,
            "mode": "hybrid",
            "matcher_selector": {"enabled": True, "strategy": "rules"},
            "simulator": {"mode": "ideal_simulator"},
            "qaoa_correspondence": {"enabled": True, "depth_p": 2},
        }
    }
    validated = validate_quantum_config(cfg)
    assert validated["enabled"] is True
    assert validated["mode"] == "hybrid"


def test_config_validator_rejects_unknown_keys():
    cfg = {
        "quantum": {
            "enabled": False,
            "mode": "classical",
            "unknown_fake_option": 123,
        }
    }
    with pytest.raises(ValueError, match="Unknown configuration key"):
        validate_quantum_config(cfg)


def test_config_validator_rejects_invalid_mode():
    cfg = {
        "quantum": {
            "enabled": True,
            "mode": "unsupported_superconducting_cloud_mode",
        }
    }
    with pytest.raises(ValueError, match="Invalid quantum.mode"):
        validate_quantum_config(cfg)


def test_load_default_quantum_config():
    cfg = load_and_validate_quantum_config()
    assert "enabled" in cfg
    assert "mode" in cfg


def test_extended_quantum_run_record(tmp_path):
    run_id = generate_run_id("test")
    record = QuantumRunRecord(
        run_id=run_id,
        pair_id="lunar_001",
        module="qaoa_correspondence",
        pipeline_mode="hybrid",
        active_components=["qaoa_correspondence"],
        backend_type="simulator",
        backend_mode="ideal_simulator",
        repro_command="python -m pytest",
        decision_logs={"gap_to_exact": 0.05},
    )

    assert len(record.git_commit) > 0
    assert record.backend_type == "simulator"
    assert record.decision_logs["gap_to_exact"] == 0.05

    saved_file = record.save(results_dir=tmp_path)
    assert saved_file.exists()

    with open(saved_file, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["git_commit"] == record.git_commit
    assert loaded["pipeline_mode"] == "hybrid"


def test_ablation_ladder_runner(tmp_path):
    runner = AblationLadderRunner(output_dir=tmp_path, seed=42)
    sample_pairs = runner._generate_synthetic_test_pairs()[:2]
    results = runner.run_ladder(sample_pairs=sample_pairs)

    assert len(results) == 6
    names = [r.config_name for r in results]
    assert "1. pure_classical" in names[0]
    assert "6. combined_hybrid" in names[5]

    for r in results:
        assert isinstance(r, AblationConfigResult)
        assert r.n_pairs > 0
        assert r.success_rate >= 0.0

    # Ensure summary files were generated
    csv_file = tmp_path / "ablation_summary.csv"
    md_file = tmp_path / "ablation_summary.md"
    assert csv_file.exists()
    assert md_file.exists()

    with open(md_file, "r", encoding="utf-8") as fh:
        md_text = fh.read()
    assert "pure_classical" in md_text
    assert "combined_hybrid" in md_text

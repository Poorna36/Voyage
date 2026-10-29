"""
tests/verify_phase8.py
======================
Phase 8 Verification Suite — Leaderboard & System Validation.

Tests all components of Phase 8:
  - T8-01: evaluate_pairs produces valid EvaluationRecord JSON (INTERFACES.md §4)
  - T8-02: Partition isolation — only "eval" checkpoints affect reported RMSE
  - T8-03: gt_interannotator_rmse_px computed from "qc" checkpoints
  - T8-04: aggregate.py reads pair_results and outputs valid leaderboard.csv
  - T8-05: leakage_audit catches geo_cell / pair overlap between train & test
  - T8-06: leakage_audit passes on clean manifest with zero leakage
  - T8-07: system_validation passes all 11 criteria on passing benchmark data
  - T8-08: system_validation correctly fails on poor accuracy (RMSE >= 1.0 px)
  - T8-09: Polar stratum inclusion is verified and required
  - T8-10: TMC-2-WAC sensor pair is reported separately and non-gating
  - T8-11: IIRS-WAC evaluated against sub-80m absolute RMSE target
  - T8-12: Synthetic GT generator builds complete valid test benchmark

Usage:
  pytest tests/verify_phase8.py -v
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from scripts.evaluate_pairs import evaluate_pair, project_coordinates
from scripts.generate_synthetic_gt import generate_benchmark_dataset, generate_gt_file
from src.evaluation.aggregate import aggregate, run_aggregation, write_leaderboard_csv
from src.evaluation.leakage_audit import audit_manifest, run_audit
from src.evaluation.metrics import compute_all_metrics, gt_interannotator_rmse, rmse
from src.evaluation.system_validation import evaluate_system


def test_t8_01_evaluate_pairs_schema():
    """T8-01: evaluate_pair produces valid EvaluationRecord JSON matching INTERFACES.md §4."""
    pair_record = {
        "pair_id": "ohr_test_001__nac_polar",
        "split": "test",
        "terrain_class": "polar_highland",
        "latitude_center_deg": -86.2,
        "delta_azimuth_deg": 15.0,
        "crater_density_per_km2": 4.5,
        "src": {"sensor": "OHRC"},
        "ref": {"type": "NAC"},
    }

    gt_data = {
        "pair_id": "ohr_test_001__nac_polar",
        "annotator": "manual_grid_6x6",
        "n_checkpoints": 4,
        "checkpoints": [
            {"id": 0, "src_xy": [100.0, 100.0], "ref_xy": [150.0, 150.0], "partition": "eval"},
            {"id": 1, "src_xy": [200.0, 100.0], "ref_xy": [250.0, 150.0], "partition": "eval"},
            {"id": 2, "src_xy": [100.0, 200.0], "ref_xy": [150.0, 250.0], "partition": "eval"},
            {"id": 3, "src_xy": [200.0, 200.0], "ref_xy": [250.0, 250.0], "partition": "eval"},
        ],
    }

    # True shift: +50 in both x and y
    geo_model = {
        "model_type": "affine",
        "model_matrix": [
            [1.0, 0.0, 50.2],
            [0.0, 1.0, 49.8],
        ],
        "inlier_count": 85,
        "inlier_ratio": 0.75,
        "runtime_s": 2.1,
    }

    record = evaluate_pair(
        pair_record=pair_record,
        gt_data=gt_data,
        geometry_or_model=geo_model,
        matcher="lightglue",
        is_arbitration_winner=True,
    )

    assert record["pair_id"] == "ohr_test_001__nac_polar"
    assert record["matcher"] == "lightglue"
    assert record["split"] == "test"
    assert "stratum" in record
    assert record["stratum"]["sensor_pair"] == "OHRC-NAC"
    assert record["stratum"]["latitude_bin"] == "polar"
    assert "metrics" in record
    assert record["metrics"]["rmse_px"] < 0.50
    assert record["metrics"]["pct_lt_1px"] == 1.0
    assert record["gt_checkpoint_count"] == 4
    assert record["arbitration_winner"] is True


def test_t8_02_eval_partition_isolation():
    """T8-02: Only 'eval' partition points affect RMSE. Inserting 'fit' or 'qc' must not change RMSE."""
    base_eval_pts = [
        {"id": 0, "src_xy": [10.0, 10.0], "ref_xy": [20.0, 20.0], "partition": "eval"},
        {"id": 1, "src_xy": [50.0, 50.0], "ref_xy": [60.0, 60.0], "partition": "eval"},
        {"id": 2, "src_xy": [90.0, 90.0], "ref_xy": [100.0, 100.0], "partition": "eval"},
    ]

    pred_ref_xy = np.array([
        [20.2, 19.9],
        [60.1, 60.3],
        [99.8, 100.1],
    ], dtype=np.float64)

    m1 = compute_all_metrics(pred_ref_xy, base_eval_pts)
    rmse_base = m1["rmse_px"]

    # Insert a "fit" point with large error
    augmented_pts = list(base_eval_pts) + [
        {"id": 99, "src_xy": [100.0, 100.0], "ref_xy": [500.0, 500.0], "partition": "fit"},
        {"id": 98, "src_xy": [10.0, 10.0], "ref_xy": [20.5, 20.5], "partition": "qc"},
    ]

    m2 = compute_all_metrics(pred_ref_xy, augmented_pts)
    rmse_augmented = m2["rmse_px"]

    assert abs(rmse_base - rmse_augmented) < 1e-9, (
        f"Partition violation: inserting fit/qc points changed RMSE ({rmse_base} vs {rmse_augmented})"
    )


def test_t8_03_interannotator_rmse():
    """T8-03: gt_interannotator_rmse is computed between original and QC re-annotated points."""
    orig = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]], dtype=np.float64)
    qc = np.array([[100.2, 99.8], [200.1, 200.3], [299.8, 300.1]], dtype=np.float64)

    err = gt_interannotator_rmse(orig, qc)
    assert 0.10 < err < 0.40, f"Expected realistic human annotation error, got: {err}"


def test_t8_04_aggregate_leaderboard_csv():
    """T8-04: aggregate.py reads pair_results and outputs valid leaderboard.csv."""
    records = [
        {
            "pair_id": "p1",
            "matcher": "lightglue",
            "split": "test",
            "stratum": {
                "sensor_pair": "OHRC-NAC",
                "terrain_class": "polar_highland",
                "latitude_bin": "polar",
                "delta_az_bin": "lt30",
                "crater_density_bin": "high",
                "ref_type": "NAC",
            },
            "metrics": {
                "rmse_px": 0.45,
                "pct_lt_1px": 0.95,
                "pct_lt_0p5px": 0.68,
                "medae_px": 0.38,
                "inlier_count": 120,
                "inlier_ratio": 0.80,
                "spatial_coverage": 0.85,
                "grid_density_std": 1.5,
                "refinement_gain_px": 0.18,
                "runtime_s": 2.5,
            },
        },
        {
            "pair_id": "p2",
            "matcher": "lightglue",
            "split": "test",
            "stratum": {
                "sensor_pair": "OHRC-NAC",
                "terrain_class": "polar_highland",
                "latitude_bin": "polar",
                "delta_az_bin": "lt30",
                "crater_density_bin": "high",
                "ref_type": "NAC",
            },
            "metrics": {
                "rmse_px": 0.55,
                "pct_lt_1px": 0.91,
                "pct_lt_0p5px": 0.60,
                "medae_px": 0.44,
                "inlier_count": 100,
                "inlier_ratio": 0.70,
                "spatial_coverage": 0.75,
                "grid_density_std": 2.0,
                "refinement_gain_px": 0.12,
                "runtime_s": 3.0,
            },
        },
    ]

    rows = aggregate(records, split_filter="test")
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["matcher"] == "lightglue"
    assert r0["n_pairs"] == 2
    assert abs(r0["rmse_px_mean"] - 0.50) < 1e-4
    assert abs(r0["spatial_coverage_mean"] - 0.80) < 1e-4


def test_t8_05_leakage_audit_catches_overlap():
    """T8-05: leakage_audit detects geo_cell or pair overlaps between train and test splits."""
    leaky_manifest = [
        {"pair_id": "pair_01", "split": "train", "geo_cell": "cell_A"},
        {"pair_id": "pair_02", "split": "test", "geo_cell": "cell_A"},  # LEAK: same cell in both splits
    ]
    passed, violations = audit_manifest(leaky_manifest)
    assert passed is False
    assert any("geo_cell" in v for v in violations)


def test_t8_06_leakage_audit_clean():
    """T8-06: leakage_audit passes cleanly on isolated train/test splits."""
    clean_manifest = [
        {"pair_id": "pair_01", "split": "train", "geo_cell": "cell_train_01"},
        {"pair_id": "pair_02", "split": "train", "geo_cell": "cell_train_02"},
        {"pair_id": "pair_03", "split": "test", "geo_cell": "cell_test_01", "gt_path": "data/gt/p3_gt.json"},
    ]
    passed, violations = audit_manifest(clean_manifest)
    assert passed is True
    assert len(violations) == 0


def test_t8_07_system_validation_pass():
    """T8-07: system_validation passes all 11 criteria on synthetic benchmark."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        paths = generate_benchmark_dataset(base)

        # Aggregate pair results to leaderboard.csv
        lb_csv = base / "results" / "leaderboard.csv"
        run_aggregation(base / "results", lb_csv, split_filter="test")

        # Run system validation
        report = evaluate_system(
            leaderboard_csv_path=lb_csv,
            manifest_path=paths["manifest_path"],
            iirs_results_dir=paths["iirs_dir"],
            pair_results_dir=paths["pair_results_dir"],
        )

        assert report.overall_passed is True, f"System validation failed: {report.summary}"
        assert report.stretch_goals_met >= 8


def test_t8_08_system_validation_fail_gating():
    """T8-08: system_validation correctly fails if best matcher RMSE exceeds 1.0 px."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "bad_leaderboard.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("matcher,sensor_pair,split,terrain_class,latitude_bin,delta_az_bin,crater_density_bin,ref_type,n_pairs,rmse_px_mean,pct_lt_1px_mean,spatial_coverage_mean,grid_density_std_mean,inlier_ratio_mean,n_failures\n")
            f.write("lightglue,OHRC-NAC,test,polar_highland,polar,lt30,high,NAC,5,1.85,0.45,0.40,5.2,0.04,2\n")

        report = evaluate_system(leaderboard_csv_path=csv_path)
        assert report.overall_passed is False
        # Criterion 1 (RMSE < 1.0) must fail
        c1 = [c for c in report.criteria if c.id == 1][0]
        assert c1.passed_required is False


def test_t8_09_polar_stratum_mandatory():
    """T8-09: Polar stratum inclusion is mandatory; missing polar stratum causes validation failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "no_polar_leaderboard.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("matcher,sensor_pair,split,terrain_class,latitude_bin,delta_az_bin,crater_density_bin,ref_type,n_pairs,rmse_px_mean,pct_lt_1px_mean,spatial_coverage_mean,grid_density_std_mean,inlier_ratio_mean,n_failures\n")
            f.write("lightglue,OHRC-NAC,test,equatorial_mare,equatorial,lt30,low,NAC,5,0.42,0.92,0.78,1.8,0.55,0\n")

        report = evaluate_system(leaderboard_csv_path=csv_path)
        c9 = [c for c in report.criteria if c.id == 9][0]
        assert c9.passed_required is False


def test_t8_10_tmc2_wac_non_gating():
    """T8-10: TMC-2–WAC shortfall is reported separately and does NOT cause overall system failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "tmc_shortfall_leaderboard.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("matcher,sensor_pair,split,terrain_class,latitude_bin,delta_az_bin,crater_density_bin,ref_type,n_pairs,rmse_px_mean,pct_lt_1px_mean,spatial_coverage_mean,grid_density_std_mean,inlier_ratio_mean,n_failures\n")
            # Primary passes
            f.write("lightglue,OHRC-NAC,test,polar_highland,polar,lt30,high,NAC,5,0.45,0.92,0.80,2.0,0.60,0\n")
            # TMC-2 has high error
            f.write("sift,TMC-2-WAC,test,equatorial_mare,equatorial,lt30,low,WAC,3,2.85,0.30,0.40,6.0,0.08,1\n")

        report = evaluate_system(leaderboard_csv_path=csv_path)
        c10 = [c for c in report.criteria if c.id == 10][0]
        assert c10.is_gating is False
        assert c10.passed_required is True  # Non-gating pass
        assert c10.passed_stretch is False  # Stretch target missed


def test_t8_11_iirs_wac_absolute_meters():
    """T8-11: IIRS-WAC target is evaluated in absolute meters (RMSE_m < 80.0 m)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        iirs_dir = Path(tmpdir) / "results" / "iirs"
        iirs_dir.mkdir(parents=True, exist_ok=True)
        lb_file = iirs_dir / "leaderboard.csv"
        with open(lb_file, "w", encoding="utf-8") as f:
            f.write("pair_id,sensor_pair,rmse_px,rmse_m,accuracy_target_m,target_met,candidate_count,selected_count,inlier_count,inlier_ratio,spatial_coverage,grid_density_std,runtime_s,created_at\n")
            f.write("iirs_test_01,IIRS-WAC,0.42,33.6,80.0,True,300,50,42,0.84,0.70,2.1,1.8,2026-08-31T00:00:00Z\n")

        csv_path = Path(tmpdir) / "optical_leaderboard.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("matcher,sensor_pair,split,terrain_class,latitude_bin,delta_az_bin,crater_density_bin,ref_type,n_pairs,rmse_px_mean,pct_lt_1px_mean,spatial_coverage_mean,grid_density_std_mean,inlier_ratio_mean,n_failures\n")
            f.write("lightglue,OHRC-NAC,test,polar_highland,polar,lt30,high,NAC,5,0.45,0.92,0.80,2.0,0.60,0\n")

        report = evaluate_system(leaderboard_csv_path=csv_path, iirs_results_dir=iirs_dir)
        c7 = [c for c in report.criteria if c.id == 7][0]
        assert c7.passed_required is True
        assert c7.passed_stretch is True  # 33.6m < 40.0m


def test_t8_12_synthetic_gt_generator():
    """T8-12: generate_synthetic_gt builds full valid manifest and 6x6 GT files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_benchmark_dataset(Path(tmpdir))
        gt_files = list(paths["gt_dir"].glob("*_gt.json"))
        assert len(gt_files) >= 4

        # Verify first GT file structure
        with open(gt_files[0], "r", encoding="utf-8") as f:
            gt_doc = json.load(f)

        assert gt_doc["annotator"] == "manual_grid_6x6"
        assert gt_doc["n_checkpoints"] >= 36
        assert gt_doc["qc_reannotated_pct"] >= 0.15

        partitions = {c["partition"] for c in gt_doc["checkpoints"]}
        assert "eval" in partitions
        assert "fit" in partitions
        assert "qc" in partitions

        # Check coordinates shape and ranges
        for c in gt_doc["checkpoints"]:
            assert len(c["src_xy"]) == 2
            assert len(c["ref_xy"]) == 2

# -*- coding: utf-8 -*-
"""
tests/test_iirs.py
==================
Phase 5 Unit and Integration Tests — IIRS Parallel Track (Feature F24).

Tests:
  T-IIRS-01: QUB reader loads 3D hyperspectral cube & parses wavelength metadata
  T-IIRS-02: Hapke photometric correction normalizes geometry & preserves positive reflectance
  T-IIRS-03: Registration band selection picks closest wavelength to WAC 643nm
  T-IIRS-04: Module isolation — ohrc_nac / tmc_wac configs NEVER invoke IIRS module
  T-IIRS-05: End-to-end IIRS registration on synthetic pair produces valid correspondences
  T-IIRS-06: Result directory separation — outputs saved to results/iirs/, not pair_results/
  T-IIRS-07: Leaderboard tagging & accuracy target tracking (RMSE_m < 80.0 m)

Run:
  python tests/test_iirs.py
  or
  pytest tests/test_iirs.py -v
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pytest

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.matching.iirs import (
    IIRSMatcher,
    IIRSMetadata,
    _hapke_bidirectional_reflectance,
    hapke_correction,
    read_qub,
    select_registration_band,
    write_synthetic_qub,
)


# ── Test Runner Utilities ────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

_results: List[Tuple[str, bool, str]] = []


def _test(name: str):
    def decorator(fn: Callable):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  {GREEN}PASS{RESET}  {name}")
        except AssertionError as exc:
            _results.append((name, False, str(exc)))
            print(f"  {RED}FAIL{RESET}  {name}  -> {exc}")
        except Exception as exc:
            _results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  {RED}FAIL{RESET}  {name}  -> {type(exc).__name__}: {exc}")
        return fn
    return decorator


# ── Tests ────────────────────────────────────────────────────────────────────

def test_iirs_01_qub_reader():
    """T-IIRS-01: QUB reader loads synthetic 3D cube & parses metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qub_path = Path(tmpdir) / "test_ch2_iir.npz"
        write_synthetic_qub(
            out_path=qub_path,
            shape=(8, 64, 64),
            seed=123,
            solar_incidence_deg=45.0,
            emission_deg=10.0,
            phase_deg=40.0,
            gsd_m=80.0,
        )

        cube, meta = read_qub(qub_path)

        assert isinstance(cube, np.ndarray), "Cube must be a numpy ndarray"
        assert cube.shape == (8, 64, 64), f"Expected shape (8, 64, 64), got {cube.shape}"
        assert cube.dtype == np.float32, f"Expected float32 dtype, got {cube.dtype}"
        assert len(meta.wavelengths_nm) == 8, f"Expected 8 wavelengths, got {len(meta.wavelengths_nm)}"
        assert meta.solar_incidence_deg == 45.0
        assert meta.emission_deg == 10.0
        assert meta.gsd_m == 80.0
        assert meta.bands == 8
        assert meta.lines == 64
        assert meta.samples == 64


def test_iirs_02_hapke_correction():
    """T-IIRS-02: Hapke photometric model produces finite, valid normalized reflectance."""
    r_30 = _hapke_bidirectional_reflectance(
        i_rad=math.radians(30.0),
        e_rad=math.radians(0.0),
        g_rad=math.radians(30.0),
        w=0.25,
    )
    assert np.isfinite(r_30) and r_30 > 0.0, f"Standard Hapke reflectance must be positive, got {r_30}"

    rng = np.random.default_rng(42)
    cube = rng.uniform(0.1, 0.6, size=(5, 32, 32)).astype(np.float32)

    corrected = hapke_correction(
        cube=cube,
        solar_incidence_deg=65.0,
        emission_deg=15.0,
        phase_deg=60.0,
    )

    assert corrected.shape == cube.shape, "Corrected cube must match input shape"
    assert np.all(np.isfinite(corrected)), "Corrected values must all be finite"
    assert np.all(corrected >= 0.0), "Photometric reflectance must be non-negative"
    assert not np.array_equal(cube, corrected), "Corrected cube must differ from raw cube"


def test_iirs_03_band_selection():
    """T-IIRS-03: Band selection picks band closest to WAC 643nm filter."""
    wavelengths = [800.0, 950.0, 1200.0, 1600.0, 2000.0, 3000.0]
    cube = np.ones((len(wavelengths), 32, 32), dtype=np.float32)
    for b in range(len(wavelengths)):
        cube[b] *= (b + 1) * 0.15

    band_img, info = select_registration_band(
        cube=cube,
        wavelengths_nm=wavelengths,
        target_wavelength_nm=643.0,
        strategy="auto",
    )

    assert band_img.shape == (32, 32), f"Expected 2D image shape (32, 32), got {band_img.shape}"
    assert info["selected_band_index"] == 0, f"Expected Band 0 (800nm) nearest to 643nm, got {info['selected_band_index']}"
    assert info["selected_wavelength_nm"] == 800.0
    assert info["target_wavelength_nm"] == 643.0
    assert band_img.min() >= 0.0 and band_img.max() <= 1.0, "Band image must be normalized to [0, 1]"


def test_iirs_04_module_isolation():
    """T-IIRS-04: Verify IIRS is strictly isolated from OHRC and TMC pipelines."""
    from src.matching import sift, rift, lightglue, crater

    assert not hasattr(sift, "IIRSMatcher"), "SIFT module must not expose IIRSMatcher"
    assert not hasattr(rift, "IIRSMatcher"), "RIFT module must not expose IIRSMatcher"
    assert not hasattr(lightglue, "IIRSMatcher"), "LightGlue module must not expose IIRSMatcher"
    assert not hasattr(crater, "IIRSMatcher"), "Crater module must not expose IIRSMatcher"

    configs_dir = _repo_root / "configs"
    if configs_dir.exists():
        for cfg_file in configs_dir.glob("*.yaml"):
            if cfg_file.name != "iirs_wac.yaml":
                content = cfg_file.read_text(errors="ignore")
                assert "src.matching.iirs" not in content, f"Config {cfg_file.name} must not reference iirs module"


def test_iirs_05_end_to_end_synthetic_registration():
    """T-IIRS-05: End-to-end IIRS registration on synthetic pair produces valid correspondences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qub_path = Path(tmpdir) / "synthetic_iirs.npz"
        write_synthetic_qub(
            out_path=qub_path,
            shape=(6, 128, 128),
            seed=42,
            solar_incidence_deg=35.0,
            emission_deg=5.0,
            phase_deg=35.0,
            gsd_m=80.0,
        )

        with np.load(str(qub_path)) as qub_data:
            base_slice = np.array(qub_data["cube"][0])
        wac_ref = np.roll(base_slice, shift=(4, 3), axis=(0, 1))

        config = {
            "iirs": {
                "results_dir": str(Path(tmpdir) / "results_iirs"),
                "photometric_correction": True,
                "accuracy_target_m": 80.0,
                "target_wavelength_nm": 643.0,
                "selection": {
                    "confidence_min": 0.0,
                    "grid_rows": 4,
                    "grid_cols": 4,
                    "cap_per_cell": 5,
                    "budget": 50,
                    "coverage_min": 0.2,
                },
            }
        }

        matcher = IIRSMatcher(config=config)
        result = matcher.run(
            qub_source=qub_path,
            wac_reference=wac_ref,
            pair_id="iirs_synth_001__wac",
            save_results=True,
        )

        assert result["sensor_pair"] == "IIRS-WAC"
        assert result["photometric_correction_applied"] is True
        assert "metrics" in result
        assert result["metrics"]["rmse_px"] < 10.0, f"RMSE in pixels too high: {result['metrics']['rmse_px']}"
        assert result["metrics"]["accuracy_target_m"] == 80.0


def test_iirs_06_result_separation():
    """T-IIRS-06: Verify results are stored in dedicated results/iirs/ directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qub_path = Path(tmpdir) / "synthetic_iirs.npz"
        write_synthetic_qub(out_path=qub_path, shape=(4, 64, 64), seed=99)

        wac_ref = np.random.default_rng(99).uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)

        out_iirs = Path(tmpdir) / "results" / "iirs"
        config = {
            "iirs": {
                "results_dir": str(out_iirs),
                "photometric_correction": True,
                "accuracy_target_m": 80.0,
            }
        }

        matcher = IIRSMatcher(config=config)
        result = matcher.run(
            qub_source=qub_path,
            wac_reference=wac_ref,
            pair_id="iirs_sep_test",
            save_results=True,
        )

        pair_json = out_iirs / "iirs_sep_test" / "iirs_result.json"
        assert pair_json.exists(), f"Expected result file at {pair_json}"
        assert not (Path(tmpdir) / "results" / "pair_results").exists(), "Must not write to main pair_results directory"


def test_iirs_07_leaderboard_tagging_and_accuracy():
    """T-IIRS-07: Verify IIRS leaderboard format and sub-80m target tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_iirs = Path(tmpdir) / "results" / "iirs"
        qub_path = Path(tmpdir) / "qub.npz"
        write_synthetic_qub(out_path=qub_path, shape=(4, 64, 64), seed=77, gsd_m=80.0)

        wac_ref = np.random.default_rng(77).uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)

        config = {
            "iirs": {
                "results_dir": str(out_iirs),
                "accuracy_target_m": 80.0,
            }
        }

        matcher = IIRSMatcher(config=config)
        matcher.run(qub_source=qub_path, wac_reference=wac_ref, pair_id="pair_lb_test", save_results=True)

        lb_file = out_iirs / "leaderboard.csv"
        assert lb_file.exists(), f"Expected leaderboard file at {lb_file}"

        with open(lb_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) >= 1, "Leaderboard must have at least 1 entry"
        entry = rows[0]
        assert entry["sensor_pair"] == "IIRS-WAC", f"Expected sensor_pair 'IIRS-WAC', got {entry['sensor_pair']}"
        assert "accuracy_target_m" in entry
        assert float(entry["accuracy_target_m"]) == 80.0
        assert "target_met" in entry


# ── Standalone CLI runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SIH 2026 — Phase 5 (IIRS Parallel Track) Test Suite")
    print("=" * 60)

    _test("T-IIRS-01: QUB reader & metadata parsing")(test_iirs_01_qub_reader)
    _test("T-IIRS-02: Hapke photometric correction")(test_iirs_02_hapke_correction)
    _test("T-IIRS-03: Band selection (closest to WAC 643nm)")(test_iirs_03_band_selection)
    _test("T-IIRS-04: Module isolation from OHRC/TMC")(test_iirs_04_module_isolation)
    _test("T-IIRS-05: End-to-end synthetic registration")(test_iirs_05_end_to_end_synthetic_registration)
    _test("T-IIRS-06: Results directory separation")(test_iirs_06_result_separation)
    _test("T-IIRS-07: Leaderboard tagging & accuracy tracking")(test_iirs_07_leaderboard_tagging_and_accuracy)

    print("\n" + "=" * 60)
    print("Phase 5 Test Summary")
    print("=" * 60)
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")

    if failed == 0:
        print(f"\n{GREEN}All Phase 5 tests passed successfully!{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}{failed} tests failed.{RESET}\n")
        sys.exit(1)

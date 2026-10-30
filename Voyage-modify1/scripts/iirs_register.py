#!/usr/bin/env python3
"""
scripts/iirs_register.py
========================
CLI Entry Point for Phase 5 — IIRS Parallel Track (Feature F24).

Executes end-to-end registration of Chandrayaan-2 IIRS hyperspectral data
against LRO WAC 643nm reference imagery.

Usage:
  python scripts/iirs_register.py \
      --qub data/raw/ch2_iir_nrc_..._qub.npz \
      --wac data/reference/wac_643nm.tif \
      --config configs/iirs_wac.yaml \
      --out results/iirs/

  # Run on synthetic benchmark data (for testing without raw PRADAN files):
  python scripts/iirs_register.py --synthetic --config configs/iirs_wac.yaml

Exit Codes (per PIPELINE.md §8):
  0: Success — registration completed, outputs saved, RMSE target evaluated
  1: Target / Gate failure — accuracy target not met or match count below minimum
  2: Configuration error — missing required fields or invalid YAML
  3: Environment error — missing dependencies
  4: Data leakage error
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

try:
    import yaml
except ImportError:
    print("[ERROR] PyYAML not installed. Please run: pip install pyyaml", file=sys.stderr)
    sys.exit(3)

from src.matching.iirs import (
    IIRSMatcher,
    IIRSMetadata,
    read_qub,
    write_synthetic_qub,
)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config YAML with optional inheritance."""
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ERROR] Failed to parse config YAML: {e}", file=sys.stderr)
        sys.exit(2)

    extends = cfg.get("extends")
    if extends and extends == "default":
        default_path = config_path.parent / "default.yaml"
        if default_path.exists():
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    base_cfg = yaml.safe_load(f) or {}
                merged = {**base_cfg, **cfg}
                return merged
            except Exception:
                pass
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SIH 2026 Phase 5: Chandrayaan-2 IIRS vs LRO WAC Registration Engine"
    )
    parser.add_argument(
        "--qub",
        type=str,
        default=None,
        help="Path to input IIRS QUB file (.qub, .npz, .npy, .hdr)",
    )
    parser.add_argument(
        "--wac",
        type=str,
        default=None,
        help="Path to LRO WAC 643nm reference image (.tif, .npz, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/iirs_wac.yaml",
        help="Path to IIRS config YAML (default: configs/iirs_wac.yaml)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/iirs",
        help="Output directory for IIRS registration artifacts (default: results/iirs)",
    )
    parser.add_argument(
        "--pair-id",
        type=str,
        default=None,
        help="Custom pair ID (optional)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on generated synthetic IIRS and WAC data (for test/offline mode)",
    )
    parser.add_argument(
        "--no-photometric",
        action="store_true",
        help="Disable Hapke photometric correction (for A/B ablation studies)",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _repo_root / cfg_path

    config = load_config(cfg_path)
    if "iirs" not in config:
        print("[ERROR] Invalid configuration: missing 'iirs' section.", file=sys.stderr)
        return 2

    if args.out:
        config["iirs"]["results_dir"] = str(args.out)
    if args.no_photometric:
        config["iirs"]["photometric_correction"] = False

    print("=" * 65)
    print("SIH 2026 — Phase 5: IIRS Parallel Track (Feature F24)")
    print("Chandrayaan-2 IIRS (~80m GSD) vs LRO WAC (643nm) Registration")
    print("=" * 65)
    print(f"Config: {cfg_path}")
    print(f"Results Directory: {config['iirs'].get('results_dir', 'results/iirs')}")
    print(f"Photometric Correction (Hapke): {config['iirs'].get('photometric_correction', True)}")

    matcher = IIRSMatcher(config=config)

    if args.synthetic or (not args.qub and not args.wac):
        print("\n[INFO] Running in synthetic simulation mode...")
        scratch_dir = _repo_root / "data" / "scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        synthetic_qub_path = scratch_dir / "synthetic_iirs_test.npz"

        write_synthetic_qub(
            out_path=synthetic_qub_path,
            shape=(12, 128, 128),
            seed=42,
            solar_incidence_deg=40.0,
            emission_deg=5.0,
            phase_deg=40.0,
            gsd_m=80.0,
        )

        qub_data = np.load(str(synthetic_qub_path))
        base_slice = qub_data["cube"][0]
        ref_slice = np.roll(base_slice, shift=(3, 2), axis=(0, 1))

        res = matcher.run(
            qub_source=synthetic_qub_path,
            wac_reference=ref_slice,
            pair_id=args.pair_id or "iirs_synthetic_test__wac_643nm",
            save_results=True,
        )
    else:
        if not args.qub:
            print("[ERROR] Missing --qub input path. Use --synthetic for test mode.", file=sys.stderr)
            return 2
        if not args.wac:
            print("[ERROR] Missing --wac reference path. Use --synthetic for test mode.", file=sys.stderr)
            return 2

        res = matcher.run(
            qub_source=Path(args.qub),
            wac_reference=Path(args.wac),
            pair_id=args.pair_id,
            save_results=True,
        )

    metrics = res["metrics"]
    print("\n" + "-" * 65)
    print("Registration Results:")
    print(f"  Pair ID              : {res['pair_id']}")
    print(f"  Sensor Pair          : {res['sensor_pair']}")
    print(f"  Photometric Corrected: {res['photometric_correction_applied']}")
    print(f"  Selected Band        : #{res['band_selection']['selected_band_index']} "
          f"({res['band_selection']['selected_wavelength_nm']:.1f} nm)")
    print(f"  Matches (Raw/Sel/In) : {metrics['candidate_count']} / {metrics['selected_count']} / {metrics['inlier_count']}")
    print(f"  Inlier Ratio         : {metrics['inlier_ratio'] * 100:.1f}%")
    print(f"  Spatial Coverage     : {metrics['spatial_coverage'] * 100:.1f}%")
    print(f"  RMSE (pixels)        : {metrics['rmse_px']:.3f} px")
    print(f"  RMSE (meters)        : {metrics['rmse_m']:.2f} m")
    print(f"  Accuracy Target      : < {metrics['accuracy_target_m']:.1f} m")
    print(f"  Target Met           : {'[YES - PASS]' if metrics['target_met'] else '[NO - FAIL]'}")
    print(f"  Runtime              : {metrics['runtime_s']:.2f} s")
    print("-" * 65)

    if metrics["target_met"]:
        print("\n[SUCCESS] IIRS Registration completed within accuracy threshold.")
        return 0
    else:
        print("\n[WARNING] Registration completed but accuracy target was not met.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

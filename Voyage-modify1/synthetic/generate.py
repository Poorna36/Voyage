#!/usr/bin/env python3
"""
synthetic/generate.py — Synthetic Benchmark Generator CLI
==========================================================

Generates synthetic image pairs with exact floating-point ground truth for the
Component-Wise Synthetic Benchmark (v3.0).

Usage:
    python -m synthetic.generate --phase 1 --synthetic-base
    python synthetic/generate.py --config configs/synthetic_benchmark.yaml --phase 1
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_synthetic_benchmark import main

if __name__ == "__main__":
    sys.exit(main())

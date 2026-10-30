#!/usr/bin/env python3
"""
synthetic/evaluate.py — Synthetic Benchmark Evaluation CLI
===========================================================

Evaluates correspondence pipeline outputs against hidden GT coordinates.

Usage:
    python -m synthetic.evaluate --manifest data/synthetic/synthetic_manifest.jsonl
    python synthetic/evaluate.py --results results/ --out results/synthetic_benchmark/
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_synthetic import main

if __name__ == "__main__":
    sys.exit(main())

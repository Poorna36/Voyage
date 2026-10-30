#!/usr/bin/env python3
"""
synthetic/runner.py — Synthetic Benchmark Orchestrator
======================================================

Orchestrates synthetic generation, blind matching pipeline execution,
evaluation against hidden ground truth, and statistical scorecard generation.

Usage:
    python -m synthetic.runner --phase 1 --synthetic-base
    python synthetic/runner.py --phase 1 --matchers sift lightglue
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_synthetic_benchmark import main

if __name__ == "__main__":
    sys.exit(main())

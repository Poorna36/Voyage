#!/usr/bin/env python3
"""
scripts/run_synthetic_benchmark.py — End-to-End Synthetic Benchmark Orchestrator

Runs the complete blind synthetic benchmark pipeline:
  1. Generate synthetic pairs (calls generate_synthetic_benchmark.py)
  2. Run correspondence pipeline (calls benchmark.py via subprocess)
  3. Evaluate all stages against hidden GT (calls eval_synthetic.py)
  4. Aggregate per-condition with 95% CIs
  5. Write synthetic_component_report.csv and synthetic_benchmark_summary.md

Usage:
    python scripts/run_synthetic_benchmark.py \\
        --config  configs/synthetic_benchmark.yaml \\
        --phase   1 \\
        --matchers sift lightglue \\
        --images  data/raw/ \\
        --out-dir results/synthetic_benchmark/ \\
        [--skip-generate]  \\
        [--skip-benchmark] \\
        [--skip-eval]      \\
        [--force]          \\
        [-v]

Exit codes (per PIPELINE.md §8):
    0 — all phases completed successfully
    1 — partial failure (some pairs failed; CSV still written)
    2 — config/argument error
    3 — critical environment error (no source images, missing scripts)
    4 — leakage audit failure
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from src.evaluation.synthetic_eval import aggregate_scorecards, StageScorecard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_synthetic_benchmark")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run_script(cmd: List[str], step_name: str, cwd: Optional[Path] = None) -> int:
    """Run a Python script as a subprocess, stream its output, return exit code."""
    logger.info(">>> %s: %s", step_name, " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd or _ROOT),
        text=True,
    )
    if result.returncode not in (0, 1):  # 0=ok, 1=partial failures (allowed)
        logger.error(
            "%s exited with code %d.", step_name, result.returncode
        )
    else:
        logger.info("%s completed (exit=%d).", step_name, result.returncode)
    return result.returncode


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _aggregate_eval_summary(
    eval_summary_path: Path,
    matchers: List[str],
) -> Dict[str, Dict[str, Dict]]:
    """Aggregate eval_summary.jsonl into per-(matcher, stage) statistics.

    Returns nested dict: {matcher: {stage_metric: aggregate_dict}}
    where aggregate_dict has mean, ci_low, ci_high, n keys.
    """
    records = _load_jsonl(eval_summary_path)
    if not records:
        return {}

    # Group by (matcher, stage prefix)
    # eval_summary keys are like "L2_gt_recall", "L4_inlier_precision", etc.
    stage_prefixes = ["L1.5", "L2", "L3", "L4", "L5"]

    # Group records by matcher
    by_matcher: Dict[str, List[dict]] = {}
    for rec in records:
        m = rec.get("matcher", "unknown")
        by_matcher.setdefault(m, []).append(rec)

    result: Dict[str, Dict[str, Dict]] = {}

    for matcher, recs in by_matcher.items():
        result[matcher] = {}
        # Collect scorecards per stage
        for stage in stage_prefixes:
            prefix = stage.replace(".", "_") + "_"
            metric_vals: Dict[str, List[float]] = {}
            for rec in recs:
                for k, v in rec.items():
                    if k.startswith(prefix.replace(".", "_")):
                        metric = k[len(prefix):]
                        try:
                            fv = float(v)
                            if np.isfinite(fv):
                                metric_vals.setdefault(metric, []).append(fv)
                        except (TypeError, ValueError):
                            pass

            # Build aggregate
            stage_agg: Dict[str, float] = {}
            for metric, vals in metric_vals.items():
                arr = np.array(vals, dtype=np.float64)
                n = len(arr)
                mean = float(np.mean(arr))
                se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                z = 1.96  # 95% CI
                stage_agg[f"{metric}_mean"] = mean
                stage_agg[f"{metric}_ci_low"] = mean - z * se
                stage_agg[f"{metric}_ci_high"] = mean + z * se
                stage_agg[f"{metric}_n"] = float(n)
            if stage_agg:
                result[matcher][stage] = stage_agg

    return result


def _write_csv_report(
    agg: Dict[str, Dict[str, Dict]],
    csv_path: Path,
    phase: int,
) -> None:
    """Write synthetic_component_report.csv.

    Columns: phase, matcher, stage, metric, mean, ci_low, ci_high, n
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for matcher, stages in agg.items():
        for stage, metrics in stages.items():
            # Group metric_mean/ci_low/ci_high/n into rows
            metric_names = set()
            for key in metrics:
                for suffix in ("_mean", "_ci_low", "_ci_high", "_n"):
                    if key.endswith(suffix):
                        metric_names.add(key[: -len(suffix)])
            for metric in sorted(metric_names):
                rows.append({
                    "phase": phase,
                    "matcher": matcher,
                    "stage": stage,
                    "metric": metric,
                    "mean": metrics.get(f"{metric}_mean", float("nan")),
                    "ci_low": metrics.get(f"{metric}_ci_low", float("nan")),
                    "ci_high": metrics.get(f"{metric}_ci_high", float("nan")),
                    "n": int(metrics.get(f"{metric}_n", 0)),
                })

    if not rows:
        logger.warning("No aggregate data — CSV will be empty header only.")
        rows = [{"phase": phase, "matcher": "", "stage": "", "metric": "", "mean": "", "ci_low": "", "ci_high": "", "n": ""}]

    fieldnames = ["phase", "matcher", "stage", "metric", "mean", "ci_low", "ci_high", "n"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV report written: %s (%d rows).", csv_path, len(rows))


def _write_markdown_summary(
    agg: Dict[str, Dict[str, Dict]],
    md_path: Path,
    phase: int,
    n_pairs: int,
    n_failed: int,
    matchers: List[str],
    generated_at: str,
) -> None:
    """Write synthetic_benchmark_summary.md — narrative + statistical tables."""
    md_path.parent.mkdir(parents=True, exist_ok=True)

    stages_display = {
        "L1.5": "Matcher Selection (MSM Routing)",
        "L2":   "Raw Matcher Capacity",
        "L3":   "Spatial Filter Survival",
        "L4":   "Geometric Verification",
        "L5":   "Sub-Pixel Refinement",
    }

    key_metrics = {
        "L2":   [("gt_recall", "GT Recall", ".3f"), ("raw_rmse_px", "Raw RMSE (px)", ".4f")],
        "L3":   [("gt_survival_rate", "GT Survival Rate", ".3f"), ("fp_pruning_rate", "FP Pruning Rate", ".3f")],
        "L4":   [("model_rmse_px", "Model RMSE (px)", ".4f"), ("model_pct_lt_0p5px", "% < 0.5px", ".1f"), ("inlier_precision", "Inlier Precision", ".3f"), ("inlier_recall", "Inlier Recall", ".3f")],
        "L5":   [("refinement_gain_px", "Refinement Gain (px)", "+.4f"), ("pct_improved", "% Improved", ".3f"), ("pct_lt_0p5px", "% < 0.5px", ".3f")],
        "L1.5": [("routing_correct", "Routing Correct", ".3f")],
    }

    lines = [
        f"# SIH26166 — Synthetic GT Benchmark Report",
        f"",
        f"> **Phase {phase}** | Generated: {generated_at}  ",
        f"> Architecture: `docs/SYNTHETIC_BENCHMARK_ARCHITECTURE.md` (v3.0)",
        f"",
        f"## Run Summary",
        f"",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Benchmark Phase | {phase} |",
        f"| Pairs evaluated | {n_pairs} |",
        f"| Pairs failed | {n_failed} |",
        f"| Matchers | {', '.join(matchers)} |",
        f"",
        f"---",
        f"",
    ]

    for stage, stage_label in stages_display.items():
        lines.append(f"## {stage} — {stage_label}")
        lines.append(f"")

        has_data = any(stage in stages for stages in agg.values())
        if not has_data:
            lines.append(f"*No data — pipeline results not available for this stage.*")
            lines.append(f"")
            continue

        # Build table
        stage_key_metrics = key_metrics.get(stage, [])
        if not stage_key_metrics:
            lines.append(f"*No key metrics defined for this stage.*")
            lines.append(f"")
            continue

        # Header
        header_cols = ["Matcher"] + [label for _, label, _ in stage_key_metrics]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

        for matcher in matchers:
            stage_data = agg.get(matcher, {}).get(stage, {})
            row = [matcher]
            for metric_key, _, fmt in stage_key_metrics:
                mean_val = stage_data.get(f"{metric_key}_mean")
                ci_low = stage_data.get(f"{metric_key}_ci_low")
                ci_high = stage_data.get(f"{metric_key}_ci_high")
                n = int(stage_data.get(f"{metric_key}_n", 0))
                if mean_val is None or not np.isfinite(mean_val):
                    row.append("N/A")
                else:
                    cell = f"{mean_val:{fmt}}"
                    if n > 1 and ci_low is not None and ci_high is not None:
                        cell += f" [{ci_low:{fmt}}, {ci_high:{fmt}}] (n={n})"
                    row.append(cell)
            lines.append("| " + " | ".join(row) + " |")

        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## Notes",
        f"",
        f"- All metrics are mean ± 95% CI (±1.96 SE) across N independent seeds.",
        f"- GT assignment uses 1-to-1 Hungarian matching within {2.0} px radius.",
        f"- GT coordinates are exact floating-point; never approximated.",
        f"- See `results/synthetic_benchmark/synthetic_component_report.csv` for full data.",
        f"",
        f"*Report generated by `scripts/run_synthetic_benchmark.py` — SIH26166 Phase 10 v3.0*",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown summary written: %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end synthetic benchmark orchestrator (Phase 10 v3.0)."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_benchmark.yaml"),
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], default=1,
    )
    parser.add_argument(
        "--matchers", nargs="+",
        default=["sift", "rift2", "lightglue"],
        help="Matchers to run through the benchmark pipeline.",
    )
    parser.add_argument(
        "--images", type=Path, default=Path("data/raw/"),
        help="Source images directory for generate step.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/synthetic_benchmark/"),
    )
    parser.add_argument(
        "--synthetic-dir", type=Path, default=Path("data/synthetic/"),
    )
    parser.add_argument(
        "--pipeline-results", type=Path, default=Path("results/"),
        help="Root dir where benchmark.py writes its results.",
    )
    parser.add_argument(
        "--out-csv", type=Path, default=None,
        help="Override output CSV path (default: --out-dir/synthetic_component_report.csv).",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip synthetic pair generation (use existing data/synthetic/).",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Skip benchmark.py execution (use existing results/).",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip eval_synthetic.py (use existing eval_summary.jsonl).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Pass --force to generate and benchmark steps.",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Limit source images processed by generate step.",
    )
    parser.add_argument(
        "--synthetic-base", action="store_true",
        help="Synthesize a realistic lunar terrain base image if no source images are found.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 2

    csv_path = args.out_csv or (args.out_dir / "synthetic_component_report.csv")
    md_path = args.out_dir / "synthetic_benchmark_summary.md"
    manifest_path = args.synthetic_dir / "synthetic_manifest.jsonl"
    eval_summary_path = args.out_dir / "eval_summary.jsonl"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    overall_exit = 0

    # -------------------------------------------------------------------------
    # Step 1: Generate synthetic pairs
    # -------------------------------------------------------------------------
    if not args.skip_generate:
        gen_cmd = [
            sys.executable, str(_HERE / "generate_synthetic_benchmark.py"),
            "--config", str(args.config),
            "--images", str(args.images),
            "--out", str(args.synthetic_dir),
            "--phase", str(args.phase),
        ]
        if args.force:
            gen_cmd.append("--force")
        if args.max_pairs:
            gen_cmd += ["--max-pairs", str(args.max_pairs)]
        if args.synthetic_base:
            gen_cmd.append("--synthetic-base")
        if args.verbose:
            gen_cmd.append("-v")

        rc = _run_script(gen_cmd, "STEP 1: generate_synthetic_benchmark")
        if rc >= 2:
            logger.error("Generation failed critically (exit=%d). Aborting.", rc)
            return rc
        if rc == 4:
            return 4  # leakage
        if rc == 1:
            overall_exit = 1  # partial, continue

    # -------------------------------------------------------------------------
    # Step 2: Run correspondence pipeline on synthetic pairs
    # -------------------------------------------------------------------------
    if not args.skip_benchmark:
        if not manifest_path.exists():
            logger.error("No synthetic manifest at %s — cannot run benchmark.", manifest_path)
            return 3

        matchers_yaml = Path("configs/matchers.yaml")
        if not matchers_yaml.exists():
            matchers_yaml = Path("configs/matchers_sift.yaml")

        bench_cmd = [
            sys.executable, str(_HERE / "benchmark.py"),
            "--manifest", str(manifest_path),
            "--matchers", str(matchers_yaml),
            "--out", str(args.pipeline_results),
            "--mode", "benchmark",
            "--splits", "train",
        ]
        if args.force:
            bench_cmd.append("--force")
        if args.verbose:
            bench_cmd.append("-v")

        rc = _run_script(bench_cmd, "STEP 2: benchmark.py (S4/S5)")
        if rc >= 2:
            logger.error("Benchmark failed critically (exit=%d).", rc)
            overall_exit = max(overall_exit, 1)
        elif rc == 1:
            overall_exit = 1

        # Run S6 + S7: Geometric verification and sub-pixel refinement
        s6_s7_cmd = [
            sys.executable, str(_HERE / "run_s6_s7.py"),
            "--manifest", str(manifest_path),
            "--results", str(args.pipeline_results),
            "--data-dir", str(args.synthetic_dir),
            "--matchers", *args.matchers,
        ]
        rc_s6 = _run_script(s6_s7_cmd, "STEP 2.5: run_s6_s7.py (S6/S7)")
        if rc_s6 >= 2:
            logger.error("S6/S7 failed critically (exit=%d).", rc_s6)
            overall_exit = max(overall_exit, 1)
        elif rc_s6 == 1:
            overall_exit = 1

    # -------------------------------------------------------------------------
    # Step 3: Evaluate all stages against hidden GT
    # -------------------------------------------------------------------------
    if not args.skip_eval:
        gt_dir = args.synthetic_dir / "gt"
        eval_cmd = [
            sys.executable, str(_HERE / "eval_synthetic.py"),
            "--manifest", str(manifest_path),
            "--results", str(args.pipeline_results),
            "--gt-dir", str(gt_dir),
            "--out", str(args.out_dir),
            "--config", str(args.config),
            "--matchers", *args.matchers,
        ]
        if args.verbose:
            eval_cmd.append("-v")

        rc = _run_script(eval_cmd, "STEP 3: eval_synthetic.py")
        if rc >= 2:
            logger.error("Evaluation failed critically (exit=%d).", rc)
            overall_exit = max(overall_exit, 1)
        elif rc == 1:
            overall_exit = 1

    # -------------------------------------------------------------------------
    # Step 4: Aggregate + report
    # -------------------------------------------------------------------------
    logger.info("STEP 4: Aggregating results and writing reports...")

    manifest_records = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        manifest_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    n_pairs = len(manifest_records)
    n_failed = 0
    failures_path = args.out_dir / "eval_failures.jsonl"
    if failures_path.exists():
        with open(failures_path) as f:
            n_failed = sum(1 for line in f if line.strip())

    agg = _aggregate_eval_summary(eval_summary_path, args.matchers)

    _write_csv_report(agg, csv_path, args.phase)
    _write_markdown_summary(
        agg=agg,
        md_path=md_path,
        phase=args.phase,
        n_pairs=n_pairs,
        n_failed=n_failed,
        matchers=args.matchers,
        generated_at=generated_at,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║   SIH26166 Synthetic Benchmark — Phase %d Complete    ║\n"
        "╠══════════════════════════════════════════════════════╣\n"
        "║  Pairs evaluated : %-33d║\n"
        "║  Pairs failed    : %-33d║\n"
        "║  CSV report      : %-33s║\n"
        "║  MD summary      : %-33s║\n"
        "╚══════════════════════════════════════════════════════╝",
        args.phase,
        n_pairs, n_failed,
        csv_path.name, md_path.name,
    )

    return overall_exit


if __name__ == "__main__":
    sys.exit(main())

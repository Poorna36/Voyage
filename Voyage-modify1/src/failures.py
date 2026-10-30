"""
src/failures.py
===============
Centralized Gate Failure & Error Handling Logger.

Enforces PROGRESS.md §6.3:
  - All gate failures caught and written to failures.jsonl
  - Structure: pair_id, matcher, stage, reason, fallback_taken, created_at, extra
  - Append-only — files/lines are NEVER overwritten or deleted
  - Safe execution: logging never raises exceptions that would halt pipeline execution

References:
  - PIPELINE.md §8 (Failure Handling)
  - PROGRESS.md §6.3 (Error Handling)
  - ARCHITECTURE.md §7 (Arbitration & Fallbacks)
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Standard failure log record for failures.jsonl."""
    pair_id: str
    stage: str
    reason: str
    matcher: str = "pipeline"
    fallback_taken: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten extra into root for clean JSONL querying while preserving schema
        extra_fields = d.pop("extra", {})
        if extra_fields:
            d.update(extra_fields)
        return d


def log_gate_failure(
    destination: Union[str, Path],
    pair_id: str,
    stage: str,
    reason: str,
    matcher: str = "pipeline",
    fallback_taken: Optional[str] = None,
    **extra: Any,
) -> Path:
    """
    Append a single failure record to failures.jsonl.

    Parameters:
      destination: Path to failures.jsonl file OR parent directory
      pair_id: Unique pair identifier
      stage: Pipeline stage (e.g. S1, S2, S3, S4, S5, S6, S7, S8)
      reason: Human/machine readable reason string
      matcher: Matcher ID if applicable (e.g. sift, rift2, lightglue, crater)
      fallback_taken: Description of fallback path taken (or None)
      **extra: Additional diagnostic metadata

    Returns:
      Path to the updated failures.jsonl file
    """
    dest_path = Path(destination)
    if dest_path.is_dir() or not dest_path.suffix:
        failures_file = dest_path / "failures.jsonl"
    else:
        failures_file = dest_path

    try:
        failures_file.parent.mkdir(parents=True, exist_ok=True)
        rec = FailureRecord(
            pair_id=pair_id,
            stage=stage,
            reason=reason,
            matcher=matcher,
            fallback_taken=fallback_taken,
            extra=extra,
        )
        with open(failures_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
    except Exception as exc:
        logger.error("Failed to write failure log to %s: %s", failures_file, exc)

    return failures_file


def read_failures(destination: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read and parse all lines from failures.jsonl.

    Parameters:
      destination: Path to failures.jsonl file OR parent directory

    Returns:
      List of parsed JSON failure records
    """
    dest_path = Path(destination)
    failures_file = dest_path / "failures.jsonl" if dest_path.is_dir() else dest_path

    if not failures_file.exists():
        return []

    records = []
    with open(failures_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("failures.jsonl:%d: Invalid JSON - %s", lineno, exc)
    return records

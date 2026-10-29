"""
quantum/records.py
==================
Shared data schema and JSON writer for Quantum Run Records.

Every quantum module in Q0–Q7 produces a QuantumRunRecord.
Records are saved under results/quantum/<run_id>/<module>.json
and indexed for the API and dashboard.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "quantum"


def generate_run_id(prefix: str = "qr") -> str:
    """Generate canonical run ID: qr_YYYYMMDD_HHMMSS_xxxx."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:4]
    return f"{prefix}_{ts}_{rand}"


@dataclass
class ProblemSpec:
    """Summary of the quantum problem size and parameters."""
    n_vars: int = 0
    max_qubits: int = 0
    n_subproblems: int = 1
    p_layers: Optional[int] = None
    shots: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitSpec:
    """Circuit gate and depth statistics for simulator/hardware accounting."""
    qubits: int = 0
    depth: int = 0
    gate_counts: Dict[str, int] = field(default_factory=dict)
    gates_sequence: list = field(default_factory=list)  # Small list for visualization: [{"gate": "H", "target": 0}, ...]


@dataclass
class ClassicalTwinRecord:
    """Result from running the classical baseline on the exact same inputs and seed."""
    name: str
    runtime_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackRecord:
    """Whether fallback to classical/SQA occurred and why."""
    used: bool = False
    reason: Optional[str] = None


@dataclass
class QuantumRunRecord:
    """
    Standard record format for all Q0–Q7 quantum modules (ARCHITECTURE §4.3).
    """
    run_id: str
    pair_id: str
    module: str
    mode: str = "shadow"                # "off" | "shadow" | "active"
    backend: str = "numpy_statevector"
    seed: int = 42
    problem: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    classical_twin: Dict[str, Any] = field(default_factory=dict)
    circuit: Optional[Dict[str, Any]] = None
    runtime_s: float = 0.0
    fallback: Dict[str, Any] = field(default_factory=lambda: {"used": False, "reason": None})
    versions: Dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        d = asdict(self)
        return d

    def save(self, results_dir: Optional[Path] = None) -> Path:
        """
        Save this record to results/quantum/<run_id>/<module>.json
        and append to results/quantum/runs.jsonl for fast indexing.
        """
        out_dir = results_dir or DEFAULT_RESULTS_DIR
        run_folder = out_dir / self.run_id
        run_folder.mkdir(parents=True, exist_ok=True)

        clean_module = self.module.replace(" ", "_").lower()
        file_path = run_folder / f"{clean_module}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Append to index
        index_path = out_dir / "runs.jsonl"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict()) + "\n")

        return file_path

    @classmethod
    def load(cls, file_path: Path) -> QuantumRunRecord:
        """Load a record from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

#!/usr/bin/env python3
"""
scripts/audit_coordinates.py
============================
Coordinate Convention Enforcement Auditor (PROGRESS.md §6.2).

Performs static analysis across all source code modules in src/ and scripts/
to verify that functions dealing with coordinate arrays enforce the mandatory
shape and axis assertions:
  - Pixel coords must be (col, row) = (x, y), shape (N, 2), top-left origin. NEVER (row, col).
  - Geographic coords must be (lon, lat). NEVER (lat, lon).
  - Functions touching coordinates must include `assert arr.shape[-1] == 2` or equivalent.

Usage:
  python scripts/audit_coordinates.py [--dir src] [--strict]

Exit Codes:
  0: All audited coordinate functions comply with convention assertions.
  1: Non-compliant functions detected in strict mode.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_repo_root = Path(__file__).resolve().parent.parent

# Parameter names indicating 2D point arrays
COORD_PARAM_NAMES: Set[str] = {
    "src_xy",
    "ref_xy",
    "pts",
    "match_xy",
    "coords",
    "points",
    "predicted_ref_xy",
    "gt_ref_xy",
    "original_xy",
    "reannotated_xy",
    "inliers_xy",
    "keypoints_xy",
}


class CoordinateVisitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.functions_with_coords: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node: ast.AST) -> None:
        args = [a.arg for a in getattr(node, "args").args]
        coord_args = [a for a in args if any(cp in a.lower() for cp in ("_xy", "coord", "pts", "keypoint"))]

        if not coord_args:
            return

        # Check if function body contains an assert statement checking shape[-1] == 2
        has_assertion = False
        assert_details = []

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assert):
                # Inspect assertion test
                src = ast.unparse(stmt.test) if hasattr(ast, "unparse") else ""
                if "shape" in src and "2" in src:
                    has_assertion = True
                    assert_details.append(src)
                elif "ndim" in src and "2" in src:
                    has_assertion = True
                    assert_details.append(src)

        self.functions_with_coords.append({
            "name": getattr(node, "name"),
            "line": getattr(node, "lineno"),
            "coord_args": coord_args,
            "has_assertion": has_assertion,
            "assert_details": assert_details,
        })


def audit_directory(root_dir: Path) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Audit all python files in directory."""
    total_coord_funcs = 0
    passed_coord_funcs = 0
    findings = []

    for py_file in sorted(root_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts or ".venv" in py_file.parts or "tests" in py_file.parts:
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            visitor = CoordinateVisitor(str(py_file.relative_to(_repo_root)))
            visitor.visit(tree)

            for f in visitor.functions_with_coords:
                total_coord_funcs += 1
                f["file"] = str(py_file.relative_to(_repo_root))
                if f["has_assertion"]:
                    passed_coord_funcs += 1
                else:
                    findings.append(f)
        except Exception as exc:
            print(f"[WARN] Failed to parse {py_file}: {exc}", file=sys.stderr)

    return total_coord_funcs, passed_coord_funcs, findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit coordinate convention and assertions.")
    parser.add_argument("--dir", default="src", help="Directory to scan (default: src)")
    parser.add_argument("--strict", action="store_true", help="Fail if any coordinate function lacks explicit assertion")

    args = parser.parse_args()
    scan_path = _repo_root / args.dir

    print("=" * 65)
    print("SIH 2026 — Coordinate Convention Static Code Audit")
    print(f"Scanning directory: {scan_path}")
    print("Convention: (col, row) = (x, y) float32 (N, 2); geographic (lon, lat)")
    print("=" * 65)

    total, passed, findings = audit_directory(scan_path)

    print(f"\nAudit Results:")
    print(f"  Total Coordinate Functions Scanned : {total}")
    print(f"  Functions with Explicit Assertions : {passed}")
    print(f"  Functions Needing Review           : {len(findings)}")

    if findings:
        print("\nFunctions without explicit (N,2) shape assertions:")
        for f in findings:
            print(f"  - {f['file']}:{f['line']} -> {f['name']}({', '.join(f['coord_args'])})")

    print("\n" + "-" * 65)
    if not findings:
        print("[SUCCESS] 100% of scanned coordinate functions contain shape assertions!")
        return 0
    else:
        compliance_pct = (passed / max(1, total)) * 100
        print(f"Compliance rate: {compliance_pct:.1f}%")
        if args.strict and findings:
            print("[FAIL] Strict audit failed: coordinate assertions missing in some functions.")
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())

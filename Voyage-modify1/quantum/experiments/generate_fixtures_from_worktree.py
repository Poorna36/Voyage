import sys
import os
import json
import numpy as np

# Path to the worktree
worktree_root = r"D:\work\voyage_worktree_218c16e\Voyage-modify1"
sys.path.insert(0, worktree_root)

from quantum.service import QuantumService

svc = QuantumService()

# 1. Default problem
rec_default = svc.run_module(pair_id="baseline_default", module="Q1_selection", seed=42)
fixture_default = {
    "commit": "218c16ed3bb11114b986575f2c45a6baddf43f8c",
    "pair_id": rec_default.pair_id,
    "module": rec_default.module,
    "problem": rec_default.problem,
    "metrics": rec_default.metrics,
    "classical_twin": rec_default.classical_twin,
    "circuit": rec_default.circuit,
}

# 2. Problem where cells exceed cap
rng = np.random.default_rng(42)
N = 30
s_xy = rng.uniform(0, 500, size=(N, 2)).astype(np.float32)
r_xy = s_xy + np.array([4.0, -2.0], dtype=np.float32)
conf = rng.uniform(0.5, 0.99, size=N).astype(np.float32)

rec_capped = svc.run_module(
    pair_id="baseline_capped",
    module="Q1_selection",
    params={"grid_n": 4, "cap_per_cell": 3, "p_layers": 2, "restarts": 3},
    inputs={"src_xy": s_xy, "ref_xy": r_xy, "confidence": conf},
    seed=42,
)
fixture_capped = {
    "commit": "218c16ed3bb11114b986575f2c45a6baddf43f8c",
    "pair_id": rec_capped.pair_id,
    "module": rec_capped.module,
    "problem": rec_capped.problem,
    "metrics": rec_capped.metrics,
    "classical_twin": rec_capped.classical_twin,
    "circuit": rec_capped.circuit,
    "inputs": {
        "src_xy": s_xy.tolist(),
        "ref_xy": r_xy.tolist(),
        "confidence": conf.tolist(),
    }
}

out_dir = r"D:\work\voyage\Voyage-modify1\tests\fixtures"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "q1_baseline_default_218c16e.json"), "w") as f:
    json.dump(fixture_default, f, indent=2)

with open(os.path.join(out_dir, "q1_baseline_capped_218c16e.json"), "w") as f:
    json.dump(fixture_capped, f, indent=2)

print("Fixtures saved successfully!")
print("Default metrics:", fixture_default["metrics"])
print("Capped metrics:", fixture_capped["metrics"])

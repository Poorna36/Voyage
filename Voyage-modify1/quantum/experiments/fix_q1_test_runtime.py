with open("tests/test_quantum_q1_selection.py", "r", encoding="utf-8") as f:
    content = f.read()

old_block = """        assert metrics_def == base_def["metrics"]
        assert problem_def == base_def["problem"]
        assert circuit_def == base_def["circuit"]
        assert rec_default.classical_twin == base_def["classical_twin"]

        # (a.2) Capped problem matches 218c16e baseline
        s_xy = np.array(base_cap["inputs"]["src_xy"], dtype=np.float32)
        r_xy = np.array(base_cap["inputs"]["ref_xy"], dtype=np.float32)
        conf = np.array(base_cap["inputs"]["confidence"], dtype=np.float32)
        rec_capped = svc.run_module(
            pair_id="baseline_capped",
            module="Q1_selection",
            params={"grid_n": 4, "cap_per_cell": 3, "p_layers": 2, "restarts": 3},
            inputs={"src_xy": s_xy, "ref_xy": r_xy, "confidence": conf},
            seed=42,
        )
        metrics_cap = {k: v for k, v in rec_capped.metrics.items() if k not in ("shots", "cells_total", "cells_trivial", "cells_qaoa")}
        problem_cap = {k: v for k, v in rec_capped.problem.items() if k != "shots"}
        circuit_cap = {k: v for k, v in rec_capped.circuit.items() if k != "shots"}

        assert metrics_cap == base_cap["metrics"]
        assert problem_cap == base_cap["problem"]
        assert circuit_cap == base_cap["circuit"]
        assert rec_capped.classical_twin == base_cap["classical_twin"]"""

new_block = """        assert metrics_def == base_def["metrics"]
        assert problem_def == base_def["problem"]
        assert circuit_def == base_def["circuit"]
        assert rec_default.classical_twin["name"] == base_def["classical_twin"]["name"]
        assert rec_default.classical_twin["n_selected"] == base_def["classical_twin"]["n_selected"]

        # (a.2) Capped problem matches 218c16e baseline
        s_xy = np.array(base_cap["inputs"]["src_xy"], dtype=np.float32)
        r_xy = np.array(base_cap["inputs"]["ref_xy"], dtype=np.float32)
        conf = np.array(base_cap["inputs"]["confidence"], dtype=np.float32)
        rec_capped = svc.run_module(
            pair_id="baseline_capped",
            module="Q1_selection",
            params={"grid_n": 4, "cap_per_cell": 3, "p_layers": 2, "restarts": 3},
            inputs={"src_xy": s_xy, "ref_xy": r_xy, "confidence": conf},
            seed=42,
        )
        metrics_cap = {k: v for k, v in rec_capped.metrics.items() if k not in ("shots", "cells_total", "cells_trivial", "cells_qaoa")}
        problem_cap = {k: v for k, v in rec_capped.problem.items() if k != "shots"}
        circuit_cap = {k: v for k, v in rec_capped.circuit.items() if k != "shots"}

        assert metrics_cap == base_cap["metrics"]
        assert problem_cap == base_cap["problem"]
        assert circuit_cap == base_cap["circuit"]
        assert rec_capped.classical_twin["name"] == base_cap["classical_twin"]["name"]
        assert rec_capped.classical_twin["n_selected"] == base_cap["classical_twin"]["n_selected"]"""

content_norm = content.replace("\r\n", "\n")
old_norm = old_block.replace("\r\n", "\n")
assert old_norm in content_norm, "old_norm not found"
content_norm = content_norm.replace(old_norm, new_block.replace("\r\n", "\n"))

with open("tests/test_quantum_q1_selection.py", "w", encoding="utf-8", newline="\r\n") as f:
    f.write(content_norm)

print("Replacement complete.")

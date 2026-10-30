import json
from pathlib import Path

processed = Path("data/processed")
for gt in processed.glob("*/ground_truth.json"):
    try:
        d = json.loads(gt.read_text(encoding="utf-8"))
        kps = d.get("keypoints", [])
        inls = [k for k in kps if k.get("is_inlier")]
        benchmarks = d.get("matcher_benchmarks", {})
        print(f"[{gt.parent.name}]")
        print(f"  total keypoints: {len(kps)}, inliers: {len(inls)} ({len(inls)/len(kps)*100:.1f}%)" if kps else "  no keypoints")
        print(f"  rmse_px: {d.get('rmse_px')}")
        if benchmarks:
            print(f"  benchmarks: {list(benchmarks.keys())}")
            for m, b in benchmarks.items():
                print(f"    {m}: inliers={b.get('inliers')}/{b.get('total') or b.get('candidates')}, ratio={b.get('inlier_ratio')}, rmse={b.get('rmse_px')}")
    except Exception as e:
        print(f"Error reading {gt}: {e}")

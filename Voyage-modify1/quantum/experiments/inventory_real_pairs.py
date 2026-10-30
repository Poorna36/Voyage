import sys
import os
import json
from pathlib import Path

# Search in Voyage-modify1 and parents
search_dirs = [Path("."), Path("../Voyage-modify1"), Path(".."), Path("../../voyage")]

manifest_paths = []
meta_paths = []

for base in [Path("."), Path("..")]:
    for p in base.rglob("manifest.jsonl"):
        if "node_modules" not in str(p) and ".git" not in str(p):
            manifest_paths.append(p.resolve())
    for p in base.rglob("meta.json"):
        if "node_modules" not in str(p) and ".git" not in str(p):
            meta_paths.append(p.resolve())

print("Manifest paths found:", manifest_paths)
print("Meta.json paths found:", meta_paths)

# Let's inspect found manifests
for mp in manifest_paths:
    print(f"\nInspecting manifest: {mp}")
    with open(mp, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Total lines/records: {len(lines)}")
    sensor_pairs = {}
    for l in lines:
        try:
            rec = json.loads(l)
            sp = rec.get("sensor_pair") or (rec.get("src", {}).get("sensor", "") + "-" + rec.get("ref", {}).get("type", ""))
            sensor_pairs[sp] = sensor_pairs.get(sp, 0) + 1
        except Exception:
            pass
    print("Sensor pairs in manifest:", sensor_pairs)

# Inspect meta.json files
for mp in meta_paths:
    print(f"\nMeta.json at: {mp}")
    try:
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("Keys:", list(data.keys())[:15])
    except Exception as e:
        print("Error reading:", e)

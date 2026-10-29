import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

results = {}

# Test 1
try:
    res = client.post("/api/pipeline/run", json={"pair_id": "nonexistent_pair_12345", "matcher": "sift"})
    results["test_1_pipeline_nonexistent"] = {
        "status_code": res.status_code,
        "json": res.json(),
    }
except Exception as e:
    results["test_1_pipeline_nonexistent"] = {"error": str(e)}

# Test 2
try:
    res = client.post("/api/pipeline/run", json={"pair_id": "boguslawsky", "matcher": "invalid_magic_matcher"})
    results["test_2_pipeline_invalid_matcher"] = {
        "status_code": res.status_code,
        "json": res.json(),
    }
except Exception as e:
    results["test_2_pipeline_invalid_matcher"] = {"error": str(e)}

# Test 3
try:
    res = client.get("/api/science/craters/?q=alien_base_nonexistent")
    results["test_3_craters_zero_results"] = {
        "status_code": res.status_code,
        "count": len(res.json()),
        "json": res.json(),
    }
except Exception as e:
    results["test_3_craters_zero_results"] = {"error": str(e)}

# Test 4
try:
    res = client.get("/api/science/craters/nonexistent_crater_xyz")
    results["test_4_crater_single_nonexistent"] = {
        "status_code": res.status_code,
        "json": res.json(),
    }
except Exception as e:
    results["test_4_crater_single_nonexistent"] = {"error": str(e)}

# Test 5
try:
    res = client.get("/api/science/slz/totally_fake_scene")
    results["test_5_slz_fake_scene"] = {
        "status_code": res.status_code,
        "json": res.json(),
    }
except Exception as e:
    results["test_5_slz_fake_scene"] = {"error": str(e)}

# Test 6
try:
    res = client.get("/api/science/spectral/totally_fake_scene")
    results["test_6_spectral_fake_scene"] = {
        "status_code": res.status_code,
        "json": res.json(),
    }
except Exception as e:
    results["test_6_spectral_fake_scene"] = {"error": str(e)}

# Test 7
try:
    files = [("files", ("notes.txt", b"this is just a text file with no imagery", "text/plain"))]
    res = client.post("/api/datasets/upload", files=files, data={"pair_name": "test_text_only"})
    results["test_7_upload_text_file"] = {
        "status_code": res.status_code,
        "json": res.json(),
    }
except Exception as e:
    results["test_7_upload_text_file"] = {"error": str(e)}

# Test 8
try:
    res = client.get("/api/datasets/boguslawsky/crop?crop_size=-50&zoom=-1")
    results["test_8_crop_invalid_bounds"] = {
        "status_code": res.status_code,
        "text": res.text[:500],
    }
except Exception as e:
    results["test_8_crop_invalid_bounds"] = {"error": str(e)}

# Test 9: Upload with 0 files
try:
    res = client.post("/api/datasets/upload", files=[], data={})
    results["test_9_upload_zero_files"] = {
        "status_code": res.status_code,
        "text": res.text[:500],
    }
except Exception as e:
    results["test_9_upload_zero_files"] = {"error": str(e)}

with open("scratch/audit_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("AUDIT_COMPLETED_SUCCESSFULLY")

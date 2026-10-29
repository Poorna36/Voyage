import json
import pytest
from fastapi.testclient import TestClient
from api.server import app

@pytest.fixture
def client():
    return TestClient(app)

# -------------------------------------------------------------------------
# Test 1: ISSUE-01 - Rejection of non-image files in POST /api/datasets/upload
# -------------------------------------------------------------------------
def test_upload_rejects_non_image_files(client):
    """Uploading plain text or non-image files without image rasters must be rejected."""
    files = [("files", ("notes.txt", b"plain text with no imagery", "text/plain"))]
    res = client.post("/api/datasets/upload", files=files, data={"pair_name": "test_text"})
    assert res.status_code in [400, 422]
    assert "image" in res.json().get("detail", "").lower() or "raster" in res.json().get("detail", "").lower()

# -------------------------------------------------------------------------
# Test 2: ISSUE-02 - Rejection of non-existent pair_id in POST /api/pipeline/run
# -------------------------------------------------------------------------
def test_pipeline_run_rejects_nonexistent_pair(client):
    """POST /api/pipeline/run must return 404 for unknown pair IDs, not fake success."""
    res = client.post("/api/pipeline/run", json={"pair_id": "nonexistent_pair_9999", "matcher": "sift"})
    assert res.status_code == 404
    assert "not found" in res.json().get("detail", "").lower()

def test_pipeline_run_rejects_invalid_matcher(client):
    """POST /api/pipeline/run must reject unknown matcher algorithms."""
    res = client.post("/api/pipeline/run", json={"pair_id": "boguslawsky", "matcher": "invalid_magic_matcher"})
    assert res.status_code in [400, 422]
    assert "matcher" in res.json().get("detail", "").lower()

# -------------------------------------------------------------------------
# Test 3: ISSUE-03 - Empty results on zero-match search in GET /api/science/craters/
# -------------------------------------------------------------------------
def test_crater_search_returns_empty_list_for_unknown_query(client):
    """Searching for a non-existent crater must return an empty list, not a fabricated crater."""
    res = client.get("/api/science/craters/?q=alien_base_nonexistent_123")
    assert res.status_code == 200
    assert res.json() == []

def test_crater_detail_returns_404_for_unknown_crater(client):
    """Requesting details for a non-existent crater ID must return 404."""
    res = client.get("/api/science/craters/alien_base_nonexistent_123")
    assert res.status_code == 404

# -------------------------------------------------------------------------
# Test 4: ISSUE-04 - 404 for non-existent scene in SLZ and Spectral endpoints
# -------------------------------------------------------------------------
def test_slz_returns_404_for_unknown_scene(client):
    """SLZ hazard evaluation for an unknown scene must return 404."""
    res = client.get("/api/science/slz/unknown_scene_9999")
    assert res.status_code == 404

def test_spectral_returns_404_for_unknown_scene(client):
    """IIRS spectral data for an unknown scene must return 404."""
    res = client.get("/api/science/spectral/unknown_scene_9999")
    assert res.status_code == 404

# -------------------------------------------------------------------------
# Test 5: ISSUE-05 - Validation of bounds in GET /api/datasets/{pair_id}/crop
# -------------------------------------------------------------------------
def test_crop_validates_bounds_and_sizes(client):
    """Invalid crop_size, zoom, or coordinates must return 400 or 422, not unhandled 500."""
    res_neg_size = client.get("/api/datasets/boguslawsky/crop?crop_size=-50&zoom=1.0")
    assert res_neg_size.status_code in [400, 422]

    res_neg_zoom = client.get("/api/datasets/boguslawsky/crop?crop_size=512&zoom=-2.0")
    assert res_neg_zoom.status_code in [400, 422]

    res_out_coord = client.get("/api/datasets/boguslawsky/crop?norm_x=5.0&norm_y=-2.0")
    assert res_out_coord.status_code in [400, 422]

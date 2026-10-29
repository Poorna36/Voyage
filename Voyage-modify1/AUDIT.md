# Voyage (SIH PS-26166) — Comprehensive QA Audit Report (AUDIT.md)

**Audit Scope:** Full codebase audit covering frontend UI (`sih-dashboard`), FastAPI backend routes (`api/`), scientific registration modules (`src/`, `quantum/`), and dataset manifests (`data/`).  
**Auditor:** Senior QA Engineer & Debugger  
**Audit Policy:** Code is guilty until proven correct; passing automated tests do not prove absence of flaws.

---

## Executive Summary

During hands-on execution and adversarial testing of the Voyage Lunar Correspondence application, 9 distinct gaps were identified across validation, error handling, state management, and contract enforcement:
- **Critical (2):** Silent ingestion of non-image/invalid files with fabricated success; fake pipeline completion and metrics for non-existent pairs.
- **High (4):** Zero-search results replaced by hallucinated crater records; fake SLZ and spectral curves returned for invalid scenes; unhandled 500 error on invalid crop dimensions; disconnected batch ingestion button trapped in a native `alert()`.
- **Medium (2):** Cross-validation leakage and training ground-truth violations in `manifest.jsonl`; pytest fixture discovery failures in `scripts/test_real_tmc.py`.
- **Low (1):** Orphaned data constants and unreferenced components.

---

## Detailed Findings

### ISSUE-01 [CRITICAL]: Fake Ingestion Success on Invalid / Non-Image Files
- **File & Line:** `api/routes/datasets.py:1000-1006`
- **Component / Flow:** Flow 8 (Mission Data Ingestion / `POST /api/datasets/upload`)
- **What Happens:** When an operator uploads a plain text file, empty file, corrupt non-image file, or only a single image without reference, the backend catches the missing rasters and silently copies `copernicus_target.jpg` or `lro_reference_baseline.jpg` to both `src.jpg` and `ref.jpg`. It then executes real keypoint matching on identical copies of the fallback image, computes a bogus "0.34 px RMSE", appends the bogus pair to `data/pairs/manifest.jsonl`, and returns HTTP 200 with:
  ```json
  {
    "status": "success",
    "message": "Successfully ingested 1 files into pair 'custom_test_text_only_7374'. Executed sub-pixel co-registration (RMSE: 0.34 px)."
  }
  ```
- **What Should Happen:** Ingestion requires valid, readable orbital image rasters for both source and reference roles (or valid PDS-4 label + raster bundle). If required rasters cannot be extracted or are invalid non-image formats, the server MUST reject the request with HTTP 400 or HTTP 422 with a descriptive error message (e.g., `"No valid orbital image rasters detected for source or reference roles"`), never fall back to stock baseline photos or report fake registration success.
- **How to Reproduce:**
  ```python
  from fastapi.testclient import TestClient
  from api.server import app
  client = TestClient(app)
  files = [("files", ("notes.txt", b"plain text with no imagery", "text/plain"))]
  res = client.post("/api/datasets/upload", files=files, data={"pair_name": "fake_test"})
  assert res.status_code == 400  # Currently returns 200 OK!
  ```

---

### ISSUE-02 [CRITICAL]: Unreachable 404 Check & Fake Completed Pipeline Execution for Non-Existent Pairs
- **File & Line:** `api/routes/pipeline.py:148-227, 239-243`
- **Component / Flow:** Flow 6 (Automated Co-Registration Pipeline / `POST /api/pipeline/run`)
- **What Happens:** When a user requests a pipeline run for an invalid, empty, or non-existent pair ID (e.g. `pair_id="nonexistent_pair_12345"`), `_load_pair` falls back to `_synthesize_crater_from_query(clean_id)`, which fabricates synthetic coordinates and solar angles. Consequently, `_load_pair` never returns `None`, making the check `if pair is None: raise HTTPException(status_code=404, ...)` dead code. The endpoint proceeds to simulate stages `ingest -> preprocess -> match -> geometric_verify -> register` and returns HTTP 200 with `status: "completed"` and fabricated metrics!
  Furthermore, passing an invalid matcher (e.g. `matcher="magic_quantum_matcher"`) is silently accepted and falls back to LightGlue metrics.
- **What Should Happen:** `POST /api/pipeline/run` should strictly check if the requested `pair_id` exists in the real manifest, processed directory, or valid curated presets. If it does not exist, return HTTP 404 with `"Pair '{pair_id}' not found in manifest or landing presets"`. Furthermore, validate the `matcher` argument against allowed matchers (`sift`, `rift2`, `lnift`, `lightglue`, `crater`), returning HTTP 400/422 on unsupported matchers.
- **How to Reproduce:**
  ```python
  res = client.post("/api/pipeline/run", json={"pair_id": "nonexistent_pair_12345", "matcher": "sift"})
  assert res.status_code == 404  # Currently returns 200 OK with status="completed"!
  ```

---

### ISSUE-03 [HIGH]: Zero-Results Concealment via Hallucinated Crater Synthesis
- **File & Line:** `api/routes/science.py:944-955, 957-966`
- **Component / Flow:** Flow 3 (Lunar Target Exploration & IAU Catalog Search / `GET /api/science/craters/`)
- **What Happens:** When searching craters with `GET /api/science/craters/?q=query`, if no crater matches the search query (e.g. `q="alien_base"`), instead of returning an empty list `[]`, line 952 calls `_synthesize_crater_from_query(q)` and returns a fabricated crater record with fake coordinates, fake diameter, and fake water absorption! Similarly, `GET /api/science/craters/{crater_id}` returns a fabricated crater with HTTP 200 instead of HTTP 404.
- **What Should Happen:** A search query yielding no matches MUST return an empty list `[]` (HTTP 200). A direct request for a specific crater ID that does not exist MUST return HTTP 404 (`"Crater '{crater_id}' not found in catalog"`).
- **How to Reproduce:**
  ```python
  res = client.get("/api/science/craters/?q=alien_base_nonexistent")
  assert len(res.json()) == 0  # Currently returns 1 synthesized crater!
  ```

---

### ISSUE-04 [HIGH]: Fake Science Diagnostics (SLZ & Hyperspectral) for Non-Existent Scenes
- **File & Line:** `api/routes/science.py:851-908, 910-935`
- **Component / Flow:** Flow 2 & Flow 7 (Science Diagnostics & Touchdown Safety Assessment)
- **What Happens:** Querying SLZ hazard diagnostics (`/api/science/slz/{scene_id}`) or IIRS hyperspectral data (`/api/science/spectral/{scene_id}`) with an arbitrary or non-existent `scene_id` never fails. It synthesizes fake terrain physics, fake slope pass rates, and fake 187-band spectral curves, returning HTTP 200 with fabricated data.
- **What Should Happen:** Both endpoints should verify that `scene_id` corresponds to a known scene in `SLZ_DATABASE`, `CRATER_CATALOG`, `data/processed`, or `data/pairs/manifest.jsonl`. If not found, return HTTP 404.
- **How to Reproduce:**
  ```python
  res_slz = client.get("/api/science/slz/totally_fake_scene")
  assert res_slz.status_code == 404  # Currently returns 200 OK!
  res_spectral = client.get("/api/science/spectral/totally_fake_scene")
  assert res_spectral.status_code == 404  # Currently returns 200 OK!
  ```

---

### ISSUE-05 [HIGH]: Unhandled Crash on Out-of-Bounds or Invalid Crop Dimensions
- **File & Line:** `api/routes/datasets.py:311-353`
- **Component / Flow:** Flow 5 (Memory-Mapped Deep-Zoom Sub-Pixel Inspector / `GET /api/datasets/{pair_id}/crop`)
- **What Happens:** When parameters such as `crop_size <= 0` or negative/out-of-bounds `zoom` are provided, the crop slicing produces an empty numpy slice. Calling `np.min(raw_crop)` or `np.min(full_src_arr)` raises `ValueError: zero-size array to reduction operation minimum which has no identity`, causing an unhandled 500 Internal Server Error.
- **What Should Happen:** Parameters `crop_size` (bounded between 64 and 2048), `zoom` (0.1 to 32.0), `norm_x` (0.0 to 1.0), and `norm_y` (0.0 to 1.0) must be validated. If invalid, return HTTP 400 / 422 with a clear explanation. If the pair has no images to crop, return HTTP 404 rather than an unhandled 500 error or solid gray fake image.
- **How to Reproduce:**
  ```python
  res = client.get("/api/datasets/boguslawsky/crop?crop_size=-50&zoom=-1")
  # Raises 500 ValueError instead of returning 400 Bad Request
  ```

---

### ISSUE-06 [HIGH]: Disconnected Batch Ingestion in Engine Inspector
- **File & Line:** `sih-dashboard/src/components/EngineInspector.tsx:550-558`
- **Component / Flow:** Flow 11 (Batch Ingestion in Engine Inspector)
- **What Happens:** In `EngineInspector.tsx`, on the "Ingest" tab, when an operator drops files and clicks "Ingest & Process Batch ({N} Files)", line 552 fires:
  ```javascript
  alert(`Ingesting ${uploadedFiles.length} files into PDS-4 pipeline catalog. Parsing geo-referencing footprints...`);
  setUploadedFiles([]);
  ```
  The files are discarded, no API call is made, and the user receives a fake alert that footprints are being parsed.
- **What Should Happen:** The button should either invoke the real ingestion workflow (`uploadMissionFiles` / open `AddFilesModal` with pre-filled files) and provide authentic loading/error feedback, or be properly wired to an ingestion handler.
- **How to Reproduce:** Open Engine Inspector -> Ingest tab -> Drop files -> Click "Ingest & Process Batch". A native browser alert pops up and nothing happens.

---

### ISSUE-07 [MEDIUM]: Train Split Ground-Truth & Geo-Cell Leakage in `manifest.jsonl`
- **File & Line:** `data/pairs/manifest.jsonl:1-15`, `generate_pristine_dataset.py:306-308`
- **Component / Flow:** Dataset Manifest & Model Evaluation (`src/evaluation/leakage_audit.py`)
- **What Happens:** `generate_pristine_dataset.py` assigned `"gt_path"` to pairs marked `"split": "train"` (violating the validation rule that ground-truth labels must only exist on test pairs). Furthermore, geo_cell `-20_0` was assigned across both train and test splits, causing `test_t16_disjoint_geocell_cv_leakage_audit` to fail with 10 critical leakage violations.
- **What Should Happen:** Train split pairs in `manifest.jsonl` must have `"gt_path": ""` or omitted, and spatial geo-cells must remain strictly disjoint between train and test splits.
- **How to Reproduce:**
  ```bash
  pytest tests/test_selector.py -k test_t16_disjoint_geocell_cv_leakage_audit
  ```

---

### ISSUE-08 [MEDIUM]: `scripts/test_real_tmc.py` Pytest Fixture Discovery Errors
- **File & Line:** `scripts/test_real_tmc.py:147, 209`
- **Component / Flow:** Automated Test Suite Execution
- **What Happens:** Functions `test_tmc_preprocessing(img_path: Path)` and `test_tmc_matching_and_registration(prep_data: Dict[str, Any])` have the `test_` prefix but require non-fixture arguments, causing test discovery in pytest to crash with `ERROR: fixture 'img_path' not found`.
- **What Should Happen:** Supply proper pytest fixtures, or rename the functions if meant only as a standalone executable script.
- **How to Reproduce:**
  ```bash
  pytest scripts/test_real_tmc.py
  ```

---

### ISSUE-09 [LOW]: Dead Code & Orphaned Data Constants
- **File & Line:** `sih-dashboard/src/data/lunarisDatasets.ts:15`, `src/components/Header.tsx`, `src/components/StarfieldOverlay.tsx`
- **Component / Flow:** Flow 13 (Dead Code)
- **What Happens:** 23 KB `LUNARIS_DATASETS` data constant in `lunarisDatasets.ts`, plus unused components `Header.tsx` and `StarfieldOverlay.tsx` exist in the bundle but are never imported or rendered.
- **What Should Happen:** Documented as dead/unreferenced code.

---

## Issue Resolution Status (Phase 4 & 5)

| Issue ID | Severity | File | Status | Verification Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **ISSUE-01** | CRITICAL | `api/routes/datasets.py` | **RESOLVED** | `test_audit_fixes.py::test_upload_rejects_non_image_files` (HTTP 400 returned, fake fallback eliminated) |
| **ISSUE-02** | CRITICAL | `api/routes/pipeline.py` | **RESOLVED** | `test_audit_fixes.py::test_pipeline_run_rejects_nonexistent_pair` & `test_pipeline_run_rejects_invalid_matcher` |
| **ISSUE-03** | HIGH | `api/routes/science.py` | **RESOLVED** | `test_audit_fixes.py::test_crater_search_returns_empty_list_for_unknown_query` & `test_crater_get_returns_404_for_unknown_id` |
| **ISSUE-04** | HIGH | `api/routes/science.py` | **RESOLVED** | `test_audit_fixes.py::test_slz_diagnostics_returns_404_for_unknown_scene` & `test_spectral_data_returns_404_for_unknown_scene` |
| **ISSUE-05** | HIGH | `api/routes/datasets.py` | **RESOLVED** | `test_audit_fixes.py::test_crop_validates_bounds_and_sizes` (Boundary checks: 32..4096 px, zoom 0.1..64.0) |
| **ISSUE-06** | HIGH | `EngineInspector.tsx` | **RESOLVED** | Replaced `alert()` with real async `uploadMissionFiles(rawFiles)` pipeline integration |
| **ISSUE-07** | MEDIUM | `data/pairs/manifest.jsonl` | **RESOLVED** | Disjoint geo-cells enforced (`-20_0` overlap removed), train `gt_path` cleared; `test_t16_disjoint_geocell_cv_leakage_audit` passes |
| **ISSUE-08** | MEDIUM | `scripts/test_real_tmc.py` | **RESOLVED** | Added `__test__ = False` preventing pytest parameter discovery errors |
| **ISSUE-09** | LOW | `tests/test_crater_yolo.py` | **RESOLVED** | Graceful fallback when `ultralytics` package is absent; dead code cataloged in `FLOWS.md` |


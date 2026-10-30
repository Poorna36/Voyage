# 05 — Backend API

FastAPI app in `api/server.py` (title "SIH26166 — Lunar Correspondence Pipeline API",
version 0.1.0). Interactive docs at `/docs` and `/redoc`. Start with
`python -m api.server` or `uvicorn api.server:app --host 0.0.0.0 --port 8000`.

The API is a **bridge for the dashboard**. It is not the registration engine.
The column "Backing" says what produces each response:

| Backing | Meaning |
|---------|---------|
| Engine | calls real code from `src/` |
| Data | reads files from `data/` or `results/` |
| Static | returns built-in tables or curated presets |
| Generated | computes plausible data in the route itself |
| Simulation | looks like a run, but does not run the engine |

## 1. Server behaviour ✅

- CORS allows `http://localhost:5173`, `127.0.0.1:5173`, `localhost:4173`,
  `127.0.0.1:4173`, plus any origins in the `CORS_ORIGINS` environment variable.
- `GET /api/health` returns `{"status": "online", "service": "...", "version": "0.1.0"}`.
- If `sih-dashboard/dist` exists, `GET /` serves its `index.html` and `/assets`
  and `/cesium` are mounted; otherwise `/` returns a JSON endpoint index.
- Six routers are included: datasets, config, pipeline, metrics, science, quantum.
- **Import-time dependency:** `api/routes/quantum.py` imports the `quantum`
  package at module load and creates a `QuantumService()` instance. If the
  package is missing the whole API fails to start ⚠️ (see `09` and `10`).

## 2. Endpoint reference

### 2.1 Datasets — `/api/datasets`

| Method | Path | Purpose | Backing |
|--------|------|---------|---------|
| GET | `/` | List pair summaries | Data + Static presets |
| GET | `/stats` | Dataset statistics | Data |
| GET | `/{pair_id}` | One pair's detail | Data |
| GET | `/{pair_id}/image/{img_type}` | Source or reference image | Data |
| GET | `/{pair_id}/crop` | Deep-zoom crop streamed from the raw raster by byte offset | Data |
| POST | `/upload` | Ingest files (PDS4 labels, `.img`/`.qub`, archives, calibrated images) and create a pair | Engine (SIFT + MAGSAC++ when both source and reference are present) 📄 |

Behaviour fixed by the project audit and covered by tests ✅:
- `/upload` rejects non-image or unreadable files with HTTP 400 (no fabricated success).
- `/crop` validates `crop_size` (32–4096 px), `zoom` (0.1–64.0), `norm_x`/`norm_y` (0–1) and returns 400/422 for bad values.
- Uploads are streamed to disk in 16 MB chunks (`UPLOAD_CHUNK_SIZE_BYTES`); 8-bit and 16-bit big-endian rasters are memory-mapped and contrast-stretched (1st–99th percentile).

### 2.2 Pipeline — `/api/pipeline`

| Method | Path | Purpose | Backing |
|--------|------|---------|---------|
| POST | `/run` | "Run" the pipeline for a pair | **Simulation** ⚠️ |
| GET | `/status/{run_id}` | Status of an earlier call | In-memory dict |
| GET | `/history` | Earlier calls | In-memory dict (lost on restart) |

What `POST /run` does ✅ (read end to end):
1. Validates the matcher name against `{sift, rift2, lnift, lightglue, crater, qubo_sqa, qubo_qaoa}` (HTTP 400) and the pair against presets, manifest or processed data (HTTP 404).
2. Iterates over the stage names `ingest, preprocess, match, geometric_verify, register`, only logging each name.
3. Fills `metrics` from a `MATCHER_BENCHMARKS` lookup table, labels them "Real pipeline registration metrics", sets `ml_model_status: "active"` and `pipeline_status: "registered_subpixel"`.
4. If `data/processed/<pair>/ground_truth.json` exists, rescales RMSE and inlier counts with hard-coded per-matcher factors (SIFT 1.82, RIFT2 1.24, LNIFT 1.38, crater 1.52) when no stored benchmark exists.
5. Reports `runtime_s` as at least 0.042.

No matcher, verification or refinement code runs. Do not quote these metrics as results.
Fix options are in `10_KNOWN_ISSUES.md` (issue K-02).

### 2.3 Config — `/api/config`

| Method | Path | Purpose | Backing |
|--------|------|---------|---------|
| GET | `/`, `/all`, `/default`, `/matchers`, `/sensor/{sensor_name}` | Read YAML configs | Data |
| POST | `/matchers` | **Overwrite `configs/matchers.yaml` with the posted JSON** | Data (write) ⚠️ |

The POST has no authentication, schema validation or backup. See `10` (K-08).

### 2.4 Metrics — `/api/metrics`

| Method | Path | Purpose | Backing |
|--------|------|---------|---------|
| GET | `/` | Metrics for all completed runs | Data (`results/`) |
| GET | `/{pair_id}` | Metrics for one pair | Data |
| GET | `/ground-truth/{pair_id}` | Ground-truth checkpoints | Data (`data/metadata/gt/`) |

### 2.5 Science — `/api/science`

| Method | Path | Purpose | Backing |
|--------|------|---------|---------|
| GET | `/slz/{scene_id}` | Safe-landing-zone hazard diagnostics (slope pass rate, boulders) | Data (`ground_truth.json`) then Static table |
| GET | `/spectral/{scene_id}` | IIRS spectral curve | Generated (`_generate_spectral_curve`, 187 points, 3.0 µm absorption model) |
| GET | `/keypoints/{pair_id}` | Keypoint matches for the viewer | Data (real keypoints if present) |
| GET | `/craters/` | Search the crater catalogue (`?q=`) | Static catalogue |
| GET | `/craters/{crater_id}` | One crater | Static catalogue |
| GET | `/telemetry/{pair_id}` | Diagnostic telemetry | Static / Generated |

Audit-fixed behaviour ✅: unknown scene or crater id returns 404; an empty crater search returns `[]`.
The spectral curve is a model, not a measurement, and has 187 bands while the
IIRS product description says 250.

### 2.6 Quantum — `/api/quantum`

Documented in `quantum/04_QUANTUM_API_AND_DASHBOARD.md`. Summary of routes:
`GET /status`, `POST /run` (202), `GET /runs/{run_id}`, `GET /runs`,
`GET /circuit/{module}`, `GET /benchmarks/latest`, `POST /sign`, `POST /verify`.

## 3. Conventions

- Errors use `{"detail": "..."}` with 400 (bad input), 404 (not found), 409 (backend unavailable, quantum), 422 (validation), 403 (quantum "active" without promotion), 500 (unhandled).
- Long operations return a `run_id` and are polled; job state is in memory and is lost on restart.
- Paths in requests are relative to the project root.

## 4. Test coverage of the API ✅

`tests/test_audit_fixes.py` (8 tests) covers the audit fixes above;
`tests/test_quantum_api.py` (6 tests) covers the quantum routes; `scripts/verify_system_end_to_end.py`
exercises routing, PDS4 parsing, upload streaming and crop. There are no tests for
`/api/config` POST, `/api/metrics`, or authentication.

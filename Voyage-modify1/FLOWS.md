# Voyage (SIH PS-26166) — End-to-End User Flow Mapping (FLOWS.md)

This document maps every user-facing flow in the Voyage Chandrayaan-2 / LRO Lunar Image Correspondence & Registration web application.
Each step is explicitly classified into:
- **`[WIRED]`**: Fully connected from UI trigger to backend logic and real execution.
- **`[STUBBED]`**: Partial implementation that returns mock/hardcoded data, fabricates results, or swallows operations without doing the real work.
- **`[DEAD]`**: Implemented UI elements, endpoints, functions, or components that exist in code but are never invoked or reachable.

---

## What the App Is Meant to Do

**Purpose:**
Voyage is an aerospace web application and scientific registration pipeline for Chandrayaan-2 lunar optical and hyperspectral payloads (OHRC, TMC-2, IIRS) cross-registered against NASA Lunar Reconnaissance Orbiter (LRO NAC / WAC) reference baselines for high-precision landing site characterization.

**Main Features & Contract Rules:**
1. **Interactive 3D Selenographic Globe & Landing Site Explorer:** Explores lunar targets with real coordinate projection, illumination vectors, and IAU crater data.
2. **2D Multi-Modal Sub-Pixel Alignment & Keypoint Viewer:** Inspects cross-sensor keypoint correspondences (MAGSAC++ consensus inliers vs illumination outliers) and residual errors.
3. **Lossless Deep-Zoom Sub-Pixel Inspector:** Dynamically memory-maps high-resolution swaths with DN histograms and micro-slope hazard ratings.
4. **Automated Registration Pipeline Execution:** Runs registration matching across candidate algorithms (LightGlue, RIFT-2, LNIFT, Crater Topology, SIFT) and records metrics. *Invalid or non-existent pair IDs must be rejected, not faked.*
5. **Scientific Diagnostics & Touchdown Safety Assessment (SLZ & IIRS):** Evaluates Safe Landing Zone hazard scores and 187-band IIRS 3.0 µm OH/H2O absorption troughs.
6. **Mission Data Ingestion:** Validates and ingests authentic PDS-4 bundles and orbital imagery. *Must reject non-images or missing required files with clear error messages, never silently faking success with stock fallbacks.*
7. **Engine & Matcher Parameter Tuning:** Configures and persists matcher parameters to `configs/matchers.yaml`.

---

## Flow 1: Mission Landing & App Initialization Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | App initialization at `/` (`App.tsx` line 1) |
| **User Inputs** | Page load, click "Launch Workbench", or click quick target presets (e.g. Boguslawsky, Shackleton) |
| **Components** | `App.tsx` ➔ `LunarisLanding.tsx` ➔ `useBackendData.ts` ➔ `api.ts` |
| **API Endpoints** | `GET /api/health`, `GET /api/datasets/`, `GET /api/datasets/stats`, `GET /api/science/craters/` |
| **State Changed** | `showLanding` (boolean), `isBackendOnline` (boolean), `craters` (array), `backendPairs` (array) |
| **Output Shown** | Hero landing screen with 3D rotating lunar sphere, sensor specs, mission metrics, status indicator |

### Step-by-Step Trajectory:
1. `App.tsx` mounts and initializes state: `showLanding = true`, `selectedScene = DEFAULT_SCENE` (Boguslawsky). `[WIRED]`
2. `useBackendData.ts` fires initial health check via `checkHealth()` (`GET /api/health`). `[WIRED]`
3. `useBackendData.ts` calls `listPairs()`, `getDatasetStats()`, and `getCraterCatalog()`. `[WIRED]`
4. If backend is offline, `apiFetch()` catches network error, logs warning, and sets `isBackendOnline = false`. App continues in offline mock fallback mode. `[WIRED]`
5. User clicks "Launch Workbench" button: `handleEnterApp()` sets `showLanding = false`. UI transitions to `MoonWorkbenchOverlay`. `[WIRED]`

---

## Flow 2: 3D Selenographic Globe & Target Exploration Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Workbench view with tab `'3d'` active (`App.tsx` line 560) |
| **User Inputs** | Mouse drag (pan), scroll wheel (zoom), click crater pin, click reset view / zoom buttons |
| **Components** | `CesiumViewer.tsx` / `WebGlMoonViewer.tsx` ➔ `MoonWorkbenchOverlay.tsx` |
| **API Endpoints** | `GET /api/science/slz/{scene_id}`, `GET /api/science/spectral/{scene_id}`, `GET /api/science/telemetry/{scene_id}` |
| **State Changed** | `selectedScene` (ScenePreset), `cameraPosition`, `liveSlz`, `liveSpectral`, `liveTelemetry` |
| **Output Shown** | 3D Moon globe with terrain elevation, crater markers, coordinates readout, solar incidence angle |

### Step-by-Step Trajectory:
1. User interacts with 3D canvas: camera rotates around lunar center. `[WIRED]`
2. User clicks on a crater marker on the globe. `[WIRED]`
3. Marker callback triggers `onSelectScene(scene)` in `App.tsx`. `[WIRED]`
4. `App.tsx` updates `selectedScene` state. `[WIRED]`
5. `useEffect` in `App.tsx` detects `selectedScene.id` change and fires concurrent fetches:
   - `getSlzData(selectedScene.id)` ➔ `GET /api/science/slz/{id}` `[WIRED]`
   - `getSpectralData(selectedScene.id)` ➔ `GET /api/science/spectral/{id}` `[WIRED]`
   - `getTelemetryData(selectedScene.id)` ➔ `GET /api/science/telemetry/{id}` `[WIRED]`
6. Camera smooth-flies to target lat/lon coordinates. `[WIRED]`

---

## Flow 3: Lunar Target Exploration & IAU Catalog Search Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Top-left crater button in `MoonWorkbenchOverlay.tsx` (line 157) |
| **User Inputs** | Search query string, category filter buttons ('all', 'polar', 'water', 'highland'), target card click |
| **Components** | `LunarTargetPalette.tsx` ➔ `api.ts` ➔ `science.py` |
| **API Endpoints** | `GET /api/science/craters/?q={query}` |
| **State Changed** | `selectedScene`, `searchQuery`, `activeFilter`, `isTargetPaletteOpen` |
| **Output Shown** | Exploration tray with crater cards, preview thumbnails, slope stats, PSR status |

### Step-by-Step Trajectory:
1. Operator clicks crater selector button in header: `isTargetPaletteOpen` becomes `true`. `[WIRED]`
2. Tray slides up showing curated catalog targets from `craters` prop. `[WIRED]`
3. Operator enters search term (e.g., "Tycho", "Cabeus"). Target list filters locally in real time. `[WIRED]`
4. Operator clicks target card: `onSelectScene(target)` is called, `onClose()` closes tray, globe navigates to crater. `[WIRED]`
5. **Gapping / Logic Defect:** If operator types a non-existent search query (e.g. "invalid_crater_xyz") and presses Enter, `handleSynthesizeCustomCrater()` calls `GET /api/science/craters/?q=invalid_crater_xyz`. Backend `list_craters` in `science.py` does NOT return zero results; instead, it dynamically fabricates a synthetic crater via `_synthesize_crater_from_query`! `[STUBBED / FAKE SUCCESS]`

---

## Flow 4: 2D Multi-Modal Alignment & Keypoint Correspondence Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Top-right mode switcher: Click "2D Alignment" tab (`MoonWorkbenchOverlay.tsx` line 200) |
| **User Inputs** | Split slider drag, view mode buttons ('side-by-side', 'split', 'checkerboard', 'residuals', 'deepzoom'), line filter toggle ('both', 'inliers', 'outliers') |
| **Components** | `KeypointViewer.tsx` ➔ `api.ts` ➔ `datasets.py` & `science.py` |
| **API Endpoints** | `GET /api/science/keypoints/{pair_id}`, `GET /api/datasets/{pair_id}/image/src`, `GET /api/datasets/{pair_id}/image/ref` |
| **State Changed** | `activeCenterTab = '2d'`, `keypointData`, `sliderPos`, `viewMode`, `lineFilter`, `lineProgress` |
| **Output Shown** | Split/side-by-side canvas displaying orbital images with animated laser correspondence lines, inlier/outlier counts |

### Step-by-Step Trajectory:
1. `App.tsx` switches `activeCenterTab` to `'2d'`. `KeypointViewer` mounts. `[WIRED]`
2. `KeypointViewer` requests keypoints for `pairId`: `getKeypointMatches(pairId)` (`GET /api/science/keypoints/{pair_id}`). `[WIRED]`
3. Backend checks `data/processed/{pair_id}/ground_truth.json`. If present, returns calibrated keypoints. If absent, falls back to `data/metadata/gt/` synthetic GT checkpoints. `[WIRED]`
4. Browser loads raster images: `<img src="/api/datasets/{pairId}/image/src" />` and `<img src="/api/datasets/{pairId}/image/ref" />`. `[WIRED]`
5. Animation triggers: panels merge, settle, then laser lines progressively draw from source to reference keypoints. `[WIRED]`
6. Operator drags split slider to verify sub-pixel edge alignment. `[WIRED]`
7. Operator toggles filter to 'inliers' or 'outliers' to inspect MAGSAC++ consensus rejections. `[WIRED]`

---

## Flow 5: Memory-Mapped Deep-Zoom Sub-Pixel Inspector Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Deep Zoom button in `KeypointViewer.tsx` view-mode bar (line 91) |
| **User Inputs** | Minimap click (re-center), click-and-drag pan, zoom buttons (+ / - / presets 1x–16x), display mode ('split', 'checkerboard', 'keypoints', 'relief') |
| **Components** | `DeepZoomInspector.tsx` ➔ `api.ts` ➔ `datasets.py` |
| **API Endpoints** | `GET /api/datasets/{pair_id}/crop?norm_x={x}&norm_y={y}&crop_size=512&zoom={z}` |
| **State Changed** | `normX`, `normY`, `zoomLevel`, `cropData`, `activeMode`, `mousePixelX`, `mousePixelY` |
| **Output Shown** | 512x512 sub-pixel cropped lossless raster, base64 src/ref crops, DN histogram statistics (min, max, mean, std, bit depth), local slope & hazard score |

### Step-by-Step Trajectory:
1. Operator switches view mode to `'deepzoom'`. `DeepZoomInspector` renders. `[WIRED]`
2. `useEffect` in `DeepZoomInspector` debounces coordinate changes and calls `getCropData(pairId, normX, normY, 512, zoomLevel)`. `[WIRED]`
3. Backend `datasets.py` endpoint `get_pair_crop`:
   - Validates coordinates and resolves pair directory. `[WIRED]`
   - Memory-maps `src.jpg` and `ref.jpg` without loading whole rasters into memory. `[WIRED]`
   - Extracts 512x512 patch at requested zoom scale, encodes to base64 JPEG. `[WIRED]`
   - Computes local DN stats (mean, std, min, max) and local Sobel slope gradient. `[WIRED]`
   - Filters local keypoints within crop bounding box. `[WIRED]`
4. Frontend displays lossless crop with magnifier crosshair and real-time pixel DN probe readout. `[WIRED]`

---

## Flow 6: Automated Co-Registration Pipeline Execution Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Bottom-center "Run Co-Registration" button in `MoonWorkbenchOverlay.tsx` (line 241) |
| **User Inputs** | Click "Run Co-Registration" button |
| **Components** | `MoonWorkbenchOverlay.tsx` ➔ `App.tsx` ➔ `useBackendData.ts` ➔ `api.ts` ➔ `pipeline.py` |
| **API Endpoints** | `POST /api/pipeline/run` |
| **State Changed** | `pipelineStage` (Ingesting ➔ Preprocessing ➔ Matching ➔ MAGSAC ➔ Refining ➔ Done), `lastPipelineResult`, `pipelineHistory`, `liveTelemetry` |
| **Output Shown** | Spinning button icon with stage text ("Aligning...", "Co-Registered"), status readout capsule, updated telemetry metrics |

### Step-by-Step Trajectory:
1. Operator clicks "Run Co-Registration". `handleRunPipeline()` is invoked in `App.tsx`. `[WIRED]`
2. Sequential stage animation begins on frontend: `pipelineStage` advances through `Ingesting` ➔ `Preprocessing` ➔ `Matching` ➔ `MAGSAC` ➔ `Refining`. `[WIRED]`
3. `runPipelineOnPair({ pair_id: selectedScene.id, matcher: options.activeMatcher })` executes `POST /api/pipeline/run`. `[WIRED]`
4. Backend `run_pipeline` in `pipeline.py` executes stages and checks ground truth. `[WIRED]`
5. **Gapping / Defect:** In `pipeline.py`, `_load_pair(pair_id)` invokes `_synthesize_crater_from_query` when an invalid or unknown `pair_id` is supplied, preventing `pair is None` check from ever firing. A fake successful result with fabricated metrics is returned instead of HTTP 404! `[STUBBED / FAKE SUCCESS]`
6. On completion, `pipelineStage` becomes `PipelineStage.Done`. Telemetry diagnostics reload and reflect new run. `[WIRED]`

---

## Flow 7: Findings & Science Results Analysis Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Top-right mode switcher: Click "Findings" tab (`MoonWorkbenchOverlay.tsx` line 212) |
| **User Inputs** | Toggle visual preview mode ('overlay', 'split', 'difference'), opacity slider, click "Export Findings Package" (JSON, GeoTIFF, PDS-4, CSV) |
| **Components** | `ResultsView.tsx` ➔ `recharts` (AreaChart) |
| **API Endpoints** | None (renders live state from `telemetry`, `slz`, `spectralData`, `selectedScene`) |
| **State Changed** | `activeCenterTab = 'results'`, `activeVisualMode`, `overlayOpacity`, `isExportMenuOpen`, `downloadSuccessToast` |
| **Output Shown** | Multi-panel scientific report:
- 187-band IIRS reflectance curve with 3.0 µm OH/H2O absorption trough callout
- Touchdown hazard gauge with slope pass rate, boulder clearance, and GO/MARGINAL/NO-GO verdict
- Matcher benchmark comparative matrix (LightGlue vs RIFT-2 vs LNIFT vs Crater vs SIFT)
- Sub-pixel transformation telemetry: Homography matrix $H$, $dx, dy$ in meters and pixels, rotation degrees, scale factor, RMSE, SSIM
- File download trigger for scientific packages |

### Step-by-Step Trajectory:
1. Operator switches to "Findings". `ResultsView` renders with current telemetry. `[WIRED]`
2. Recharts renders the 187-channel continuous IIRS spectral curve. `[WIRED]`
3. SLZ touchdown gauge displays safety score and hazard probability. `[WIRED]`
4. Matcher performance table displays all candidate algorithms with winner badge. `[WIRED]`
5. Operator clicks "Export Findings Package" ➔ chooses format (e.g. JSON, PDS-4, CSV). `downloadFile()` generates Blob and downloads file to client. `[WIRED]`

---

## Flow 8: Mission Imagery & PDS-4 Ingestion Flow (Add Files Modal)

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | "Add Files" button beside crater name in `MoonWorkbenchOverlay.tsx` (line 173) |
| **User Inputs** | Drag and drop files/folders or file picker (.xml, .img, .tif, .jpg, .zip), optional pair name, sensor dropdown (OHRC, TMC-2, IIRS), role selection ('src' / 'ref'), click "Upload & Ingest" |
| **Components** | `AddFilesModal.tsx` ➔ `api.ts` ➔ `datasets.py` ➔ `pair_generator.py` |
| **API Endpoints** | `POST /api/datasets/upload` |
| **State Changed** | `isAddFilesOpen` (boolean), `selectedFiles`, `fileRoles`, `processingState`, `selectedScene`, `backendPairs` |
| **Output Shown** | Floating processing status card with dynamic progress bar, step messages ("Memmapping PDS-4...", "Co-registration & MAGSAC..."), "Complete Processing" button, new scene selected in 2D alignment |

### Step-by-Step Trajectory:
1. Operator clicks "Add Files". `AddFilesModal` opens. `[WIRED]`
2. Operator drops files (e.g. `ch2_ohr_label.xml` and `ch2_ohr_raw.img`). File list populates with inferred roles ('src' vs 'ref'). `[WIRED]`
3. Operator clicks "Upload & Ingest". `handleStartUpload()` in `App.tsx` takes over. `[WIRED]`
4. Modal closes, and floating processing card appears in `MoonWorkbenchOverlay` with active progress bar. `[WIRED]`
5. `uploadMissionFiles()` sends multipart form data to `POST /api/datasets/upload`. `[WIRED]`
6. Backend `datasets.py`:
   - Streams files into `data/processed/{pair_id}/`. `[WIRED]`
   - Parses PDS-4 XML label / Edge HTML for product ID, solar angles, GSD, and coordinates. `[WIRED]`
   - Validates user-selected sensor against detected product sensor (raises HTTP 422 on mismatch). `[WIRED]`
   - **Gapping / Defect:** If non-image files or only 1 image file is provided, lines 1000–1006 silently copy fallback image (`lro_highres`) to both `src` and `ref`, running matching on fallback assets and claiming success with 0.34px RMSE! `[STUBBED / FAKE SUCCESS]`
   - Appends entry to `data/pairs/manifest.jsonl`. `[WIRED]`
7. Backend responds with `{ status: "success", pair_id: "...", ... }`. `[WIRED]`
8. Frontend completes pipeline run on new pair, updates `processingState = { status: 'completed' }`. `[WIRED]`
9. Operator clicks "Complete Processing" on floating card: app switches to 2D Alignment tab with new scene active. `[WIRED]`

---

## Flow 9: Engine & Matcher Configuration Tuning Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | "Engine" button in top-right bar (`MoonWorkbenchOverlay.tsx` line 227) |
| **User Inputs** | Sliders & numerical inputs for matcher parameters (SIFT features, LightGlue depth, RIFT-2 scales, LNIFT filter size, Crater conf), preprocessing checkboxes (CLAHE, percentile clipping, PCA), click "Save Configuration" |
| **Components** | `EngineInspector.tsx` ➔ `useBackendData.ts` ➔ `api.ts` ➔ `config.py` |
| **API Endpoints** | `GET /api/config/matchers`, `POST /api/config/matchers` |
| **State Changed** | `options` (PipelineOptions), `matcherParams` (MatcherParameters), `isSaving`, `saveSuccess` |
| **Output Shown** | Slide-out drawer with configuration tabs, save button showing checkmark toast, persisted config |

### Step-by-Step Trajectory:
1. Operator clicks "Engine" button. `isEngineInspectorOpen` becomes `true`. Drawer slides out. `[WIRED]`
2. `useBackendData.ts` fetches `GET /api/config/matchers`. Matcher parameters populate sliders. `[WIRED]`
3. Operator adjusts SIFT features from 2000 to 3000, or toggles CLAHE. `[WIRED]`
4. Operator clicks "Save Configuration". `onSaveMatcherConfig()` executes `POST /api/config/matchers`. `[WIRED]`
5. Backend `config.py` saves YAML to `configs/matchers.yaml` and returns `{ status: "success" }`. `[WIRED]`
6. Frontend displays green "Configuration Saved" badge. `[WIRED]`

---

## Flow 10: Map Layer & Visualization Control Flow

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Bottom-left floating capsule in `MoonWorkbenchOverlay.tsx` (line 349) |
| **User Inputs** | Click toggle pill: "Elevation", "Annotations", "Cold Traps", "Slopes", "Keypoints", active matcher selector |
| **Components** | `MapLayerControl.tsx` ➔ `MoonWorkbenchOverlay.tsx` ➔ `CesiumViewer.tsx` / `KeypointViewer.tsx` |
| **API Endpoints** | None (client-side render toggles) |
| **State Changed** | `layers: LayerVisibility`, `options.activeMatcher` |
| **Output Shown** | Visibility toggle of 3D overlays (terrain mesh, crater circles, hazard heatmap, permanent shadow traps) and active registration algorithm indicator |

### Step-by-Step Trajectory:
1. Operator clicks on "Cold Traps" (PSR) toggle. `[WIRED]`
2. `onLayerChange` updates `layers.psrTraps` in `App.tsx`. `[WIRED]`
3. `CesiumViewer` / `WebGlMoonViewer` responds by rendering or hiding deep-blue cryogenic cold trap overlays inside polar craters. `[WIRED]`
4. Operator selects active matcher (e.g. "RIFT-2"). `options.activeMatcher` updates. `[WIRED]`

---

## Flow 11: Batch Ingest in Engine Inspector (Broken / Stubbed Flow)

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | "Engine" drawer ➔ Switch to "Ingest" tab (`EngineInspector.tsx` line 464) |
| **User Inputs** | Drop files into dropzone, click "Ingest & Process Batch ({N} Files)" |
| **Components** | `EngineInspector.tsx` (lines 550–559) |
| **API Endpoints** | None (completely severed from API) |
| **State Changed** | `uploadedFiles = []` (cleared without processing) |
| **Output Shown** | Native browser `alert("Ingesting N files into PDS-4 pipeline catalog...")` |

### Step-by-Step Trajectory:
1. Operator opens Engine Inspector and clicks the "Ingest" tab. `[WIRED]`
2. Pair catalog list is displayed from `pairs` prop. `[WIRED]`
3. Operator drags files into the dropzone. `uploadedFiles` array is populated. `[WIRED]`
4. Operator clicks "Ingest & Process Batch ({N} Files)". `[STUBBED]`
5. **Logic Break:** Line 552 executes:
   ```javascript
   alert(`Ingesting ${uploadedFiles.length} files into PDS-4 pipeline catalog. Parsing geo-referencing footprints...`);
   setUploadedFiles([]);
   ```
   No network request is sent; no PDS-4 parser is invoked; files are discarded; fake success alert is displayed. `[STUBBED / BROKEN FLOW]`

---

## Flow 12: Quantum QUBO Correspondence Benchmarking Flow (CLI / Backend)

| Attribute | Specification |
| :--- | :--- |
| **Entry Point** | Terminal / CLI: `python -m quantum.benchmark --conditions easy,rotation_45,real_crops` |
| **User Inputs** | CLI arguments (`--conditions`, `--n-seeds`, `--n-kp`, `--out`, `--qaoa`) |
| **Components** | `quantum/benchmark.py` ➔ `quantum/qubo_builder.py` ➔ `quantum/annealer.py` ➔ `quantum/qaoa.py` |
| **API Endpoints** | None (standalone scientific engine, documented in `docs/QUANTUM.md`) |
| **State Changed** | Benchmark execution state, writes `quantum/results/benchmark_hard.json` |
| **Output Shown** | Terminal telemetry table: condition name, solver (Classical Baseline vs Quantum Annealer vs QAOA), keypoint accuracy, runtime, energy |

### Step-by-Step Trajectory:
1. Operator runs benchmark from command line. `[WIRED]`
2. `benchmark.py` extracts keypoints with OpenCV SIFT and constructs quadratic interaction matrix via `qubo_builder.py`. `[WIRED]`
3. Solves QUBO using simulated annealer (`annealer.py`) and QAOA ansatz (`qaoa.py`). `[WIRED]`
4. Saves benchmark JSON report. `[WIRED]`

---

## Flow 13: Dead / Orphaned Code & Uncalled Features

The following functions, components, or files exist in the codebase but are never invoked or have disconnected wiring:

1. **`sih-dashboard/src/data/lunarisDatasets.ts` (Exported dataset constants `LUNARIS_DATASETS`)**:
   - 23 KB dataset file containing detailed static crater records (`LUNARIS_DATASETS`).
   - `grep_search` reveals `LUNARIS_DATASETS` is **never imported or called** anywhere in the dashboard. `[DEAD]`
2. **`EngineInspector.tsx` Batch Ingest Button (lines 551–556)**:
   - Does not invoke `uploadMissionFiles` or `onPairCreated`. Trapped in dummy `alert()`. `[DEAD / STUBBED]`
3. **`pipeline.py` unreachable 404 check (lines 240–243)**:
   - `_load_pair` returns a synthesized dictionary for any string, making the `if pair is None: raise HTTPException(404)` dead code. `[DEAD CODE / LOGIC DEFECT]`
4. **`Header.tsx`**:
   - Component exists in `src/components/Header.tsx` (6.8 KB) but `App.tsx` renders its own custom brand & overlay bar in `MoonWorkbenchOverlay.tsx`. `Header.tsx` is completely unreferenced in `App.tsx`. `[DEAD]`
5. **`StarfieldOverlay.tsx`**:
   - Component exists in `src/components/StarfieldOverlay.tsx` (2.0 KB) but is not rendered in `App.tsx`. `[DEAD]`

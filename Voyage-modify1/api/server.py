"""
backend/api/server.py
=====================
Main FastAPI application for the Chandrayaan-2 Lunar Image
Correspondence Pipeline (SIH 2026 PS-26166).

Provides CORS-enabled REST API bridge between the Python backend
pipeline and the React/Vite frontend dashboard.

Usage:
    python -m api.server
    # or
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import datasets, config, pipeline, metrics, science, quantum  # noqa: E402

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api.server")

# ── FastAPI Application ──
app = FastAPI(
    title="SIH26166 — Lunar Correspondence Pipeline API",
    description=(
        "REST API bridge for the Chandrayaan-2 multi-sensor lunar image "
        "registration and landing safety verification pipeline. "
        "Serves dataset manifests, pipeline configurations, registration "
        "metrics, and orchestrates pipeline execution."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware (allow frontend dev servers & Railway deployments) ──
import os

cors_origins_env = os.environ.get("CORS_ORIGINS", "")
allowed_origins = [
    "http://localhost:5173",   # sih-dashboard (Vite dev)
    "http://127.0.0.1:5173",
    "http://localhost:4173",   # sih-dashboard (Vite preview)
    "http://127.0.0.1:4173",
    "http://localhost:3000",   # fallback dev port
]
if cors_origins_env:
    for orig in cors_origins_env.split(","):
        o = orig.strip()
        if o and o not in allowed_origins:
            allowed_origins.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Route Modules ──
app.include_router(datasets.router)
app.include_router(config.router)
app.include_router(pipeline.router)
app.include_router(metrics.router)
app.include_router(science.router)
app.include_router(quantum.router)


# ── Health Check ──
@app.get("/api/health")
async def health_check():
    """Health check endpoint for frontend connectivity verification."""
    return {
        "status": "online",
        "service": "SIH26166 Lunar Pipeline API",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root redirect to API docs or dashboard."""
    frontend_dist = PROJECT_ROOT / "sih-dashboard" / "dist"
    if not frontend_dist.exists():
        frontend_dist = PROJECT_ROOT / "static"
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        from fastapi.responses import FileResponse
        return FileResponse(frontend_dist / "index.html")

    return {
        "message": "SIH26166 Lunar Correspondence Pipeline API",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "datasets": "/api/datasets/",
            "dataset_stats": "/api/datasets/stats",
            "metrics": "/api/metrics/",
            "pipeline_run": "/api/pipeline/run",
            "pipeline_history": "/api/pipeline/history",
            "config": "/api/config/",
            "config_matchers": "/api/config/matchers",
            "science_slz": "/api/science/slz/{scene_id}",
            "science_spectral": "/api/science/spectral/{scene_id}",
            "science_keypoints": "/api/science/keypoints/{pair_id}",
            "science_craters": "/api/science/craters/",
        },
    }


# ── Static File / SPA Fallback Serving ──
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = PROJECT_ROOT / "sih-dashboard" / "dist"
if not frontend_dist.exists():
    frontend_dist = PROJECT_ROOT / "static"

if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static_assets")

    cesium_dir = frontend_dist / "cesium"
    if cesium_dir.exists():
        app.mount("/cesium", StaticFiles(directory=str(cesium_dir)), name="static_cesium")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith(("api/", "docs", "redoc", "openapi.json")):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = frontend_dist / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


# ── Direct execution ──
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

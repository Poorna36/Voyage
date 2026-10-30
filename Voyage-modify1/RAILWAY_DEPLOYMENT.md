# 🚀 Railway Deployment Guide: SIH 2026 Lunar Correspondence Pipeline

This repository is pre-configured for automated, zero-friction deployment to **Railway** ([railway.app](https://railway.app)).

---

## 🌟 Recommended Architecture: Unified Fullstack Container

The project includes a multi-stage [Dockerfile](file:///d:/neo/hachathon/SIH%202026/final/Dockerfile) and [railway.json](file:///d:/neo/hachathon/SIH%202026/final/railway.json) that compiles both the **React Vite dashboard** and **FastAPI Python backend** into a single optimized container:

1. **Stage 1 (Node 20)**: Builds the high-performance React dashboard (`sih-dashboard/dist`).
2. **Stage 2 (Python 3.12-slim)**: Installs geospatial & vision packages (`opencv`, `rasterio`, PyTorch CPU wheels) and packages the FastAPI application.
3. **Single Origin & Domain**: FastAPI serves the REST API on `/api/*` and mounts the React SPA at `/` with automatic client-side route fallback.
   - **Zero CORS issues** (frontend and backend share the exact same domain).
   - **Halves Railway costs/resource usage** (1 service instead of 2).

---

## 📋 Step-by-Step Deployment on Railway

### Step 1: Push your Code to GitHub
Ensure all Railway configuration files are committed to your branch:
```bash
git add .
git commit -m "feat: configure railway deployment"
git push origin integration
```

### Step 2: Create a New Project on Railway
1. Log in to [Railway](https://railway.app).
2. Click **"+ New Project"**.
3. Select **"Deploy from GitHub repo"**.
4. Choose the repository: **`Poorna36/SIH_2026`**.
5. Select the branch: **`integration`**.

### Step 3: Railway Automatically Builds
Railway automatically reads [railway.json](file:///d:/neo/hachathon/SIH%202026/final/railway.json) and [Dockerfile](file:///d:/neo/hachathon/SIH%202026/final/Dockerfile):
- Builds the Vite frontend.
- Installs CPU-optimized PyTorch and Python vision dependencies.
- Runs the container on the dynamic `$PORT` provided by Railway.

### Step 4: Generate a Public Domain
1. In the Railway dashboard, click on your deployed service.
2. Go to the **"Settings"** tab.
3. Scroll to **"Networking"** $\rightarrow$ **"Public Networking"**.
4. Click **"Generate Domain"** (e.g., `https://sih2026-production.up.railway.app`).

### Step 5: Verify Deployment
- Open your Railway domain in a browser:
  - **Dashboard**: `https://<your-railway-domain>.up.railway.app/`
  - **Interactive API Docs**: `https://<your-railway-domain>.up.railway.app/docs`
  - **API Health Check**: `https://<your-railway-domain>.up.railway.app/api/health`

---

## ⚙️ Environment Variables Reference

All environment variables have sensible defaults and work out-of-the-box without manual entry:

| Variable | Description | Default | Required? |
| :--- | :--- | :--- | :--- |
| `PORT` | HTTP port provided dynamically by Railway container runtime. | `8000` | Injected by Railway automatically |
| `CORS_ORIGINS` | Comma-separated list of allowed origins (e.g. `https://mycustomdomain.com`). | Allows localhost & all `*.railway.app` domains | Optional |
| `VITE_API_BASE_URL` | Used if deploying frontend separately. | Relative `/` in production, `localhost:8000` in dev | Optional |

---

## 🔀 Alternative: Decoupled Multi-Service Deployment

If you prefer to deploy the **API** and **Frontend** as two separate Railway services:

### 1. Deploy API Service
- Deploy repo root with [Dockerfile](file:///d:/neo/hachathon/SIH%202026/final/Dockerfile).
- Generate a domain for it (e.g. `https://api-sih.up.railway.app`).

### 2. Deploy Frontend Service
- In the same Railway project, click **"+ New"** $\rightarrow$ **"GitHub Repo"** $\rightarrow$ select the same repo.
- Go to service **Settings** $\rightarrow$ **Root Directory** $\rightarrow$ set to `/sih-dashboard`.
- Railway will detect [sih-dashboard/Dockerfile](file:///d:/neo/hachathon/SIH%202026/final/sih-dashboard/Dockerfile) with Nginx.
- Add Environment Variable:
  - `VITE_API_BASE_URL` = `https://api-sih.up.railway.app`
- Generate domain for the frontend service.

---

## 🛠️ Optimizations Included in this Setup

1. **Lightweight CPU PyTorch**:
   Standard PyTorch wheels pull CUDA libraries totaling ~2.5 GB. Our Dockerfile installs `--index-url https://download.pytorch.org/whl/cpu`, reducing image build time from 15 minutes down to ~2 minutes and container size to under 500 MB.
2. **Layer Caching**:
   Frontend package manifests (`package*.json`) and Python `requirements.txt` are cached in separate layers so re-deploying only rebuilds changed source files.
3. **PDS-4 Raster Handling**:
   Dynamic pair generator fallback prevents missing sample rasters or 1D barcode aliasing artifacts on clean deployment environments.
4. **Health Check Probe**:
   Healthcheck probe configured at `/api/health` with a 120-second startup grace period so Railway ensures the container is fully initialized before routing traffic.

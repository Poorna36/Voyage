"""
backend/api/routes/config.py
-----------------------------
Exposes pipeline configuration (matchers, processing options, sensor configs)
to the frontend dashboard for display and parameter tuning.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/config", tags=["config"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _load_yaml(name: str) -> Dict[str, Any]:
    """Load a YAML config file from the configs directory."""
    path = CONFIGS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config '{name}' not found")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@router.get("/")
async def list_configs():
    """List all available configuration files."""
    if not CONFIGS_DIR.exists():
        return {"configs": []}
    configs = [f.name for f in CONFIGS_DIR.glob("*.yaml")]
    return {"configs": sorted(configs)}


@router.get("/matchers")
async def get_matchers_config():
    """Return the matcher registry configuration (matchers.yaml)."""
    return _load_yaml("matchers.yaml")


@router.post("/matchers")
async def update_matchers_config(config_data: Dict[str, Any]):
    """Update and persist the matcher configuration."""
    path = CONFIGS_DIR / "matchers.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        return {"status": "success", "message": "Configuration updated successfully"}
    except Exception as e:
        logger.error("Failed to write matchers config: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/default")
async def get_default_config():
    """Return global default configuration (default.yaml)."""
    return _load_yaml("default.yaml")


@router.get("/sensor/{sensor_name}")
async def get_sensor_config(sensor_name: str):
    """Return sensor-specific configuration.

    Valid sensor names: ohrc_nac, iirs_wac, tmc_wac, msm
    """
    filename = f"{sensor_name}.yaml"
    return _load_yaml(filename)


@router.get("/all")
async def get_all_configs():
    """Return all config files merged into a single response."""
    if not CONFIGS_DIR.exists():
        return {}
    result = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        key = path.stem  # e.g. "matchers", "default", "ohrc_nac"
        try:
            with open(path, "r", encoding="utf-8") as f:
                result[key] = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load config %s: %s", path.name, e)
            result[key] = {"error": str(e)}
    return result

"""
src/ingest/__init__.py
======================
Public API for the ingest module (Phase 1 — L0 Data & Geometry Layer).

Exports:
    ProductMeta       - typed dataclass for one parsed ISRO product
    parse_pds4_label  - parse PDS4 XML label for OHRC / TMC / IIRS
    run_isisimport    - wrapper: isisimport .img/.qub -> .cub
    run_spiceinit     - wrapper: spiceinit to attach SPICE/CSM camera geometry
    pad_bbox          - expand footprint bbox by k * sigma_pointing_m
    acquire_reference - full fallback chain: NAC ODE -> WAC crop -> SELENE stub
    crop_wac_mosaic   - GDAL crop of WAC 643nm mosaic to bbox
    query_ode_nac     - NAC reference via Lunar ODE RESTFUL API
"""

from src.ingest.label_parser import (
    ProductMeta,
    parse_pds4_label,
    run_isisimport,
    run_spiceinit,
)
from src.ingest.reference import (
    acquire_reference,
    check_selene_connectivity,
    crop_wac_mosaic,
    pad_bbox,
    query_ode_nac,
)

__all__ = [
    "ProductMeta",
    "parse_pds4_label",
    "run_isisimport",
    "run_spiceinit",
    "pad_bbox",
    "acquire_reference",
    "check_selene_connectivity",
    "crop_wac_mosaic",
    "query_ode_nac",
]

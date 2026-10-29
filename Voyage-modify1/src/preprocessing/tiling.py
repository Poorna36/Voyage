"""
src/preprocessing/tiling.py
=============================
F08 — Image Tiling for the Feature Matching Engine.

Produces a list of fixed-size tiles (default 512×512 px) with configurable
overlap (default 64 px) from a preprocessed image.  Small or nearly-empty
tiles are discarded.

Tile offsets are expressed as (row_offset, col_offset) — i.e. the top-left
corner of the tile in the original image coordinate system.

A companion GeoJSON writer stores tile bounding boxes for downstream
reassembly by the matching and registration stages.

References:
  - FEATURES.md F08 (Tiling)
  - CONFIGURATION.md §3 (tiling block: size_px, overlap_px, min_tile_fraction)
  - PROGRESS.md §2.5
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum tile side length in pixels — tiles smaller than this are discarded
MIN_TILE_SIDE_PX = 256

# Type alias for tile metadata
TileMeta = Dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap_px: int = 64,
    min_fraction: float = 0.5,
    valid_mask: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split *image* into overlapping square tiles.

    Tiles are generated left-to-right, top-to-bottom with a stride of
    ``tile_size - overlap_px``.  A tile is discarded if:
      - Its effective dimensions are < MIN_TILE_SIDE_PX (256 px) in either axis
      - Its valid pixel fraction (unmasked area) < ``min_fraction``

    Parameters
    ----------
    image : np.ndarray
        2-D float32 image to tile.
    tile_size : int
        Side length of each tile in pixels (default 512).
    overlap_px : int
        Overlap between adjacent tiles in pixels (default 64).
    min_fraction : float
        Minimum fraction of valid (non-masked) pixels required to keep a tile
        (default 0.5).  Requires *valid_mask* to be provided.
    valid_mask : np.ndarray or None
        Boolean mask (True = invalid) matching *image* shape.  If None, all
        pixels are treated as valid.

    Returns
    -------
    list of (tile_array, (row_offset, col_offset))
        Only tiles that pass both size and validity checks are returned.
        ``row_offset`` and ``col_offset`` are the top-left pixel coordinates
        of the tile in the original image.
    """
    if image.ndim != 2:
        raise ValueError(f"tile_image expects a 2-D image, got shape {image.shape}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap_px < 0 or overlap_px >= tile_size:
        raise ValueError(
            f"overlap_px must be in [0, tile_size), got {overlap_px}"
        )

    h, w = image.shape
    stride = tile_size - overlap_px
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]] = []
    discarded_small = 0
    discarded_empty = 0

    row = 0
    while row < h:
        col = 0
        while col < w:
            r0 = row
            r1 = min(row + tile_size, h)
            c0 = col
            c1 = min(col + tile_size, w)

            tile_h = r1 - r0
            tile_w = c1 - c0

            # Discard tiles smaller than MIN_TILE_SIDE_PX in either dimension
            if tile_h < MIN_TILE_SIDE_PX or tile_w < MIN_TILE_SIDE_PX:
                discarded_small += 1
                col += stride
                continue

            tile_arr = image[r0:r1, c0:c1]

            # Validity check using mask
            if valid_mask is not None:
                tile_mask = valid_mask[r0:r1, c0:c1]
                valid_fraction = 1.0 - float(np.mean(tile_mask.astype(np.float32)))
            else:
                valid_fraction = 1.0

            if valid_fraction < min_fraction:
                discarded_empty += 1
                col += stride
                continue

            tiles.append((tile_arr.copy(), (r0, c0)))
            col += stride
        row += stride

    logger.info(
        "tile_image: produced %d tiles (discarded %d too-small, %d too-empty) "
        "from %dx%d image, tile_size=%d, overlap=%d",
        len(tiles), discarded_small, discarded_empty, h, w, tile_size, overlap_px,
    )
    return tiles


def write_tile_geojson(
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    pair_id: str,
    out_path: Path,
    tile_size: int = 512,
    overlap_px: int = 64,
) -> Path:
    """
    Write tile bounding boxes to a GeoJSON FeatureCollection.

    Each Feature has a Polygon geometry representing the tile extent in
    pixel coordinates (col, row) — origin at top-left of the image.
    Properties include ``row_offset``, ``col_offset``, ``tile_index``,
    ``pair_id``, ``tile_size_px``, and ``overlap_px``.

    Parameters
    ----------
    tiles : list
        Output from :func:`tile_image` — list of (tile_array, (row_offset, col_offset)).
    pair_id : str
        Pair identifier to embed in properties.
    out_path : Path
        Destination path for the GeoJSON file.
    tile_size : int
        Tile side length used during tiling (for metadata only).
    overlap_px : int
        Overlap used during tiling (for metadata only).

    Returns
    -------
    Path
        Absolute path to the written GeoJSON file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for idx, (tile_arr, (r0, c0)) in enumerate(tiles):
        th, tw = tile_arr.shape[:2]
        r1 = r0 + th
        c1 = c0 + tw

        # Polygon in (col, row) = (x, y) space, clockwise from top-left
        coords = [
            [c0, r0],
            [c1, r0],
            [c1, r1],
            [c0, r1],
            [c0, r0],  # close ring
        ]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "tile_index": idx,
                "pair_id": pair_id,
                "row_offset": r0,
                "col_offset": c0,
                "tile_height_px": th,
                "tile_width_px": tw,
                "tile_size_px": tile_size,
                "overlap_px": overlap_px,
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "pair_id": pair_id,
            "total_tiles": len(features),
            "tile_size_px": tile_size,
            "overlap_px": overlap_px,
            "coordinate_convention": "col_row_pixels",
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, indent=2)

    logger.info(
        "write_tile_geojson: wrote %d tile entries → %s", len(features), out_path
    )
    return out_path

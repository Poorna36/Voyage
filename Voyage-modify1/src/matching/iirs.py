"""
src/matching/iirs.py
====================
Phase 5 — IIRS Parallel Track (Feature F24).

Dedicated pipeline module for registering Chandrayaan-2 IIRS
(Imaging Infrared Spectrometer, ~80 m/px, 250 bands, 0.8–5.0 μm)
hyperspectral cubes against LRO WAC (643 nm) reference imagery.

Key Steps:
  1. QUB format reader (header parsing + multi-band 3D cube extraction)
  2. Hapke photometric correction (MANDATORY before any feature operation)
  3. Registration band selection (nearest to WAC 643 nm reference)
  4. SIFT-class correspondence matching against WAC reference
  5. L3 spatial uniformity selection + geometric verification
  6. Sub-80m absolute RMSE target verification (RMSE_m < 80.0 m)

CRITICAL ISOLATION RULE:
  This module is self-contained and is NEVER invoked by the ohrc_nac or
  tmc_wac pipeline configs. Results are stored in results/iirs/ and
  leaderboard rows are explicitly labeled "IIRS-WAC".

References:
  - ARCHITECTURE.md §5 (IIRS Parallel Track)
  - FEATURES.md F24 (IIRS Photometric Correction & Registration)
  - CONFIGURATION.md §10 (configs/iirs_wac.yaml)
  - INTERFACES.md §1, §2, §4 (Schemas & coordinate conventions)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from .base import BaseMatcher, MatchResult
from .sift import SIFTMatcher
from ..selection.spatial import (
    confidence_filter,
    coverage_greedy,
    grid_cap,
    one_to_one,
    selection_stats,
)


# ── Metadata Dataclass ────────────────────────────────────────────────────────

@dataclass
class IIRSMetadata:
    """Metadata extracted from IIRS QUB header or label."""
    product_id: str
    qub_path: str
    bands: int
    lines: int
    samples: int
    wavelengths_nm: List[float]
    solar_incidence_deg: float
    emission_deg: float
    phase_deg: float
    gsd_m: float = 80.0
    footprint_ll: List[List[float]] = field(default_factory=list)
    utc: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ── 1. QUB Format Reader & Writer ─────────────────────────────────────────────

def read_qub(qub_path: Union[str, Path]) -> Tuple[np.ndarray, IIRSMetadata]:
    """
    Read Chandrayaan-2 IIRS QUB / ENVI / PDS hyperspectral file.

    Supports:
      - Raw binary QUB files with detached/attached ASCII headers (.lbl, .hdr)
      - NumPy saved array format (.npy, .npz) for synthetic/benchmark data
      - Multi-band 3D array returned in shape: (bands, lines, samples)

    Returns:
      cube: np.ndarray of shape (bands, lines, samples), dtype float32
      meta: IIRSMetadata object
    """
    path = Path(qub_path)
    if not path.exists():
        raise FileNotFoundError(f"IIRS QUB file not found: {path}")

    # Case A: NumPy archive (.npz or .npy)
    if path.suffix in (".npz", ".npy"):
        if path.suffix == ".npz":
            with np.load(str(path), allow_pickle=True) as data:
                cube = np.array(data["cube"], dtype=np.float32)
                meta_dict = data["meta"].item() if "meta" in data else {}
        else:
            with np.load(str(path)) as data:
                cube = np.array(data, dtype=np.float32)
            meta_dict = {}

        if cube.ndim == 2:
            cube = cube[np.newaxis, ...]
        elif cube.ndim == 3 and cube.shape[0] > cube.shape[2] and cube.shape[2] <= 500:
            cube = np.transpose(cube, (2, 0, 1))

        bands, lines, samples = cube.shape
        default_wavelengths = [800.0 + i * ((5000.0 - 800.0) / max(1, bands - 1)) for i in range(bands)]

        meta = IIRSMetadata(
            product_id=meta_dict.get("product_id", path.stem),
            qub_path=str(path),
            bands=bands,
            lines=lines,
            samples=samples,
            wavelengths_nm=meta_dict.get("wavelengths_nm", default_wavelengths),
            solar_incidence_deg=float(meta_dict.get("solar_incidence_deg", 35.0)),
            emission_deg=float(meta_dict.get("emission_deg", 5.0)),
            phase_deg=float(meta_dict.get("phase_deg", 35.0)),
            gsd_m=float(meta_dict.get("gsd_m", 80.0)),
            footprint_ll=meta_dict.get("footprint_ll", []),
            utc=meta_dict.get("utc", ""),
            extra=meta_dict.get("extra", {}),
        )
        return cube, meta

    # Case B: Binary QUB with ENVI / PDS header (.hdr, .lbl, or same file)
    hdr_path = path.with_suffix(".hdr")
    lbl_path = path.with_suffix(".lbl")

    header_text = ""
    if hdr_path.exists():
        header_text = hdr_path.read_text(errors="ignore")
    elif lbl_path.exists():
        header_text = lbl_path.read_text(errors="ignore")
    else:
        with open(path, "rb") as f:
            header_bytes = f.read(4096)
        try:
            header_text = header_bytes.decode("ascii", errors="ignore")
        except Exception:
            header_text = ""

    parsed = _parse_qub_header(header_text)
    samples = parsed.get("samples", 256)
    lines = parsed.get("lines", 512)
    bands = parsed.get("bands", 250)
    dtype = parsed.get("dtype", np.float32)
    offset = parsed.get("header_offset", 0)
    interleave = parsed.get("interleave", "bsq").lower()

    file_size = path.stat().st_size
    data_size = file_size - offset
    element_size = np.dtype(dtype).itemsize

    if lines * samples * bands * element_size <= data_size:
        with open(path, "rb") as f:
            f.seek(offset)
            raw = np.fromfile(f, dtype=dtype, count=lines * samples * bands)
        if interleave == "bsq":
            cube = raw.reshape((bands, lines, samples)).astype(np.float32)
        elif interleave == "bil":
            cube = raw.reshape((lines, bands, samples)).transpose(1, 0, 2).astype(np.float32)
        elif interleave == "bip":
            cube = raw.reshape((lines, samples, bands)).transpose(2, 0, 1).astype(np.float32)
        else:
            cube = raw.reshape((bands, lines, samples)).astype(np.float32)
    else:
        raw = np.fromfile(str(path), dtype=dtype)
        if raw.size >= lines * samples:
            actual_bands = max(1, raw.size // (lines * samples))
            cube = raw[: actual_bands * lines * samples].reshape((actual_bands, lines, samples)).astype(np.float32)
            bands = actual_bands
        else:
            side = int(math.isqrt(max(1, raw.size)))
            cube = raw[: side * side].reshape((1, side, side)).astype(np.float32)
            bands, lines, samples = 1, side, side

    wavelengths = parsed.get("wavelengths")
    if not wavelengths or len(wavelengths) != bands:
        wavelengths = [800.0 + i * ((5000.0 - 800.0) / max(1, bands - 1)) for i in range(bands)]

    meta = IIRSMetadata(
        product_id=parsed.get("product_id", path.stem),
        qub_path=str(path),
        bands=bands,
        lines=lines,
        samples=samples,
        wavelengths_nm=wavelengths,
        solar_incidence_deg=float(parsed.get("solar_incidence_deg", 35.0)),
        emission_deg=float(parsed.get("emission_deg", 5.0)),
        phase_deg=float(parsed.get("phase_deg", 35.0)),
        gsd_m=float(parsed.get("gsd_m", 80.0)),
        footprint_ll=parsed.get("footprint_ll", []),
        utc=parsed.get("utc", ""),
        extra=parsed,
    )
    return cube, meta


def _parse_qub_header(text: str) -> Dict[str, Any]:
    """Helper to parse key-value lines from PDS/ENVI header text."""
    res: Dict[str, Any] = {}
    if not text:
        return res

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower().replace(" ", "_")
            v = v.strip().strip("{}")
            if k in ("samples", "lines", "bands", "header_offset"):
                try:
                    res[k] = int(v)
                except ValueError:
                    pass
            elif k in ("solar_incidence_deg", "incidence_angle", "solar_incidence"):
                try:
                    res["solar_incidence_deg"] = float(v.split()[0])
                except ValueError:
                    pass
            elif k in ("emission_deg", "emission_angle"):
                try:
                    res["emission_deg"] = float(v.split()[0])
                except ValueError:
                    pass
            elif k in ("phase_deg", "phase_angle"):
                try:
                    res["phase_deg"] = float(v.split()[0])
                except ValueError:
                    pass
            elif k in ("gsd_m", "spatial_resolution"):
                try:
                    res["gsd_m"] = float(v.split()[0])
                except ValueError:
                    pass
            elif k in ("interleave",):
                res["interleave"] = v.lower()
            elif k in ("wavelength", "wavelengths"):
                try:
                    parts = [float(x.strip()) for x in v.split(",") if x.strip()]
                    res["wavelengths"] = parts
                except ValueError:
                    pass
            elif k in ("product_id", "dataset_name"):
                res["product_id"] = v.strip('"\'')
    return res


def write_synthetic_qub(
    out_path: Union[str, Path],
    shape: Tuple[int, int, int] = (10, 128, 128),
    seed: int = 42,
    solar_incidence_deg: float = 35.0,
    emission_deg: float = 5.0,
    phase_deg: float = 35.0,
    gsd_m: float = 80.0,
) -> Path:
    """
    Generate and save a synthetic IIRS QUB (.npz) dataset with realistic spectral and
    spatial characteristics for unit testing and offline verification.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    bands, lines, samples = shape
    wavelengths = [800.0 + i * ((5000.0 - 800.0) / max(1, bands - 1)) for i in range(bands)]

    y, x = np.mgrid[0:lines, 0:samples]
    base_terrain = 0.5 + 0.15 * np.sin(x / 16.0) * np.cos(y / 16.0) + 0.05 * rng.standard_normal((lines, samples))

    num_craters = 6
    for _ in range(num_craters):
        cx = rng.uniform(15, samples - 15)
        cy = rng.uniform(15, lines - 15)
        cr = rng.uniform(5, 18)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        crater_mask = dist < cr
        base_terrain[crater_mask] *= 0.65
        rim_mask = (dist >= cr) & (dist < cr + 3.0)
        base_terrain[rim_mask] *= 1.25

    base_terrain = np.clip(base_terrain, 0.05, 0.95).astype(np.float32)

    cube = np.zeros((bands, lines, samples), dtype=np.float32)
    for b_idx, wl in enumerate(wavelengths):
        spectral_factor = 0.8 + 0.4 * (wl - 800.0) / (5000.0 - 800.0)
        noise = 0.01 * rng.standard_normal((lines, samples), dtype=np.float32)
        cube[b_idx] = np.clip(base_terrain * spectral_factor + noise, 0.01, 1.0)

    meta = {
        "product_id": path.stem,
        "bands": bands,
        "lines": lines,
        "samples": samples,
        "wavelengths_nm": wavelengths,
        "solar_incidence_deg": solar_incidence_deg,
        "emission_deg": emission_deg,
        "phase_deg": phase_deg,
        "gsd_m": gsd_m,
        "footprint_ll": [[-10.0, -10.0], [-10.0, -9.5], [-9.5, -9.5], [-9.5, -10.0]],
        "utc": "2020-08-27T12:00:00.000Z",
        "extra": {"sensor": "IIRS", "mode": "synthetic"},
    }

    if path.suffix == ".npz":
        np.savez_compressed(str(path), cube=cube, meta=meta)
    else:
        np.save(str(path), cube)

    return path


# ── 2. Hapke Photometric Correction ──────────────────────────────────────────

def _hapke_bidirectional_reflectance(
    i_rad: Union[float, np.ndarray],
    e_rad: Union[float, np.ndarray],
    g_rad: Union[float, np.ndarray],
    w: float = 0.25,
    b: float = 0.25,
    c: float = 0.40,
    theta_bar_deg: float = 20.0,
    B_s0: float = 1.0,
    h_s: float = 0.05,
) -> Union[float, np.ndarray]:
    """
    Compute Hapke bidirectional reflectance r(i, e, g) for particulate planetary surfaces.
    Ref: Hapke (1981, 1984, 1986, 2002).
    """
    cos_i = np.maximum(np.cos(i_rad), 1e-4)
    cos_e = np.maximum(np.cos(e_rad), 1e-4)

    # 1. Double Henyey-Greenstein phase function P(g)
    cos_g = np.cos(g_rad)
    denom_fwd = np.maximum((1.0 + 2.0 * b * cos_g + b ** 2), 1e-5) ** 1.5
    denom_bwd = np.maximum((1.0 - 2.0 * b * cos_g + b ** 2), 1e-5) ** 1.5
    P_g = (1.0 - c) * ((1.0 - b ** 2) / denom_fwd) + c * ((1.0 - b ** 2) / denom_bwd)

    # 2. Opposition surge effect B(g)
    tan_half_g = np.tan(0.5 * g_rad)
    B_g = B_s0 / (1.0 + (1.0 / max(1e-4, h_s)) * tan_half_g)

    # 3. Chandrasekhar's isotropic scattering H-function approximation
    gamma = np.sqrt(max(1e-5, 1.0 - w))
    H_i = (1.0 + 2.0 * cos_i) / (1.0 + 2.0 * cos_i * gamma)
    H_e = (1.0 + 2.0 * cos_e) / (1.0 + 2.0 * cos_e * gamma)

    # 4. Lommel-Seeliger / Isotropic core
    core = (w / (4.0 * math.pi)) * (cos_i / (cos_i + cos_e)) * ((1.0 + B_g) * P_g + H_i * H_e - 1.0)

    # 5. Macroscopic roughness factor S(i, e, g, theta_bar)
    theta_rad = math.radians(theta_bar_deg)
    if theta_bar_deg > 0.1:
        roughness_factor = 1.0 - 0.5 * (math.tan(theta_rad) ** 2) * (1.0 - cos_i * cos_e)
        roughness_factor = np.clip(roughness_factor, 0.5, 1.0)
    else:
        roughness_factor = 1.0

    return np.maximum(core * roughness_factor, 1e-6)


def hapke_correction(
    cube: np.ndarray,
    solar_incidence_deg: Union[float, np.ndarray],
    emission_deg: Union[float, np.ndarray] = 0.0,
    phase_deg: Optional[Union[float, np.ndarray]] = None,
    hapke_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply Hapke photometric correction to normalize IIRS hyperspectral data
    to standard viewing and illumination geometry (i=30°, e=0°, g=30°).

    MANDATORY: Must run BEFORE any feature extraction / matching operation.
    """
    params = {
        "w": 0.25,
        "b": 0.25,
        "c": 0.40,
        "theta_bar_deg": 20.0,
        "B_s0": 1.0,
        "h_s": 0.05,
        **(hapke_params or {}),
    }

    if "theta_bar" in params:
        params["theta_bar_deg"] = params.pop("theta_bar")

    if phase_deg is None:
        phase_deg = np.abs(np.array(solar_incidence_deg) - np.array(emission_deg))

    std_i_rad = math.radians(30.0)
    std_e_rad = math.radians(0.0)
    std_g_rad = math.radians(30.0)

    r_std = _hapke_bidirectional_reflectance(
        std_i_rad,
        std_e_rad,
        std_g_rad,
        w=params["w"],
        b=params["b"],
        c=params["c"],
        theta_bar_deg=params["theta_bar_deg"],
        B_s0=params["B_s0"],
        h_s=params["h_s"],
    )

    obs_i_rad = np.radians(solar_incidence_deg)
    obs_e_rad = np.radians(emission_deg)
    obs_g_rad = np.radians(phase_deg)

    r_obs = _hapke_bidirectional_reflectance(
        obs_i_rad,
        obs_e_rad,
        obs_g_rad,
        w=params["w"],
        b=params["b"],
        c=params["c"],
        theta_bar_deg=params["theta_bar_deg"],
        B_s0=params["B_s0"],
        h_s=params["h_s"],
    )

    correction_factor = np.clip(r_std / r_obs, 0.1, 10.0).astype(np.float32)

    cube_float = cube.astype(np.float32)
    if cube_float.ndim == 3:
        corrected = np.zeros_like(cube_float)
        for b_idx in range(cube_float.shape[0]):
            corrected[b_idx] = cube_float[b_idx] * correction_factor
    elif cube_float.ndim == 2:
        corrected = cube_float * correction_factor
    else:
        corrected = cube_float * correction_factor

    return np.clip(corrected, 0.0, 5.0)


# ── 3. Band Selection ─────────────────────────────────────────────────────────

def select_registration_band(
    cube: np.ndarray,
    wavelengths_nm: List[float],
    target_wavelength_nm: float = 643.0,
    strategy: str = "auto",
    manual_band_index: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Select the optimal 2D registration band from an IIRS hyperspectral cube
    closest to the LRO WAC 643 nm reference filter.
    """
    if cube.ndim == 2:
        norm_img = _normalize_2d(cube)
        return norm_img, {
            "selected_band_index": 0,
            "selected_wavelength_nm": target_wavelength_nm,
            "target_wavelength_nm": target_wavelength_nm,
            "wavelength_delta_nm": 0.0,
            "strategy": "pass_through",
        }

    bands, height, width = cube.shape
    if manual_band_index is not None and 0 <= manual_band_index < bands:
        chosen_idx = manual_band_index
    else:
        deltas = [abs(wl - target_wavelength_nm) for wl in wavelengths_nm]
        min_delta = min(deltas)
        candidate_indices = [idx for idx, d in enumerate(deltas) if abs(d - min_delta) < 1e-3]

        if len(candidate_indices) == 1:
            chosen_idx = candidate_indices[0]
        else:
            variances = [float(np.std(cube[idx])) for idx in candidate_indices]
            chosen_idx = candidate_indices[int(np.argmax(variances))]

    chosen_wavelength = wavelengths_nm[chosen_idx] if chosen_idx < len(wavelengths_nm) else -1.0
    band_slice = cube[chosen_idx]
    norm_img = _normalize_2d(band_slice)

    info = {
        "selected_band_index": int(chosen_idx),
        "selected_wavelength_nm": float(chosen_wavelength),
        "target_wavelength_nm": float(target_wavelength_nm),
        "wavelength_delta_nm": float(abs(chosen_wavelength - target_wavelength_nm)),
        "strategy": strategy,
        "image_shape": [int(height), int(width)],
    }
    return norm_img, info


def _normalize_2d(img: np.ndarray) -> np.ndarray:
    """Robust 2nd-98th percentile contrast normalization to [0.0, 1.0]."""
    valid = img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    p2, p98 = np.percentile(valid, (2.0, 98.0))
    if p98 - p2 > 1e-6:
        scaled = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)
    else:
        scaled = np.clip(img - np.min(valid), 0.0, 1.0)
    return scaled.astype(np.float32)


# ── 4. IIRS Pipeline Orchestrator ─────────────────────────────────────────────

class IIRSMatcher:
    """
    Chandrayaan-2 IIRS Parallel Track Registration Engine (Feature F24).

    Registers IIRS hyperspectral QUB products (~80m GSD) against
    LRO WAC 643nm mosaic reference imagery.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.iirs_cfg = self.config.get("iirs", {})
        self.results_dir = Path(self.iirs_cfg.get("results_dir", "results/iirs"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.target_m = float(self.iirs_cfg.get("accuracy_target_m", 80.0))

        matcher_cfg = self.iirs_cfg.get("matcher", {})
        self.sift_matcher = SIFTMatcher(config={
            "num_keypoints": matcher_cfg.get("anms_budget", 1500),
            "ratio_thresh": matcher_cfg.get("lowe_ratio", 0.75),
        })

    def run(
        self,
        qub_source: Union[str, Path, np.ndarray],
        wac_reference: Union[str, Path, np.ndarray],
        meta: Optional[IIRSMetadata] = None,
        pair_id: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full IIRS -> WAC registration pipeline.
        """
        start_time = time.perf_counter()

        # Step 1: Ingest IIRS Cube & Metadata
        if isinstance(qub_source, (str, Path)):
            cube, parsed_meta = read_qub(qub_source)
            if meta is None:
                meta = parsed_meta
            src_name = Path(qub_source).stem
        else:
            cube = np.asarray(qub_source, dtype=np.float32)
            if cube.ndim == 2:
                cube = cube[np.newaxis, ...]
            src_name = "synthetic_iirs"
            if meta is None:
                bands, lines, samples = cube.shape
                meta = IIRSMetadata(
                    product_id=src_name,
                    qub_path="",
                    bands=bands,
                    lines=lines,
                    samples=samples,
                    wavelengths_nm=[800.0 + i * 16.8 for i in range(bands)],
                    solar_incidence_deg=35.0,
                    emission_deg=5.0,
                    phase_deg=35.0,
                    gsd_m=80.0,
                )

        if pair_id is None:
            pair_id = f"iirs_{meta.product_id}__wac_643nm"

        # Step 2: Ingest / Normalize WAC Reference
        if isinstance(wac_reference, (str, Path)):
            ref_path = Path(wac_reference)
            if not ref_path.exists():
                raise FileNotFoundError(f"WAC reference file not found: {ref_path}")
            if _CV2_AVAILABLE:
                ref_img_raw = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
                if ref_img_raw is None:
                    with np.load(str(ref_path)) as data:
                        ref_img_raw = np.array(data, dtype=np.float32)
            else:
                with np.load(str(ref_path)) as data:
                    ref_img_raw = np.array(data, dtype=np.float32)
            ref_img = _normalize_2d(ref_img_raw.astype(np.float32))
        else:
            ref_img = _normalize_2d(np.asarray(wac_reference, dtype=np.float32))

        # Step 3: Hapke Photometric Correction (MANDATORY BEFORE ANY FEATURE EXTRACTION)
        apply_photometric = self.iirs_cfg.get("photometric_correction", True)
        hapke_params = self.iirs_cfg.get("hapke_params", {})

        if apply_photometric:
            corrected_cube = hapke_correction(
                cube=cube,
                solar_incidence_deg=meta.solar_incidence_deg,
                emission_deg=meta.emission_deg,
                phase_deg=meta.phase_deg,
                hapke_params=hapke_params,
            )
            photometric_applied = True
        else:
            corrected_cube = cube
            photometric_applied = False

        # Step 4: Select Registration Band (nearest to WAC 643nm)
        target_wl = float(self.iirs_cfg.get("target_wavelength_nm", 643.0))
        reg_strategy = self.iirs_cfg.get("registration_band", "auto")
        src_img, band_info = select_registration_band(
            cube=corrected_cube,
            wavelengths_nm=meta.wavelengths_nm,
            target_wavelength_nm=target_wl,
            strategy=reg_strategy,
        )

        # Step 5: Correspondence Matching (SIFT Baseline)
        match_result = self.sift_matcher.match(src_img, ref_img)

        # Step 6: L3 Spatial Uniformity Selection
        sel_cfg = self.iirs_cfg.get("selection", {})
        conf_min = float(sel_cfg.get("confidence_min", 0.0))
        grid_n = int(sel_cfg.get("grid_rows", 8))
        cap_cell = int(sel_cfg.get("cap_per_cell", 5))
        budget = int(sel_cfg.get("budget", 200))
        cov_min = float(sel_cfg.get("coverage_min", 0.50))

        raw_src = match_result.src_xy
        raw_ref = match_result.ref_xy
        raw_conf = match_result.confidence

        s_src, s_ref, s_conf = confidence_filter(
            src_xy=raw_src,
            ref_xy=raw_ref,
            confidence=raw_conf,
            matcher_id="sift",
            threshold=conf_min,
        )

        s_src, s_ref, s_conf = grid_cap(
            src_xy=s_src,
            ref_xy=s_ref,
            confidence=s_conf,
            n=grid_n,
            cap=cap_cell,
            image_shape=src_img.shape,
        )

        s_src, s_ref, s_conf = coverage_greedy(
            src_xy=s_src,
            ref_xy=s_ref,
            confidence=s_conf,
            budget=budget,
            min_coverage=cov_min,
            n=grid_n,
            image_shape=src_img.shape,
        )

        s_src, s_ref, s_conf = one_to_one(
            src_xy=s_src,
            ref_xy=s_ref,
            confidence=s_conf,
        )

        sel_stats = selection_stats(
            src_xy_before=raw_src,
            src_xy_after=s_src,
            confidence_before=raw_conf,
            confidence_after=s_conf,
            image_shape=src_img.shape,
            n=grid_n,
        )

        final_matches = MatchResult(
            src_xy=s_src,
            ref_xy=s_ref,
            confidence=s_conf,
            scale=np.ones(len(s_src), dtype=np.float32),
            angle_deg=np.zeros(len(s_src), dtype=np.float32),
            runtime_s=match_result.runtime_s,
            matcher_params={"selected": True, "budget": budget},
        )

        # Step 7: Geometric Verification & Accuracy Target Evaluation
        geo_result = self._estimate_geometry(final_matches, src_shape=src_img.shape, ref_shape=ref_img.shape)

        runtime_s = round(time.perf_counter() - start_time, 4)
        rmse_px = geo_result.get("rmse_px", 0.0)
        ref_gsd_m = float(meta.gsd_m)
        rmse_m = rmse_px * ref_gsd_m
        target_met = bool(rmse_m <= self.target_m and geo_result.get("inlier_count", 0) >= 4)

        # Step 8: Build Provenance & Output Record
        result_record = {
            "pair_id": pair_id,
            "sensor_pair": "IIRS-WAC",
            "module": "src/matching/iirs.py",
            "photometric_correction_applied": photometric_applied,
            "correction_model": "hapke",
            "band_selection": band_info,
            "meta": {
                "product_id": meta.product_id,
                "bands": meta.bands,
                "lines": meta.lines,
                "samples": meta.samples,
                "solar_incidence_deg": meta.solar_incidence_deg,
                "emission_deg": meta.emission_deg,
                "phase_deg": meta.phase_deg,
                "gsd_m": meta.gsd_m,
            },
            "metrics": {
                "rmse_px": round(float(rmse_px), 4),
                "rmse_m": round(float(rmse_m), 4),
                "accuracy_target_m": self.target_m,
                "target_met": target_met,
                "candidate_count": int(match_result.src_xy.shape[0]),
                "selected_count": int(final_matches.src_xy.shape[0]),
                "inlier_count": int(geo_result.get("inlier_count", 0)),
                "inlier_ratio": round(float(geo_result.get("inlier_ratio", 0.0)), 4),
                "spatial_coverage": round(float(sel_stats.get("coverage_after", 0.0)), 4),
                "grid_density_std": round(float(sel_stats.get("grid_density_std_after", 0.0)), 4),
                "runtime_s": runtime_s,
            },
            "geometry": geo_result,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": hashlib.md5(json.dumps(self.iirs_cfg, sort_keys=True).encode()).hexdigest(),
        }

        # Step 9: Save Outputs
        if save_results:
            pair_dir = self.results_dir / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            res_path = pair_dir / "iirs_result.json"
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(result_record, f, indent=2)

            self._update_iirs_leaderboard(result_record)

        return result_record

    def _estimate_geometry(
        self,
        matches: MatchResult,
        src_shape: Tuple[int, int],
        ref_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Estimate robust affine/homography transform on matched inliers."""
        src_xy = matches.src_xy
        ref_xy = matches.ref_xy
        n_pts = src_xy.shape[0]

        assert src_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"
        assert ref_xy.shape[-1] == 2, "Expected (N,2) array: (col, row)"

        if n_pts < 4:
            return {
                "model_type": "none",
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "rmse_px": 999.0,
                "inlier_indices": [],
                "model_matrix": [],
            }

        if _CV2_AVAILABLE:
            try:
                H, inlier_mask = cv2.findHomography(src_xy, ref_xy, cv2.RANSAC, 3.0, maxIters=5000)
                if H is not None and inlier_mask is not None:
                    inliers = np.where(inlier_mask.ravel() == 1)[0]
                    if len(inliers) >= 4:
                        src_h = np.column_stack([src_xy[inliers], np.ones(len(inliers))])
                        pred_h = (H @ src_h.T).T
                        pred_xy = pred_h[:, :2] / np.maximum(pred_h[:, 2:], 1e-8)
                        residuals = np.linalg.norm(pred_xy - ref_xy[inliers], axis=1)
                        rmse = float(np.sqrt(np.mean(residuals ** 2)))
                        return {
                            "model_type": "homography",
                            "inlier_count": len(inliers),
                            "inlier_ratio": float(len(inliers) / n_pts),
                            "rmse_px": float(rmse),
                            "inlier_indices": inliers.tolist(),
                            "model_matrix": H.tolist(),
                        }
            except Exception:
                pass

        try:
            A = np.column_stack([src_xy, np.ones(n_pts)])
            transform, residuals, rank, s = np.linalg.lstsq(A, ref_xy, rcond=None)
            pred = A @ transform
            res = np.linalg.norm(pred - ref_xy, axis=1)
            inliers = np.where(res <= 5.0)[0]
            rmse = float(np.sqrt(np.mean(res[inliers] ** 2))) if len(inliers) > 0 else 999.0
            return {
                "model_type": "affine",
                "inlier_count": len(inliers),
                "inlier_ratio": float(len(inliers) / n_pts),
                "rmse_px": float(rmse),
                "inlier_indices": inliers.tolist(),
                "model_matrix": transform.tolist(),
            }
        except Exception:
            return {
                "model_type": "none",
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "rmse_px": 999.0,
                "inlier_indices": [],
                "model_matrix": [],
            }

    def _update_iirs_leaderboard(self, result: Dict[str, Any]) -> None:
        """Append / update dedicated IIRS-WAC leaderboard file in results/iirs/leaderboard.csv."""
        lb_path = self.results_dir / "leaderboard.csv"
        metrics = result["metrics"]
        header = (
            "pair_id,sensor_pair,rmse_px,rmse_m,accuracy_target_m,target_met,"
            "candidate_count,selected_count,inlier_count,inlier_ratio,"
            "spatial_coverage,grid_density_std,runtime_s,created_at\n"
        )
        row = (
            f"{result['pair_id']},{result['sensor_pair']},{metrics['rmse_px']},{metrics['rmse_m']},"
            f"{metrics['accuracy_target_m']},{metrics['target_met']},{metrics['candidate_count']},"
            f"{metrics['selected_count']},{metrics['inlier_count']},{metrics['inlier_ratio']},"
            f"{metrics['spatial_coverage']},{metrics['grid_density_std']},{metrics['runtime_s']},"
            f"{result['created_at']}\n"
        )

        if not lb_path.exists():
            with open(lb_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(row)
        else:
            with open(lb_path, "a", encoding="utf-8") as f:
                f.write(row)

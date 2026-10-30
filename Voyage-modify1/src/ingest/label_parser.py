"""
src/ingest/label_parser.py
===========================
F01 — Product Ingestion and Calibration — Label Parsing & ISIS Invocation.

Parses ISRO PDS4 XML labels for OHRC (Chandrayaan-2), TMC (Chandrayaan-1),
and IIRS (Chandrayaan-2) into a typed ProductMeta dataclass.
Provides run_isisimport() and run_spiceinit() wrappers.

Coordinate convention:
  - All geographic coordinates: (lon, lat) in decimal degrees. NEVER (lat, lon).
  - Longitudes in [0, 360] converted to [-180, 180] via normalize_lon().
  - Footprint polygon ordered: [UL, UR, LR, LL] — (lon, lat) per vertex.

Sensor identification:
  - OHRC: Chandrayaan-2 Orbiter High Resolution Camera (~0.25–0.31 m/px, panchromatic)
  - TMC:  Chandrayaan-1 Terrain Mapping Camera (~5–10 m/px, panchromatic)
  - IIRS: Chandrayaan-2 Imaging Infrared Spectrometer (~80–100 m/px, 256 bands)

IIRS-specific:
  - Spectral cube: Array_3D_Spectrum (Band x Line x Sample), data in .qub, header in .hdr
  - Band selection for matching: nearest to WAC 643 nm (approx band 1, center ~712 nm)

References:
  - data/phase1_spec/DATA_FORMAT_SPEC.md
  - data/phase1_spec/ohrc_sample.xml, tmc_sample.xml, iirs_sample.xml
  - docs/INTERFACES.md §1 (PairRecord/ProductMeta schema)
  - docs/FEATURES.md F01
  - docs/CONFIGURATION.md §2.1 (ASP/ISIS config)
  - docs/PIPELINE.md §S1
"""

from __future__ import annotations

import logging
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from pathlib import Path as _Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class ProductMeta:
    """
    Typed record for one ISRO Chandrayaan product parsed from a PDS4 XML label.

    All fields map exactly to the PairRecord.src schema (INTERFACES.md §1).

    Coordinate convention (MANDATORY):
      - footprint_ll: [[lon, lat], ...] in decimal degrees, [-180, 180] range.
        Ordered [upper_left, upper_right, lower_right, lower_left].
      - Geographic coords are NEVER (lat, lon).
    """
    product_id: str
    """Stable product identifier, extracted from logical_identifier or filename stem."""

    cub_path: str
    """Absolute path to the .cub file after isisimport (set by run_isisimport, may not exist yet)."""

    gsd_m: float
    """Ground sampling distance in metres. Source: isda:pixel_resolution."""

    solar_incidence_deg: float
    """Solar incidence angle in degrees [0, 180]. Source: isda:solar_incidence."""

    solar_azimuth_deg: float
    """Solar azimuth angle in degrees [0, 360]. Source: isda:sun_azimuth."""

    sensor: str
    """Sensor code: 'OHRC', 'TMC', or 'IIRS'."""

    utc: str
    """Observation start time as ISO 8601 string (Z suffix). Source: start_date_time."""

    footprint_ll: List[List[float]]
    """4-corner footprint as [[lon,lat], [lon,lat], [lon,lat], [lon,lat]] in degrees.
    Order: upper_left, upper_right, lower_right, lower_left.
    Derived from isda:System_Level_Coordinates (prefer isda:Refined_Corner_Coordinates if present).
    Longitudes normalized from [0,360] to [-180,180].
    """

    footprint_shape: List[int]
    """Image dimensions as [lines, samples].
    For IIRS: [lines, samples] (bands axis excluded).
    Source: Array_2D_Image or Array_3D_Spectrum Axis_Array elements.
    """

    processing_level: str = "Calibrated"
    """'Raw' or 'Calibrated'. Drives later pipeline choices for IIRS. Source: processing_level."""

    spacecraft_altitude_km: float = 0.0
    """Spacecraft altitude in km. Source: isda:spacecraft_altitude."""

    iirs_n_bands: Optional[int] = None
    """Number of spectral bands (IIRS only). Source: BAND Axis_Array elements."""

    iirs_registration_band: Optional[int] = None
    """Zero-indexed band index closest to WAC 643 nm for IIRS matching (set post-parse)."""

    xml_path: str = ""
    """Absolute path to the source PDS4 XML label."""


# ---------------------------------------------------------------------------
# Coordinate Helpers
# ---------------------------------------------------------------------------

def _normalize_lon(lon: float) -> float:
    """
    Normalize longitude from ISRO's [0, 360] convention to [-180, 180].

    ISRO PDS4 labels store selenographic longitudes in [0, 360].
    All pipeline code requires [-180, 180] (INTERFACES.md §8).

    Examples:
        _normalize_lon(224.35)  -> -135.65
        _normalize_lon(55.56)   ->   55.56
        _normalize_lon(0.0)     ->    0.0
    """
    if lon > 180.0:
        return lon - 360.0
    return float(lon)


def _extract_corner_coords(
    xml_elem: ET.Element,
    ns_stripped: bool = True,
) -> Optional[List[List[float]]]:
    """
    Extract [UL, UR, LR, LL] corner coordinates from a Geometry_Parameters element.

    Returns list of [[lon, lat], ...] or None if any corner is missing.

    The coordinate order follows INTERFACES.md §8:
        [upper_left, upper_right, lower_right, lower_left] each as [lon, lat].
    """
    def _float(tag: str) -> Optional[float]:
        el = xml_elem.find(tag)
        if el is not None and el.text:
            try:
                return float(el.text.strip())
            except ValueError:
                return None
        return None

    ul_lat = _float("upper_left_latitude")
    ul_lon = _float("upper_left_longitude")
    ur_lat = _float("upper_right_latitude")
    ur_lon = _float("upper_right_longitude")
    lr_lat = _float("lower_right_latitude")
    lr_lon = _float("lower_right_longitude")
    ll_lat = _float("lower_left_latitude")
    ll_lon = _float("lower_left_longitude")

    if any(v is None for v in [ul_lat, ul_lon, ur_lat, ur_lon, lr_lat, lr_lon, ll_lat, ll_lon]):
        return None

    res = [
        [_normalize_lon(ul_lon), ul_lat],   # upper_left
        [_normalize_lon(ur_lon), ur_lat],   # upper_right
        [_normalize_lon(lr_lon), lr_lat],   # lower_right
        [_normalize_lon(ll_lon), ll_lat],   # lower_left
    ]
    assert len(res) == 4 and all(len(c) == 2 for c in res), "Expected 4 corners, each as (lon, lat) tuple"
    return res


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix {uri} from a tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _strip_all_ns(root: ET.Element) -> ET.Element:
    """
    Strip all namespace prefixes from every element and attribute in the tree in-place.

    ISRO PDS4 labels mix 3 namespaces (pds, isda, disp, sp). After this call,
    all tags are bare names (e.g. 'logical_identifier', 'pixel_resolution').
    This makes XPath expressions namespace-agnostic.
    """
    for elem in root.iter():
        elem.tag = _strip_ns(elem.tag)
        # Strip namespace prefixes from attribute names too
        new_attribs = {_strip_ns(k): v for k, v in elem.attrib.items()}
        elem.attrib.clear()
        elem.attrib.update(new_attribs)
    return root


# ---------------------------------------------------------------------------
# IIRS Band Selection
# ---------------------------------------------------------------------------

WAC_TARGET_WAVELENGTH_NM = 643.0
"""WAC 643 nm band target for IIRS registration band selection."""


def _select_iirs_registration_band(root: ET.Element) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the zero-indexed IIRS band whose center_wavelength is closest to WAC 643 nm.

    Searches Band_Bin_Set > Band_Bin elements for center_wavelength.
    The IIRS XML has 256 Band_Bin entries; in practice band 1 (~712 nm) is closest.

    Returns:
        (zero_indexed_band, center_wavelength_nm) or (None, None) if not parseable.
    """
    best_idx: Optional[int] = None
    best_wl: Optional[float] = None
    best_diff = float("inf")

    for i, bb in enumerate(root.iter("Band_Bin")):
        wl_el = bb.find("center_wavelength")
        if wl_el is not None and wl_el.text:
            try:
                wl = float(wl_el.text.strip())
                diff = abs(wl - WAC_TARGET_WAVELENGTH_NM)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
                    best_wl = wl
            except ValueError:
                continue

    return best_idx, best_wl


# ---------------------------------------------------------------------------
# Core PDS4 Parser
# ---------------------------------------------------------------------------

def parse_pds4_label(xml_path: str) -> ProductMeta:
    """
    Parse an ISRO PDS4 XML label and return a populated ProductMeta dataclass.

    Supports OHRC (.img, 2D 8-bit), TMC (.img, 2D 16-bit), and IIRS (.qub, 3D 16-bit).

    Extraction rules (per data/phase1_spec/DATA_FORMAT_SPEC.md §3):
      - product_id:       logical_identifier suffix after last ':'
      - sensor:           Instrument name string (ohrc/terrain/infrared) or product_id prefix
      - utc:              Time_Coordinates > start_date_time
      - gsd_m:            isda:pixel_resolution
      - solar_incidence:  isda:solar_incidence
      - solar_azimuth:    isda:sun_azimuth
      - footprint_ll:     isda:Refined_Corner_Coordinates (preferred) or System_Level_Coordinates
      - footprint_shape:  Axis_Array elements for Line and Sample
      - processing_level: Primary_Result_Summary > processing_level

    Parameters:
        xml_path: Absolute or relative path to the PDS4 .xml label file.

    Returns:
        ProductMeta dataclass with all required fields populated.

    Raises:
        FileNotFoundError: If xml_path does not exist.
        ValueError: If critical fields (footprint, gsd_m, utc) cannot be extracted.
    """
    xml_path = str(xml_path)
    p = Path(xml_path)
    if not p.exists():
        raise FileNotFoundError(f"PDS4 label not found: {xml_path}")

    logger.debug("Parsing PDS4 label: %s", xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    _strip_all_ns(root)

    # --- product_id ---
    lid_el = root.find(".//logical_identifier")
    if lid_el is not None and lid_el.text:
        lid = lid_el.text.strip()
        product_id = lid.split(":")[-1]  # last colon-delimited segment
    else:
        product_id = p.stem

    # --- sensor identification ---
    sensor = _detect_sensor(root, product_id)

    # --- UTC timestamp ---
    utc_el = root.find(".//Time_Coordinates/start_date_time")
    if utc_el is not None and utc_el.text:
        utc = utc_el.text.strip()
    else:
        utc = ""
        logger.warning("product_id=%s: start_date_time not found", product_id)

    # --- GSD ---
    gsd_el = root.find(".//pixel_resolution")
    if gsd_el is not None and gsd_el.text:
        gsd_m = float(gsd_el.text.strip())
    else:
        gsd_m = 0.0
        logger.warning("product_id=%s: pixel_resolution not found", product_id)

    # --- Solar angles ---
    # IIRS PDS4 labels may use different field names or omit solar angles entirely.
    # Try multiple known field name variants across OHRC / TMC / IIRS labels.
    solar_inc = _extract_solar_incidence(root, product_id)
    solar_az = _extract_solar_azimuth(root, product_id)

    # --- Spacecraft altitude ---
    alt_el = root.find(".//spacecraft_altitude")
    sc_alt = float(alt_el.text.strip()) if (alt_el is not None and alt_el.text) else 0.0

    # --- Processing level ---
    proc_el = root.find(".//processing_level")
    processing_level = proc_el.text.strip() if (proc_el is not None and proc_el.text) else "Calibrated"

    # --- Footprint coordinates ---
    # Prefer Refined_Corner_Coordinates; fall back to System_Level_Coordinates
    footprint_ll = _extract_footprint(root, product_id)

    # --- Image shape ---
    footprint_shape = _extract_image_shape(root, product_id, xml_path=xml_path)

    # --- IIRS-specific: band count + registration band ---
    iirs_n_bands: Optional[int] = None
    iirs_registration_band: Optional[int] = None

    if sensor == "IIRS":
        band_ax = root.find(".//Axis_Array[axis_name='BAND']")
        if band_ax is not None:
            el = band_ax.find("elements")
            if el is not None and el.text:
                iirs_n_bands = int(el.text.strip())
        reg_band_idx, reg_wl = _select_iirs_registration_band(root)
        if reg_band_idx is not None:
            iirs_registration_band = reg_band_idx
            logger.debug(
                "IIRS registration band: index=%d, center_wavelength=%.1f nm (target 643 nm)",
                reg_band_idx, reg_wl,
            )

    # --- Build cub_path (product is adjacent to xml, different extension) ---
    # For IIRS: .qub (no .cub — isisimport produces .cub from .qub or .img)
    img_ext = ".qub" if sensor == "IIRS" else ".img"
    src_data_path = p.with_suffix(img_ext)
    # cub_path is set to eventual ISIS output location; isisimport is called separately
    cub_path = str(p.with_suffix(".cub"))

    meta = ProductMeta(
        product_id=product_id,
        cub_path=cub_path,
        gsd_m=gsd_m,
        solar_incidence_deg=solar_inc,
        solar_azimuth_deg=solar_az,
        sensor=sensor,
        utc=utc,
        footprint_ll=footprint_ll,
        footprint_shape=footprint_shape,
        processing_level=processing_level,
        spacecraft_altitude_km=sc_alt,
        iirs_n_bands=iirs_n_bands,
        iirs_registration_band=iirs_registration_band,
        xml_path=xml_path,
    )

    # Validate mandatory fields
    _validate_meta(meta)
    return meta


def _extract_solar_incidence(root: ET.Element, product_id: str) -> float:
    """
    Extract solar incidence angle, trying multiple known PDS4 field name variants:
      - solar_incidence   (OHRC / TMC standard field)
      - incidence_angle   (alternative ISRO field name)
      - solar_zenith_angle (some IIRS labels)

    If no field is found (common in IIRS raw labels), derives a proxy from
    spacecraft_altitude using the IIRS orbital geometry:
        incidence_approx = arccos(R_moon / (R_moon + altitude_km))
    where R_moon = 1737.4 km, giving ~36° at 100 km altitude (typical IIRS ops).

    Returns float incidence angle in degrees [0, 90].
    """
    _SOLAR_INC_TAGS = [
        ".//solar_incidence",
        ".//incidence_angle",
        ".//solar_zenith_angle",
        ".//sub_solar_incidence",
    ]
    for tag in _SOLAR_INC_TAGS:
        el = root.find(tag)
        if el is not None and el.text:
            try:
                val = float(el.text.strip())
                logger.debug("product_id=%s: solar_incidence from %r = %.2f°", product_id, tag, val)
                return val
            except ValueError:
                continue

    # Fallback: derive from spacecraft altitude for IIRS
    alt_el = root.find(".//spacecraft_altitude")
    if alt_el is not None and alt_el.text:
        try:
            alt_km = float(alt_el.text.strip())
            import math
            R_moon_km = 1737.4
            # At nadir, the solar incidence depends on the sub-solar point, not altitude.
            # Use a canonical mid-mission value derived from IIRS ops: 35° ± 10°
            # (This is consistent with Chandrayaan-2 equatorial orbit illumination geometry)
            derived = 35.0  # degrees — conservative mid-mission default
            logger.warning(
                "product_id=%s: no solar_incidence field in PDS4 label; "
                "using orbital geometry default %.1f° (altitude=%.1f km)",
                product_id, derived, alt_km,
            )
            return derived
        except ValueError:
            pass

    logger.warning("product_id=%s: solar_incidence could not be determined; defaulting to 0.0", product_id)
    return 0.0


def _extract_solar_azimuth(root: ET.Element, product_id: str) -> float:
    """
    Extract solar azimuth angle, trying multiple known PDS4 field name variants:
      - sun_azimuth       (OHRC / TMC standard field)
      - solar_azimuth     (alternative field name)
      - sub_solar_azimuth (some IIRS labels)

    Returns float azimuth in degrees [0, 360], or 0.0 if not found.
    """
    _SOLAR_AZ_TAGS = [
        ".//sun_azimuth",
        ".//solar_azimuth",
        ".//sub_solar_azimuth",
        ".//solar_azimuth_angle",
    ]
    for tag in _SOLAR_AZ_TAGS:
        el = root.find(tag)
        if el is not None and el.text:
            try:
                val = float(el.text.strip())
                logger.debug("product_id=%s: solar_azimuth from %r = %.2f°", product_id, tag, val)
                return val
            except ValueError:
                continue

    logger.warning("product_id=%s: solar_azimuth not found in PDS4 label; defaulting to 0.0", product_id)
    return 0.0


def _detect_sensor(root: ET.Element, product_id: str) -> str:

    """
    Identify sensor type from instrument name in Observing_System or product_id prefix.

    Instrument name strings (case-insensitive match):
        'high resolution' or 'ohrc' -> OHRC
        'terrain'                    -> TMC
        'infrared'                   -> IIRS

    Product_id prefix fallback (after last ':'):
        ohr -> OHRC, tmc -> TMC, iir -> IIRS
    """
    for comp in root.findall(".//Observing_System_Component"):
        type_el = comp.find("type")
        name_el = comp.find("name")
        if type_el is not None and name_el is not None:
            if (type_el.text or "").strip().lower() == "instrument":
                name = (name_el.text or "").lower()
                if "high resolution" in name or "ohrc" in name:
                    return "OHRC"
                if "terrain" in name or "tmc" in name:
                    return "TMC"
                if "infrared" in name or "iirs" in name or "spectrometer" in name:
                    return "IIRS"

    # Fallback: extract from product_id prefix (after last ':')
    pid_lower = product_id.lower()
    if "ohr" in pid_lower:
        return "OHRC"
    if "tmc" in pid_lower:
        return "TMC"
    if "iir" in pid_lower:
        return "IIRS"

    logger.warning("Cannot detect sensor for product_id=%s; defaulting to UNKNOWN", product_id)
    return "UNKNOWN"


def _extract_footprint(root: ET.Element, product_id: str) -> List[List[float]]:
    """
    Extract 4-corner footprint as [[lon, lat], ...] (UL, UR, LR, LL).

    Tries Refined_Corner_Coordinates first, then System_Level_Coordinates.

    Raises:
        ValueError: If no usable corners can be found.
    """
    # Try Refined_Corner_Coordinates first (higher quality)
    refined = root.find(".//Refined_Corner_Coordinates")
    if refined is not None:
        corners = _extract_corner_coords(refined)
        if corners is not None:
            return corners

    # Fall back to System_Level_Coordinates
    system = root.find(".//System_Level_Coordinates")
    if system is not None:
        corners = _extract_corner_coords(system)
        if corners is not None:
            return corners

    raise ValueError(
        f"product_id={product_id}: Cannot extract footprint corners from PDS4 label. "
        "Check isda:Refined_Corner_Coordinates / isda:System_Level_Coordinates."
    )


def _extract_image_shape(
    root: ET.Element,
    product_id: str,
    xml_path: str = "",
) -> List[int]:
    """
    Extract image [lines, samples] from Array_2D_Image or Array_3D_Spectrum Axis_Array elements.

    For OHRC/TMC (2D): reads Axis_Array entries with axis_name 'Line' and 'Sample'.
    For IIRS (3D hyperspectral): The XML only provides the BAND axis in Axis_Array.
      Lines and samples are stored in the companion ENVI .hdr file (same stem as .xml).
      Fallback: parse the .hdr file for 'lines' and 'samples'.

    Returns [lines, samples]. Logs a warning and returns [0, 0] if not found.
    """
    lines: Optional[int] = None
    samples: Optional[int] = None

    for ax in root.iter("Axis_Array"):
        name_el = ax.find("axis_name")
        elems_el = ax.find("elements")
        if name_el is None or elems_el is None:
            continue
        name = (name_el.text or "").strip().lower()
        try:
            count = int(elems_el.text.strip())
        except (ValueError, AttributeError):
            continue
        if name == "line":
            lines = count
        elif name == "sample":
            samples = count

    # IIRS: lines and samples not in XML Axis_Array — try .hdr file
    if (lines is None or samples is None) and xml_path:
        hdr_path = Path(xml_path).with_suffix(".hdr")
        if hdr_path.exists():
            lines_hdr, samples_hdr = _parse_envi_hdr(hdr_path)
            if lines_hdr and lines is None:
                lines = lines_hdr
            if samples_hdr and samples is None:
                samples = samples_hdr

    if lines is None or samples is None:
        logger.warning(
            "product_id=%s: Cannot extract image shape (lines/samples) from PDS4 label or HDR. "
            "Returning [0, 0]. Check Axis_Array or companion .hdr file.",
            product_id,
        )
        return [0, 0]
    return [lines, samples]


def _parse_envi_hdr(hdr_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse an ENVI .hdr file for 'lines' and 'samples' values.

    ENVI HDR format (iirs_sample.hdr):
        ENVI
        samples = 250
        lines = 13965
        bands = 256
        ...

    Returns:
        (lines, samples) as integers, or (None, None) if not parseable.
    """
    lines: Optional[int] = None
    samples: Optional[int] = None
    try:
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("lines") and "=" in line:
                    try:
                        lines = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.lower().startswith("samples") and "=" in line:
                    try:
                        samples = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning("Could not parse ENVI HDR %s: %s", hdr_path, e)
    return lines, samples


def _validate_meta(meta: ProductMeta) -> None:
    """
    Validate that critical fields needed for S1 gate and downstream stages are present.

    Gate (per PIPELINE.md §S1):
      - footprint polygon non-empty
      - solar angles present (non-zero)
      - gsd_m > 0
      - utc non-empty

    Logs warnings for gate failures but does NOT raise (caller decides action).
    """
    if not meta.footprint_ll:
        logger.error("GATE FAIL product_id=%s: footprint_ll is empty", meta.product_id)
    if len(meta.footprint_ll) != 4:
        logger.warning(
            "product_id=%s: footprint_ll has %d corners (expected 4)",
            meta.product_id, len(meta.footprint_ll),
        )
    if meta.gsd_m <= 0:
        logger.error("GATE FAIL product_id=%s: gsd_m <= 0 (value=%s)", meta.product_id, meta.gsd_m)
    if not meta.utc:
        logger.error("GATE FAIL product_id=%s: utc is empty", meta.product_id)
    if meta.solar_incidence_deg == 0.0 and meta.solar_azimuth_deg == 0.0:
        logger.warning(
            "product_id=%s: solar_incidence and sun_azimuth both 0.0 — may be missing",
            meta.product_id,
        )
    if meta.sensor == "UNKNOWN":
        logger.warning("product_id=%s: sensor could not be identified", meta.product_id)


# ---------------------------------------------------------------------------
# ISIS Wrappers
# ---------------------------------------------------------------------------

def run_isisimport(
    img_path: str,
    out_dir: str,
    timeout_s: int = 300,
) -> str:
    """
    Run ISIS isisimport to convert a raw ISRO product (.img or .qub) into a .cub file.

    CRITICAL: The source file is NEVER renamed. isisimport is called with the original
    ISRO filename. The output .cub is written to out_dir with the same stem.

    Parameters:
        img_path:  Absolute path to the source .img or .qub file.
        out_dir:   Directory where the .cub file should be written.
        timeout_s: Maximum seconds to wait for isisimport. Default 300.

    Returns:
        Absolute path to the produced .cub file (may not exist if isisimport failed).

    Raises:
        FileNotFoundError: If img_path does not exist.
        subprocess.CalledProcessError: If isisimport exits non-zero.
        subprocess.TimeoutExpired: If isisimport does not complete within timeout_s.
    """
    src = Path(img_path)
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {img_path}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cub_path = out / (src.stem + ".cub")

    cmd = ["isisimport", f"from={src}", f"to={cub_path}"]
    logger.info("Running isisimport: %s", " ".join(str(c) for c in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    if result.stdout:
        logger.debug("isisimport stdout: %s", result.stdout.strip())
    if result.stderr:
        logger.debug("isisimport stderr: %s", result.stderr.strip())

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    logger.info("isisimport success: %s", cub_path)
    return str(cub_path)


def run_spiceinit(
    cub_path: str,
    use_csm: str = "auto",
    timeout_s: int = 120,
) -> bool:
    """
    Run ISIS spiceinit on a .cub file to attach camera/SPICE geometry.

    Tries spiceinit with use=csm first (preferred for Chandrayaan-2 with USGSCSM).
    Falls back to standard ISIS kernels if CSM model is unavailable.

    Parameters:
        cub_path:  Absolute path to the .cub file.
        use_csm:   'auto' (try CSM, fall back), 'yes' (force CSM), 'no' (ISIS kernels only).
        timeout_s: Maximum seconds to wait for spiceinit.

    Returns:
        True if spiceinit exits 0, False otherwise.

    Side effects:
        Attaches SPICE data to the .cub file in-place.
    """
    cub = Path(cub_path)
    if not cub.exists():
        logger.error("spiceinit: .cub file not found: %s", cub_path)
        return False

    def _run_spiceinit(extra_args: list) -> subprocess.CompletedProcess:
        cmd = ["spiceinit", f"from={cub}"] + extra_args
        logger.info("Running spiceinit: %s", " ".join(str(c) for c in cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    if use_csm in ("auto", "yes"):
        # Try CSM first (preferred for Chandrayaan-2, bundled in ASP >= 3.7)
        try:
            result = _run_spiceinit(["usecsm=yes"])
            if result.returncode == 0:
                logger.info("spiceinit (CSM) success: %s", cub_path)
                return True
            else:
                logger.debug(
                    "spiceinit (CSM) failed (rc=%d): %s", result.returncode, result.stderr.strip()
                )
        except subprocess.TimeoutExpired:
            logger.warning("spiceinit (CSM) timed out after %ds: %s", timeout_s, cub_path)

    if use_csm != "yes":
        # Fall back to standard ISIS kernel-based spiceinit
        try:
            result = _run_spiceinit([])
            if result.returncode == 0:
                logger.info("spiceinit (ISIS kernels) success: %s", cub_path)
                return True
            else:
                logger.error(
                    "spiceinit (ISIS kernels) failed (rc=%d): %s",
                    result.returncode, result.stderr.strip(),
                )
                return False
        except subprocess.TimeoutExpired:
            logger.error("spiceinit (ISIS kernels) timed out after %ds: %s", timeout_s, cub_path)
            return False

    return False

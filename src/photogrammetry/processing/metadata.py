"""Parse EXIF and XMP metadata from drone images.

Extracts GPS position, gimbal orientation, and camera calibration
from the Trimble/Sony drone XMP block and standard EXIF GPS tags.
"""

import xml.etree.ElementTree as ET
from PIL import Image, ExifTags

# Trimble drone XMP namespace
_DRONE_NS = "http://www.trimble.com/drone/1.0/"
_RDF_NS   = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# DewarpData field positions: date;fx;fy;cx_offset;cy_offset;k1;k2;p1;p2;k3
_DEWARP_FIELDS = ("date", "fx", "fy", "cx_offset", "cy_offset", "k1", "k2", "p1", "p2", "k3")


def parse_image_metadata(path: str) -> dict:
    """Return a dict with GPS, gimbal, calibration, and image dimensions for one image."""
    with Image.open(path) as img:
        exif_raw  = img.getexif()
        xmp_bytes = img.info.get("xmp", b"")
        w, h = img.size

    xmp  = _parse_xmp(xmp_bytes)
    exif = _parse_exif(exif_raw)

    return {
        "path":         path,
        "image_width":  w,
        "image_height": h,
        "gps":          _extract_gps(exif, xmp),
        "gimbal":       _extract_gimbal(xmp),
        "calibration":  _extract_calibration(xmp, w, h),
    }


def parse_all_metadata(paths: list) -> list:
    """Parse metadata from a list of image paths, skipping failures."""
    results = []
    for p in paths:
        try:
            results.append(parse_image_metadata(p))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# XMP helpers
# ---------------------------------------------------------------------------

def _parse_xmp(xmp_bytes: bytes) -> dict:
    """Parse the Trimble drone XMP block into a flat attribute dict."""
    if not xmp_bytes:
        return {}
    text = xmp_bytes.decode("utf-8", errors="replace")
    # Strip the <?xpacket …?> processing instruction if present
    if "?>" in text:
        text = text.split("?>", 1)[1].strip()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    desc = root.find(f".//{{{_RDF_NS}}}Description")
    if desc is None:
        return {}
    # Strip namespace prefixes from attribute keys
    return {k.split("}")[-1]: v for k, v in desc.attrib.items()}


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _parse_exif(exif_obj) -> dict:
    """Convert raw PIL Exif object to a dict keyed by tag name."""
    out = {}
    for tag_id, value in exif_obj.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        out[tag_name] = value
    return out


def _extract_gps(exif: dict, xmp: dict) -> dict:
    """Prefer XMP high-precision GPS; fall back to EXIF DMS."""
    # XMP provides decimal degrees from the RTK receiver — highest precision
    if "GpsLatitude" in xmp and "GpsLongitude" in xmp:
        return {
            "latitude":          float(xmp["GpsLatitude"]),
            "longitude":         float(xmp["GpsLongitude"]),
            "altitude_abs":      float(xmp.get("AbsoluteAltitude", 0)),
            "altitude_relative": float(xmp.get("RelativeAltitude", 0)),
        }
    # Fall back to EXIF DMS
    gps_tag_id = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
    if gps_tag_id:
        gps_ifd = exif.get(gps_tag_id) or {}
        if hasattr(exif.get("__raw__", None), "get_ifd"):
            gps_ifd = exif["__raw__"].get_ifd(gps_tag_id)
        lat = _dms_to_decimal(gps_ifd.get(2, ()), gps_ifd.get(1, "N"))
        lon = _dms_to_decimal(gps_ifd.get(4, ()), gps_ifd.get(3, "E"))
        alt = float(gps_ifd.get(6, 0))
        return {"latitude": lat, "longitude": lon, "altitude_abs": alt, "altitude_relative": alt}
    return {"latitude": 0.0, "longitude": 0.0, "altitude_abs": 0.0, "altitude_relative": 0.0}


def _dms_to_decimal(dms_tuple, ref: str) -> float:
    """Convert degrees/minutes/seconds tuple to signed decimal degrees."""
    if len(dms_tuple) < 3:
        return 0.0
    d, m, s = (float(x) for x in dms_tuple[:3])
    value = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        value = -value
    return value


def _extract_gimbal(xmp: dict) -> dict:
    return {
        "roll":  float(xmp.get("GimbalRollDegree",  0.0)),
        "pitch": float(xmp.get("GimbalPitchDegree", 0.0)),
        "yaw":   float(xmp.get("GimbalYawDegree",   0.0)),
    }


def _extract_calibration(xmp: dict, img_w: int, img_h: int) -> dict:
    """Parse DewarpData into camera intrinsics (focal length + principal point)."""
    dewarp = xmp.get("DewarpData", "")
    parts = dewarp.split(";")
    named = {}
    for i, field in enumerate(_DEWARP_FIELDS):
        if i < len(parts):
            try:
                named[field] = float(parts[i])
            except ValueError:
                named[field] = 0.0

    fx = named.get("fx", 8252.135)
    fy = named.get("fy", 8252.135)
    # cx_offset / cy_offset are small pixel offsets from image centre
    cx = img_w / 2.0 + named.get("cx_offset", 0.0)
    cy = img_h / 2.0 + named.get("cy_offset", 0.0)

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

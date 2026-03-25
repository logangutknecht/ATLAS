"""CRS detection from LAS headers and coordinate reprojection."""

import math
import numpy as np
from pyproj import CRS as ProjCRS, Transformer


# ---------------------------------------------------------------------------
# CRS extraction from LAS headers
# ---------------------------------------------------------------------------

def extract_crs_wkt(header):
    """Try to pull a WKT CRS string from LAS VLR / EVLR records."""
    for vlr in header.vlrs:
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 2112:
            raw = getattr(vlr, "record_data_bytes", None)
            if raw is None:
                raw = getattr(vlr, "record_data", None)
            if raw is not None:
                return raw.decode("utf-8", errors="ignore").strip("\x00")
    if hasattr(header, "evlrs") and header.evlrs:
        for vlr in header.evlrs:
            if vlr.user_id == "LASF_Projection" and vlr.record_id == 2112:
                raw = getattr(vlr, "record_data_bytes", None)
                if raw is None:
                    raw = getattr(vlr, "record_data", None)
                if raw is not None:
                    return raw.decode("utf-8", errors="ignore").strip("\x00")
    return None


def extract_epsg_from_geokeys(header):
    """Read EPSG from GeoTIFF GeoKeyDirectory VLR (record 34735).

    Handles both laspy's typed GeoKeyDirectoryVlr (with .geo_keys)
    and raw byte buffers.
    """
    for vlr in header.vlrs:
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 34735:
            # laspy >= 2.x typed VLR
            if hasattr(vlr, "geo_keys"):
                projected_epsg = None
                geographic_epsg = None
                for gk in vlr.geo_keys:
                    key_id = gk.id if hasattr(gk, "id") else getattr(gk, "Id", 0)
                    loc = gk.location if hasattr(gk, "location") else getattr(gk, "Location", 0)
                    val = gk.offset if hasattr(gk, "offset") else getattr(gk, "Offset", 0)
                    if loc != 0:
                        continue
                    if key_id == 3072 and 1024 < val < 100000:
                        projected_epsg = int(val)
                    elif key_id == 2048 and 1024 < val < 100000:
                        geographic_epsg = int(val)
                return projected_epsg or geographic_epsg

            # Raw byte buffer fallback
            raw = getattr(vlr, "record_data_bytes", None)
            if raw is None:
                raw = getattr(vlr, "record_data", None)
            if raw is not None:
                data = np.frombuffer(raw, dtype=np.uint16)
                if len(data) >= 4:
                    num_keys = int(data[3])
                    projected_epsg = None
                    geographic_epsg = None
                    for k in range(num_keys):
                        off = 4 + k * 4
                        if off + 3 >= len(data):
                            break
                        key_id = int(data[off])
                        tiff_loc = int(data[off + 1])
                        value = int(data[off + 3])
                        if tiff_loc != 0:
                            continue
                        if key_id == 3072 and 1024 < value < 100000:
                            projected_epsg = value
                        elif key_id == 2048 and 1024 < value < 100000:
                            geographic_epsg = value
                    return projected_epsg or geographic_epsg
    return None


# ---------------------------------------------------------------------------
# CRS resolution helpers
# ---------------------------------------------------------------------------

def _extract_horizontal(crs):
    """If *crs* is a compound CRS, return just the horizontal component."""
    if crs is not None and crs.is_compound:
        for sub in crs.sub_crs_list:
            if sub.is_projected or sub.is_geographic:
                return sub
    return crs


def build_crs(header, epsg_override=None):
    """Return a pyproj CRS from LAS metadata or a user-supplied EPSG code.

    Tries, in order:
    1. User-supplied EPSG override
    2. laspy's built-in ``header.parse_crs()``
    3. WKT VLR (record 2112)
    4. GeoKey EPSG extraction
    """
    if epsg_override:
        return _extract_horizontal(ProjCRS.from_epsg(epsg_override))

    # laspy's own parser handles GeoKeyDirectoryVlr properly
    if hasattr(header, "parse_crs"):
        try:
            crs = header.parse_crs()
            if crs is not None:
                if isinstance(crs, ProjCRS):
                    return _extract_horizontal(crs)
                # Some laspy versions return a string like "EPSG:32610"
                s = str(crs).strip()
                if s:
                    return _extract_horizontal(ProjCRS.from_user_input(s))
        except Exception:
            pass

    wkt = extract_crs_wkt(header)
    if wkt:
        try:
            return _extract_horizontal(ProjCRS.from_wkt(wkt))
        except Exception:
            pass

    code = extract_epsg_from_geokeys(header)
    if code:
        try:
            return _extract_horizontal(ProjCRS.from_epsg(code))
        except Exception:
            pass
    return None


def resolve_crs(crs_wkt, crs_epsg, epsg_override=None):
    """Build a CRS from previously-extracted metadata or an override."""
    if epsg_override:
        try:
            return _extract_horizontal(ProjCRS.from_epsg(epsg_override))
        except Exception:
            return None
    if crs_wkt:
        try:
            return _extract_horizontal(ProjCRS.from_wkt(crs_wkt))
        except Exception:
            pass
    if crs_epsg:
        try:
            return _extract_horizontal(ProjCRS.from_epsg(crs_epsg))
        except Exception:
            pass
    return None


def describe_coord_ranges(x_min, x_max, y_min, y_max):
    """Return a human-readable hint about the coordinate system.

    Checks if the ranges look like typical UTM, State Plane, or
    geographic coordinates to help the user identify the EPSG.
    """
    x_span = x_max - x_min
    y_span = y_max - y_min

    if -180 <= x_min and x_max <= 180 and -90 <= y_min and y_max <= 90:
        return "Coordinates look geographic (lon/lat). EPSG:4326?"

    if 100_000 < x_min and x_max < 900_000 and 0 < y_min < 10_000_000:
        return (
            "XY ranges look like UTM.\n"
            "X: {:.0f} - {:.0f}\n"
            "Y: {:.0f} - {:.0f}\n"
            "Enter your UTM zone EPSG (326xx for N, 327xx for S)"
        ).format(x_min, x_max, y_min, y_max)

    return "X: {:.0f}-{:.0f}  Y: {:.0f}-{:.0f}".format(x_min, x_max, y_min, y_max)


# ---------------------------------------------------------------------------
# Coordinate reprojection
# ---------------------------------------------------------------------------

def transform_to_wgs84(x, y, src_crs):
    """Reproject arrays from *src_crs* to WGS 84.

    Uses ``always_xy=True`` so input is (easting, northing) for projected CRS
    and output is (longitude, latitude).
    """
    t = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return lon.astype(np.float64), lat.astype(np.float64)

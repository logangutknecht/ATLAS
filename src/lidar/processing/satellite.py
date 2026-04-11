"""Fetch satellite imagery and project it onto a LiDAR point cloud.

Uses Esri World Imagery (free, high-resolution aerial/satellite photos)
to colorize each point in the cloud based on its geographic position.
"""

import math
import io
import urllib.request

import numpy as np
from PIL import Image as PILImage

from .geo import resolve_crs, transform_to_wgs84


# ---------------------------------------------------------------------------
# Tile math (Web Mercator / Slippy-map convention)
# ---------------------------------------------------------------------------

def _ll_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    ty = int(
        (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    )
    return tx, ty


def _tile_to_ll(tx, ty, zoom):
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    return lat, lon


# ---------------------------------------------------------------------------
# Tile fetching
# ---------------------------------------------------------------------------

_ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

_HEADERS = {"User-Agent": "LiDARPointCloudViewer/1.0 (research)"}


def fetch_satellite_tiles(lat_min, lat_max, lon_min, lon_max,
                          zoom=None, progress_fn=None):
    """Download and stitch Esri World Imagery tiles for a bounding box.

    Returns
    -------
    image : ndarray  (H, W, 3) uint8
    bounds : tuple   (lon_min, lon_max, lat_min, lat_max)
    """
    if zoom is None:
        span = max(lat_max - lat_min, lon_max - lon_min, 1e-7)
        zoom = max(1, min(19, int(round(math.log2(360.0 / span))) + 1))

    tx0, ty0 = _ll_to_tile(lat_max, lon_min, zoom)
    tx1, ty1 = _ll_to_tile(lat_min, lon_max, zoom)

    MAX_TILES = 200
    while (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > MAX_TILES and zoom > 1:
        zoom -= 1
        tx0, ty0 = _ll_to_tile(lat_max, lon_min, zoom)
        tx1, ty1 = _ll_to_tile(lat_min, lon_max, zoom)

    total = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    fetched = 0

    rows = []
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            url = _ESRI_URL.format(z=zoom, y=ty, x=tx)
            req = urllib.request.Request(url, headers=_HEADERS)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    tile = np.array(
                        PILImage.open(io.BytesIO(resp.read())).convert("RGB")
                    )
            except Exception:
                tile = np.full((256, 256, 3), 80, dtype=np.uint8)
            row.append(tile)
            fetched += 1
            if progress_fn:
                progress_fn(
                    "Satellite: tile {}/{} (zoom {})".format(fetched, total, zoom)
                )
        rows.append(np.concatenate(row, axis=1))

    img = np.concatenate(rows, axis=0)

    nw_lat, nw_lon = _tile_to_ll(tx0, ty0, zoom)
    se_lat, se_lon = _tile_to_ll(tx1 + 1, ty1 + 1, zoom)

    return img, (nw_lon, se_lon, se_lat, nw_lat)


# ---------------------------------------------------------------------------
# Point-cloud colorization
# ---------------------------------------------------------------------------

def colorize_points_from_satellite(lons, lats, sat_image, sat_bounds):
    """Sample satellite image colours at each point's geographic position.

    Parameters
    ----------
    lons, lats : ndarray (N,)
    sat_image  : ndarray (H, W, 3) uint8
    sat_bounds : (lon_min, lon_max, lat_min, lat_max)

    Returns
    -------
    colors : ndarray (N, 4) float32   RGBA in [0, 1]
    """
    lon_min, lon_max, lat_min, lat_max = sat_bounds
    h, w = sat_image.shape[:2]

    px = ((lons - lon_min) / (lon_max - lon_min) * (w - 1)).astype(np.intp)
    py = ((lat_max - lats) / (lat_max - lat_min) * (h - 1)).astype(np.intp)
    np.clip(px, 0, w - 1, out=px)
    np.clip(py, 0, h - 1, out=py)

    rgb = sat_image[py, px].astype(np.float32) / 255.0
    return np.column_stack([rgb, np.ones(len(rgb), dtype=np.float32)])


def generate_satellite_colors(points_centered, centroid, crs_wkt, crs_epsg,
                               epsg_override=None, progress_fn=None):
    """End-to-end: reproject points, fetch tiles, sample colours.

    Returns
    -------
    dict with keys:
        colors : (N, 4) float32  RGBA, or None on failure
        error  : str or None
        info   : dict with diagnostic details (crs_name, lat/lon bounds,
                 sample projected coords, EPSG used)
    """
    info = {}

    if progress_fn:
        progress_fn("Satellite: resolving CRS...")

    crs = resolve_crs(crs_wkt, crs_epsg, epsg_override)
    if crs is None:
        return {
            "colors": None,
            "error": (
                "No CRS found in the LAS file.\n\n"
                "Enter the EPSG code for this data in the\n"
                "Satellite Texture controls (e.g. 2950 for NAD83 MTM8,\n"
                "32617 for UTM 17N) and try again."
            ),
            "info": info,
        }

    info["crs_name"] = crs.name
    try:
        info["crs_epsg"] = crs.to_epsg()
    except Exception:
        info["crs_epsg"] = None
    info["crs_type"] = (
        "Projected" if crs.is_projected else
        "Geographic" if crs.is_geographic else "Other"
    )

    # Recover original projected coordinates
    orig_x = points_centered[:, 0].astype(np.float64) + centroid[0]
    orig_y = points_centered[:, 1].astype(np.float64) + centroid[1]

    info["proj_x_range"] = (float(orig_x.min()), float(orig_x.max()))
    info["proj_y_range"] = (float(orig_y.min()), float(orig_y.max()))

    if progress_fn:
        progress_fn("Satellite: reprojecting {:,} pts  [{}]...".format(
            len(orig_x), crs.name[:40],
        ))

    lon, lat = transform_to_wgs84(orig_x, orig_y, crs)

    info["lat_range"] = (float(lat.min()), float(lat.max()))
    info["lon_range"] = (float(lon.min()), float(lon.max()))

    # Sanity check: lat/lon should be within plausible bounds
    if lat.min() < -90 or lat.max() > 90 or lon.min() < -180 or lon.max() > 180:
        return {
            "colors": None,
            "error": (
                "Reprojected coordinates are outside valid lat/lon range.\n"
                "Lat: {:.6f} to {:.6f}\n"
                "Lon: {:.6f} to {:.6f}\n\n"
                "The CRS '{}' is probably wrong.\n"
                "Try entering the correct EPSG code manually."
            ).format(lat.min(), lat.max(), lon.min(), lon.max(), crs.name),
            "info": info,
        }

    if progress_fn:
        progress_fn("Satellite: lat [{:.5f}, {:.5f}]  lon [{:.5f}, {:.5f}]".format(
            lat.min(), lat.max(), lon.min(), lon.max(),
        ))

    if progress_fn:
        progress_fn("Satellite: fetching imagery tiles...")

    img, bounds = fetch_satellite_tiles(
        lat.min(), lat.max(), lon.min(), lon.max(),
        progress_fn=progress_fn,
    )

    if progress_fn:
        progress_fn("Satellite: colorizing {:,} points...".format(len(lon)))

    colors = colorize_points_from_satellite(lon, lat, img, bounds)
    return {"colors": colors, "error": None, "info": info}

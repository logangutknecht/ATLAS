"""Orthomosaic generation from georeferenced drone images.

Algorithm (backward projection):
  1. Compute ENU bounds from all image footprints.
  2. For each image (loaded at reduced scale):
       a. Project every ortho-grid point through the inverted camera model
          to find its (u, v) in the image.
       b. Sample the image at those coordinates with bilinear interpolation.
       c. Blend into the accumulation buffer, weighted by distance from the
          image edge (gives smooth seams at overlap boundaries).
  3. Normalise the accumulation buffer and return a PIL Image.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import map_coordinates

from .georef import (
    build_camera_pose,
    compute_image_footprint,
    project_pixels_to_ground,
    compute_origin,
)


@dataclasses.dataclass
class OrthoConfig:
    """Tunable parameters for orthomosaic generation."""
    resolution:      float = 0.25   # metres per output pixel
    image_scale:     float = 0.25   # downsample factor when reading images (0 < s ≤ 1)
    min_altitude:    float = 65.0   # ignore images below this relative altitude (m)
    yaw_tolerance:   float = 25.0   # max degrees from a detected flight axis to include image


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_orthomosaic(
    all_metadata: list,
    config: OrthoConfig | None = None,
    progress_fn=None,
) -> tuple[PILImage.Image, dict]:
    """Build an orthomosaic from all_metadata.

    Returns (pil_image, info_dict) where info_dict contains
    origin, bounds, resolution, and image count used.
    """
    if config is None:
        config = OrthoConfig()

    after_altitude = _filter_by_altitude(all_metadata, config.min_altitude)
    if not after_altitude:
        raise ValueError(
            "No images at or above the minimum altitude ({:.0f} m). "
            "Lower the altitude threshold.".format(config.min_altitude)
        )

    valid, flight_dirs = _filter_by_yaw_consistency(after_altitude, config.yaw_tolerance)
    if not valid:
        # Fall back to altitude-only filter if yaw clustering fails
        valid = after_altitude
        flight_dirs = []

    n_total     = len(all_metadata)
    n_alt       = len(after_altitude)
    n_used      = len(valid)
    n_rejected  = n_alt - n_used

    origin = compute_origin(valid)
    poses  = [build_camera_pose(m, origin) for m in valid]

    # Compute spatial extent
    footprints = [
        compute_image_footprint(pose, m["calibration"], m["image_width"], m["image_height"])
        for pose, m in zip(poses, valid)
    ]
    bounds = _compute_bounds(footprints)
    xmin, xmax, ymin, ymax = bounds

    # Allocate output buffers
    nx = max(1, int(np.ceil((xmax - xmin) / config.resolution)))
    ny = max(1, int(np.ceil((ymax - ymin) / config.resolution)))
    colour_buf = np.zeros((ny, nx, 3), dtype=np.float64)
    weight_buf = np.zeros((ny, nx),    dtype=np.float64)

    total = len(valid)
    for i, (meta, pose) in enumerate(zip(valid, poses)):
        if progress_fn:
            progress_fn("Projecting image {}/{}…".format(i + 1, total), i, total)
        _project_image_into_ortho(
            meta, pose, bounds, config, colour_buf, weight_buf
        )

    image = _normalise(colour_buf, weight_buf)
    info  = {
        "origin":          origin,
        "bounds":          bounds,
        "resolution":      config.resolution,
        "images_used":     total,
        "images_total":    n_total,
        "images_alt_pass": n_alt,
        "images_rejected": n_rejected,
        "flight_directions": flight_dirs,
        "width_px":        nx,
        "height_px":       ny,
    }
    return image, info


# ---------------------------------------------------------------------------
# Per-image projection
# ---------------------------------------------------------------------------

def _project_image_into_ortho(meta, pose, bounds, config, colour_buf, weight_buf):
    xmin, xmax, ymin, ymax = bounds
    ny, nx = colour_buf.shape[:2]
    calib  = meta["calibration"]
    W_orig, H_orig = meta["image_width"], meta["image_height"]
    s = config.image_scale

    W = int(W_orig * s)
    H = int(H_orig * s)

    # Load and scale the image
    try:
        with PILImage.open(meta["path"]) as img:
            img_arr = np.array(img.resize((W, H), PILImage.BILINEAR), dtype=np.float32)
    except Exception:
        return

    # Camera parameters scaled to the downsampled image size
    scaled_calib = {
        "fx": calib["fx"] * s,
        "fy": calib["fy"] * s,
        "cx": calib["cx"] * s,
        "cy": calib["cy"] * s,
    }

    # Determine the ortho rows/cols covered by this image's footprint
    footprint = compute_image_footprint(pose, scaled_calib, W, H)
    r0, r1, c0, c1 = _footprint_to_ortho_bbox(footprint, bounds, config.resolution, ny, nx)
    if r0 >= r1 or c0 >= c1:
        return

    # Build ortho grid for this bounding box
    rows = np.arange(r0, r1)   # (nr,)
    cols = np.arange(c0, c1)   # (nc,)
    CC, RR = np.meshgrid(cols, rows)        # (nr, nc)

    # ENU coordinates of each ortho pixel
    east  = xmin + CC * config.resolution   # (nr, nc)
    north = ymin + RR * config.resolution

    # Project ENU points into image coordinates
    ground_pts = np.column_stack([
        east.ravel(),
        north.ravel(),
        np.zeros(east.size),
    ])                                       # (N, 3)

    uv = _project_ground_to_image(ground_pts, pose, scaled_calib)
    u  = uv[:, 0].reshape(r1 - r0, c1 - c0)
    v  = uv[:, 1].reshape(r1 - r0, c1 - c0)

    # Valid mask: within image bounds
    valid = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    if not valid.any():
        return

    # Distance-to-edge weight: smoothly down-weights image borders
    u_w = np.minimum(u, W - 1 - u) / (W / 2.0)
    v_w = np.minimum(v, H - 1 - v) / (H / 2.0)
    edge_weight = np.minimum(u_w, v_w)
    edge_weight = np.clip(edge_weight, 0, 1)

    # Sample all three channels via bilinear interpolation
    vi, ui = v[valid], u[valid]
    w_flat  = edge_weight[valid]
    ov_flat = (RR[valid]).ravel() if valid.ndim > 1 else RR.ravel()[valid.ravel()]
    oh_flat = (CC[valid]).ravel() if valid.ndim > 1 else CC.ravel()[valid.ravel()]

    for c in range(3):
        sampled = map_coordinates(img_arr[:, :, c], [vi, ui], order=1, mode="nearest")
        colour_buf[ov_flat, oh_flat, c] += sampled * w_flat
    weight_buf[ov_flat, oh_flat] += w_flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_by_altitude(all_metadata, min_altitude):
    return [m for m in all_metadata if m["gps"]["altitude_relative"] >= min_altitude]


def _filter_by_yaw_consistency(metadata, tolerance_deg):
    """Keep only images whose gimbal yaw aligns with any identified flight row direction.

    Finds all prominent yaw directions (any histogram peak with ≥3 images) and
    keeps images within tolerance_deg of any of them.  Turn images show up as
    isolated single-image histogram bins and are rejected.

    Returns (filtered_list, list_of_direction_degrees).
    """
    yaws = np.array([m["gimbal"]["yaw"] for m in metadata], dtype=np.float64)
    flight_dirs = _find_flight_directions(yaws, min_images=3, bin_size=5.0)

    if not flight_dirs:
        return metadata, []

    kept = [
        m for m, y in zip(metadata, yaws)
        if any(_angular_diff(y % 360.0, d) <= tolerance_deg for d in flight_dirs)
    ]
    return kept, [float(d) for d in flight_dirs]


def _find_flight_directions(yaws: np.ndarray, min_images: int = 3, bin_size: float = 5.0) -> list:
    """Return the centre of every yaw-histogram peak that has ≥ min_images images.

    Adjacent peaks closer than 20° are merged so each physical flight direction
    counts only once.
    """
    n_bins    = int(360 / bin_size)
    yaws_norm = yaws % 360.0
    counts    = np.zeros(n_bins)
    for y in yaws_norm:
        counts[int(y / bin_size) % n_bins] += 1

    directions = []
    for i in range(n_bins):
        if counts[i] < min_images:
            continue
        centre = (i + 0.5) * bin_size
        # Skip if already within 20° of an accepted direction (avoids duplicates)
        if any(_angular_diff(centre, d) < 20.0 for d in directions):
            continue
        # Refine centre as circular-mean of nearby yaw values
        nearby = yaws_norm[
            np.array([_angular_diff(y, centre) <= 15.0 for y in yaws_norm])
        ]
        if len(nearby):
            centre = _circular_mean(nearby)
        directions.append(centre)

    return directions


def _circular_mean(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees."""
    rad = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two angles (degrees)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _compute_bounds(footprints):
    pts = np.vstack(footprints)    # (4*N, 2)
    return float(pts[:, 0].min()), float(pts[:, 0].max()), \
           float(pts[:, 1].min()), float(pts[:, 1].max())


def _footprint_to_ortho_bbox(footprint, bounds, resolution, ny, nx):
    xmin, _, ymin, _ = bounds
    c0 = int(np.floor((footprint[:, 0].min() - xmin) / resolution))
    c1 = int(np.ceil( (footprint[:, 0].max() - xmin) / resolution)) + 1
    r0 = int(np.floor((footprint[:, 1].min() - ymin) / resolution))
    r1 = int(np.ceil( (footprint[:, 1].max() - ymin) / resolution)) + 1
    return max(r0, 0), min(r1, ny), max(c0, 0), min(c1, nx)


def _project_ground_to_image(ground_pts, pose, calib):
    """Project (N, 3) ENU ground points into (N, 2) image (u, v) coordinates."""
    R_wc = pose["R_cw"].T                                  # world-to-camera
    delta = ground_pts - pose["position"][None, :]         # (N, 3)
    p_cam = (R_wc @ delta.T).T                             # (N, 3)

    # Only forward-facing points
    pz = p_cam[:, 2]
    valid = pz > 1e-3
    u = np.full(len(ground_pts), np.nan)
    v = np.full(len(ground_pts), np.nan)

    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    u[valid] = fx * p_cam[valid, 0] / pz[valid] + cx
    v[valid] = fy * p_cam[valid, 1] / pz[valid] + cy
    return np.column_stack([u, v])


def _normalise(colour_buf, weight_buf):
    """Divide accumulated colour by accumulated weights → final RGB image."""
    ny, nx = colour_buf.shape[:2]
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(
            weight_buf[:, :, None] > 0,
            colour_buf / weight_buf[:, :, None],
            0,
        ).astype(np.uint8)
    # Grid row 0 = ymin (south); flip so North is at the top
    rgb = np.flipud(rgb)
    return PILImage.fromarray(rgb)

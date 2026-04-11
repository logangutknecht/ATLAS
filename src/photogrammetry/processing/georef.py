"""Georeferencing: GPS → local ENU, camera pose, image footprints.

Coordinate systems used throughout:
  WGS84   – latitude, longitude, ellipsoidal altitude (degrees / metres)
  ECEF    – Earth-Centred Earth-Fixed Cartesian  (metres)
  ENU     – local East-North-Up tangent plane at the survey origin  (metres)
  Camera  – X right, Y down, Z into scene  (pixels / metres)
"""

import numpy as np
from pyproj import Transformer


# ---------------------------------------------------------------------------
# WGS84 → local ENU
# ---------------------------------------------------------------------------

_WGS84_TO_ECEF = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


def gps_to_enu(lats, lons, alts, origin_lat: float, origin_lon: float, origin_alt: float):
    """Convert arrays of WGS84 coordinates to local ENU (metres).

    Returns (east, north, up) arrays relative to the given origin.
    """
    lats, lons, alts = np.asarray(lats), np.asarray(lons), np.asarray(alts)
    ox, oy, oz = _WGS84_TO_ECEF.transform(origin_lon, origin_lat, origin_alt)
    xs, ys, zs = _WGS84_TO_ECEF.transform(lons, lats, alts)

    lat0, lon0 = np.radians(origin_lat), np.radians(origin_lon)
    R = np.array([
        [-np.sin(lon0),               np.cos(lon0),              0             ],
        [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)  ],
        [ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)  ],
    ])

    delta = np.vstack([xs - ox, ys - oy, zs - oz])   # (3, N)
    enu   = R @ delta
    return enu[0], enu[1], enu[2]


def compute_origin(all_metadata: list) -> tuple:
    """Return the mean GPS position of all images as the ENU origin."""
    lats = [m["gps"]["latitude"]  for m in all_metadata]
    lons = [m["gps"]["longitude"] for m in all_metadata]
    alts = [m["gps"]["altitude_abs"] for m in all_metadata]
    return float(np.mean(lats)), float(np.mean(lons)), float(np.mean(alts))


# ---------------------------------------------------------------------------
# Camera rotation  (gimbal angles → 3×3 camera-to-ENU rotation)
# ---------------------------------------------------------------------------

def build_camera_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return a 3×3 rotation matrix that maps camera-frame vectors to ENU world.

    Camera frame:  +X right,  +Y down (in image),  +Z into scene
    ENU world:     +X East,   +Y North,             +Z Up

    Gimbal angle conventions (Trimble/drone):
      yaw   – degrees clockwise from geographic North (0 = North, 90 = East)
      pitch – degrees forward tilt from nadir  (0 = pointing straight down)
      roll  – degrees right roll               (0 = level, positive = right down)
    """
    yaw   = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll  = np.radians(roll_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    # R_yaw  – rotate around world Z by (-yaw), i.e. CCW by yaw_degrees
    R_yaw = np.array([[ cy, sy, 0.],
                      [-sy, cy, 0.],
                      [ 0., 0., 1.]])

    # R_base – nadir, North-aligned camera → ENU
    #   cam X = East, cam Y = South (= -North), cam Z = Down (= -Up)
    R_base = np.array([[1.,  0.,  0.],
                       [0., -1.,  0.],
                       [0.,  0., -1.]])

    # R_pitch – forward tilt in camera frame (positive pitch = camera looks
    # ahead of nadir, so we negate the angle to rotate cam Z toward -cam Y,
    # which R_base maps to the drone's forward/heading direction in ENU)
    R_pitch = np.array([[1.,   0.,   0.],
                        [0.,  cp,  -sp],
                        [0.,  sp,   cp]])

    # R_roll  – right roll in camera frame (Ry)
    R_roll = np.array([[ cr, 0., sr],
                       [ 0., 1., 0.],
                       [-sr, 0., cr]])

    return R_yaw @ R_base @ R_pitch @ R_roll


# ---------------------------------------------------------------------------
# Camera pose
# ---------------------------------------------------------------------------

def build_camera_pose(metadata: dict, origin: tuple) -> dict:
    """Compute the camera position in ENU and rotation matrix.

    Returns a dict with keys:
      position  – (3,) float64 ENU position [E, N, U]
      R_cw      – (3,3) camera-to-world rotation matrix
    """
    gps     = metadata["gps"]
    gimbal  = metadata["gimbal"]
    origin_lat, origin_lon, origin_alt = origin

    e, n, u = gps_to_enu(
        [gps["latitude"]], [gps["longitude"]], [gps["altitude_abs"]],
        origin_lat, origin_lon, origin_alt,
    )
    # Override the Up component with the relative altitude for ray casting
    # (relative altitude is height above ground, which is what we need for
    # projecting onto z=0)
    position = np.array([float(e[0]), float(n[0]), gps["altitude_relative"]])

    R_cw = build_camera_rotation(gimbal["roll"], gimbal["pitch"], gimbal["yaw"])
    return {"position": position, "R_cw": R_cw}


# ---------------------------------------------------------------------------
# Image footprint
# ---------------------------------------------------------------------------

def compute_image_footprint(pose: dict, calib: dict, img_w: int, img_h: int) -> np.ndarray:
    """Return the 4 ground corners of the image as (4, 2) ENU [E, N] array.

    Computed by ray-casting the 4 image corners onto the z=0 ground plane.
    """
    corners_px = np.array([
        [0,     0    ],
        [img_w, 0    ],
        [img_w, img_h],
        [0,     img_h],
    ], dtype=np.float64)
    ground = project_pixels_to_ground(corners_px, pose, calib)
    return ground[:, :2]


def project_pixels_to_ground(
    uv: np.ndarray, pose: dict, calib: dict
) -> np.ndarray:
    """Ray-cast pixel coordinates onto the z=0 ground plane.

    Parameters
    ----------
    uv : (N, 2) array of (u, v) pixel coordinates
    pose : camera pose dict from build_camera_pose()
    calib : calibration dict from metadata

    Returns
    -------
    (N, 3) ENU ground points  (z ≈ 0 for all)
    """
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    R_cw = pose["R_cw"]
    cam_pos = pose["position"]

    # Normalised camera-frame directions for each pixel
    u = np.asarray(uv[:, 0], dtype=np.float64)
    v = np.asarray(uv[:, 1], dtype=np.float64)
    rays_cam = np.vstack([(u - cx) / fx,
                          (v - cy) / fy,
                          np.ones(len(u))])          # (3, N)

    # Rotate rays into ENU world frame
    rays_world = R_cw @ rays_cam                     # (3, N)

    # Intersect with z=0 plane: cam_pos.z + t * ray.z = 0
    z_cam = cam_pos[2]
    rz    = rays_world[2]
    # Avoid divide-by-zero for near-horizontal rays
    safe  = np.abs(rz) > 1e-6
    t = np.where(safe, -z_cam / rz, np.nan)

    ground = cam_pos[:, None] + t[None, :] * rays_world   # (3, N)
    return ground.T                                         # (N, 3)

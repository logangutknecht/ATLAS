"""Point cloud color palettes and color computation."""

import numpy as np
from vispy.color import Colormap, get_colormap


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

TURBO_COLORS = [
    (0.18995, 0.07176, 0.23217),
    (0.22500, 0.27400, 0.75660),
    (0.13700, 0.56500, 0.86800),
    (0.12200, 0.76300, 0.60600),
    (0.45400, 0.90400, 0.25200),
    (0.83500, 0.88600, 0.10600),
    (0.98800, 0.55200, 0.06000),
    (0.89000, 0.28000, 0.07600),
    (0.60000, 0.04000, 0.18000),
]
TURBO_CMAP = Colormap(TURBO_COLORS, interpolation="linear")

ASPRS_COLORS = {
    0:  (0.50, 0.50, 0.50, 1.0),
    1:  (0.60, 0.60, 0.60, 1.0),
    2:  (0.65, 0.45, 0.20, 1.0),
    3:  (0.45, 0.75, 0.30, 1.0),
    4:  (0.20, 0.65, 0.15, 1.0),
    5:  (0.05, 0.45, 0.05, 1.0),
    6:  (0.90, 0.25, 0.20, 1.0),
    7:  (0.55, 0.15, 0.55, 1.0),
    8:  (0.70, 0.70, 0.70, 1.0),
    9:  (0.15, 0.40, 0.90, 1.0),
    10: (0.35, 0.70, 0.85, 1.0),
    11: (0.40, 0.40, 0.40, 1.0),
    17: (0.90, 0.80, 0.20, 1.0),
    18: (0.80, 0.50, 0.20, 1.0),
}

RETURN_PALETTE = np.array([
    [0.12, 0.47, 0.71, 1.0],
    [1.00, 0.50, 0.05, 1.0],
    [0.17, 0.63, 0.17, 1.0],
    [0.84, 0.15, 0.16, 1.0],
    [0.58, 0.40, 0.74, 1.0],
    [0.55, 0.34, 0.29, 1.0],
    [0.89, 0.47, 0.76, 1.0],
    [0.50, 0.50, 0.50, 1.0],
], dtype=np.float32)

# Deviation (anomaly) coloring
DEV_COLOR_GROUND = (0.90, 0.55, 0.10, 1.0)   # orange — normal ground
DEV_COLOR_DIP    = (0.55, 0.20, 0.70, 1.0)   # purple — dip
DEV_COLOR_BUMP   = (0.95, 0.85, 0.45, 1.0)   # yellow — bump


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(values):
    lo, hi = values.min(), values.max()
    rng = hi - lo
    return (values - lo) / rng if rng > 0 else np.zeros_like(values)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_colors(points, attrs, mode):
    """Return an (N, 4) float32 RGBA array for the requested colouring mode."""
    if mode == "RGB" and "rgb" in attrs:
        rgb = attrs["rgb"].astype(np.float32) / 255.0
        return np.column_stack([rgb, np.ones(len(rgb), dtype=np.float32)])

    if mode == "Classification" and "classification" in attrs:
        cls = attrs["classification"].astype(np.int32)
        out = np.full((len(cls), 4), 0.5, dtype=np.float32)
        out[:, 3] = 1.0
        for code, rgba in ASPRS_COLORS.items():
            m = cls == code
            if m.any():
                out[m] = rgba
        return out

    if mode == "Return Number" and "return_number" in attrs:
        rn = attrs["return_number"].astype(np.int32)
        idx = np.clip(rn - 1, 0, len(RETURN_PALETTE) - 1)
        return RETURN_PALETTE[idx]

    if mode == "Intensity" and "intensity" in attrs:
        return get_colormap("grays").map(_normalize(attrs["intensity"])).astype(np.float32)

    # Default: elevation-based Turbo colormap
    return TURBO_CMAP.map(_normalize(points[:, 2])).astype(np.float32)

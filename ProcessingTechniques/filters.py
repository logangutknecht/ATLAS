"""Point-cloud filters: outlier removal, downsampling, ground segmentation."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import (
    distance_transform_edt,
    uniform_filter,
    minimum_filter,
    maximum_filter,
)


def statistical_outlier_removal(points, k=20, std_ratio=2.0, progress_fn=None):
    """Return a boolean keep-mask after SOR filtering.

    Points whose mean k-neighbour distance exceeds
    ``mean + std_ratio * std`` are marked as outliers.
    """
    if progress_fn:
        progress_fn("SOR: KD-tree on {:,} pts...".format(len(points)))
    tree = cKDTree(points)
    if progress_fn:
        progress_fn("SOR: neighbour distances...")
    d, _ = tree.query(points, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    return mean_d <= (mean_d.mean() + std_ratio * mean_d.std())


def voxel_downsample(points, voxel_size=0.1, progress_fn=None):
    """Return a boolean mask keeping one point per occupied voxel."""
    if progress_fn:
        progress_fn("Voxel downsampling...")
    keys = np.floor(points / voxel_size).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    mask = np.zeros(len(points), dtype=bool)
    mask[idx] = True
    return mask


def elevation_clip(points, z_min, z_max):
    """Return a boolean mask keeping points within [z_min, z_max]."""
    z = points[:, 2]
    return (z >= z_min) & (z <= z_max)


def ground_segmentation(points, cell_size=1.0, height_threshold=0.3,
                         progress_fn=None):
    """Morphological ground segmentation.

    Builds a minimum-Z surface on a regular XY grid, smooths it with
    progressive morphological opening, then classifies points within
    *height_threshold* of the surface as ground.

    Returns a boolean mask (True = ground).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if progress_fn:
        progress_fn("Ground seg: building surface grid...")

    x_min, y_min = x.min(), y.min()
    nx = int(np.ceil((x.max() - x_min) / cell_size)) + 1
    ny = int(np.ceil((y.max() - y_min) / cell_size)) + 1

    xi = np.clip(((x - x_min) / cell_size).astype(int), 0, nx - 1)
    yi = np.clip(((y - y_min) / cell_size).astype(int), 0, ny - 1)

    surface = np.full(ny * nx, np.inf)
    np.minimum.at(surface, yi * nx + xi, z)
    surface = surface.reshape(ny, nx)

    # Fill empty cells via nearest-neighbour
    empty = np.isinf(surface)
    if empty.any() and (~empty).any():
        idxs = distance_transform_edt(
            empty, return_indices=True, return_distances=False,
        )
        surface[empty] = surface[tuple(idxs[:, empty])]
    elif empty.all():
        return np.ones(len(points), dtype=bool)

    if progress_fn:
        progress_fn("Ground seg: morphological smoothing...")

    for win in (3, 5, 9):
        opened = maximum_filter(minimum_filter(surface, size=win), size=win)
        surface = np.minimum(surface, opened + height_threshold)

    surface = uniform_filter(surface, size=max(3, int(2.0 / cell_size)))

    ground_z = surface[yi, xi]
    return (z - ground_z) <= height_threshold

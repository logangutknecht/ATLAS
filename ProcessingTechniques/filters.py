"""Point-cloud filters: outlier removal, downsampling, ground segmentation."""

import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.spatial import cKDTree, Delaunay


def statistical_outlier_removal(points, k=20, std_ratio=2.0, progress_fn=None):
    """Return a boolean keep-mask after SOR filtering.

    Points whose mean k-neighbour distance exceeds
    ``mean + std_ratio * std`` are marked as outliers.
    """
    N = len(points)
    if progress_fn:
        progress_fn("SOR: KD-tree on {:,} pts…".format(N))
    tree = cKDTree(points, leafsize=32, balanced_tree=False)

    CHUNK = 500_000
    mean_d = np.empty(N, dtype=np.float64)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        d, _ = tree.query(points[start:end], k=k + 1, workers=-1)
        mean_d[start:end] = d[:, 1:].mean(axis=1)
        if progress_fn:
            progress_fn("SOR: {:,} / {:,} pts…".format(end, N))

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


# ---------------------------------------------------------------------------
# Fast Progressive TIN Densification (FPTD) ground segmentation
# Based on Li, Ye et al. 2021  (IEEE J-STARS, DOI 10.1109/JSTARS.2021.3131586)
#
# Optimisations over the per-block iterative approach:
#   1. Single GLOBAL Delaunay from seeds → no per-block overhead
#   2. Spatially-sorted find_simplex queries → O(1) amortised walk
#   3. Grid-thinned TIN cap (≤12 K pts) → fast Delaunay rebuilds
#   4. Max 5 global iterations instead of 50 per-block
#   5. Squared-distance angle test → avoids 3×N sqrt calls
# ---------------------------------------------------------------------------


def _grid_thin(xy, indices, max_pts, cell_hint):
    """Thin *indices* to ≤ *max_pts* via one-per-grid-cell selection."""
    if len(indices) <= max_pts:
        return indices
    px, py = xy[indices, 0], xy[indices, 1]
    area = max(float((px.max() - px.min()) * (py.max() - py.min())), 1.0)
    cell = max(np.sqrt(area / max_pts), cell_hint)
    gx = np.floor((px - px.min()) / cell).astype(np.int32)
    gy = np.floor((py - py.min()) / cell).astype(np.int32)
    keys = gy.astype(np.int64) * max(int(gx.max()) + 1, 1) + gx
    _, ui = np.unique(keys, return_index=True)
    return indices[ui]


def ground_segmentation(points, cell_size=1.0, height_threshold=0.3,
                         progress_fn=None):
    """Fast Progressive TIN Densification ground segmentation.

    Uses a single global Delaunay built from grid-cell seed points and
    iteratively densifies (max 5 passes).  Candidates are spatially
    sorted for cache-friendly simplex lookups.

    Parameters
    ----------
    points : (N, 3) ndarray  –  x, y, z
    cell_size : float
        Grid stride for seed selection (one seed per cell).
    height_threshold : float
        Maximum relative elevation above the highest triangle vertex
        for a point to be classified as ground.
    progress_fn : callable, optional
        ``fn(message: str)`` for progress updates.

    Returns
    -------
    mask : (N,) bool ndarray – True where a point is classified as ground.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    N = len(points)
    stride = max(float(cell_size), 0.1)
    max_sin_sq = np.sin(np.radians(30.0)) ** 2

    x_min, y_min = float(x.min()), float(y.min())
    x_max, y_max = float(x.max()), float(y.max())

    # ==================================================================
    # Step 1 — Seed selection with morphological ground filter
    # Lowest-z per stride cell, then morphological opening (erode +
    # dilate) to estimate bare ground.  Seeds far above that surface
    # are on vegetation canopy and get rejected.
    # ==================================================================
    if progress_fn:
        progress_fn("FPTD: seed selection …")

    gnx = max(1, int(np.ceil((x_max - x_min) / stride)) + 1)
    gny = max(1, int(np.ceil((y_max - y_min) / stride)) + 1)
    gxi = np.clip(np.floor((x - x_min) / stride).astype(np.int32),
                   0, gnx - 1)
    gyi = np.clip(np.floor((y - y_min) / stride).astype(np.int32),
                   0, gny - 1)

    cell_key = gyi.astype(np.int64) * gnx + gxi
    order = np.lexsort((z, cell_key))
    ks = cell_key[order]
    first = np.empty(len(ks), dtype=bool)
    first[0] = True
    first[1:] = ks[1:] != ks[:-1]

    min_pts = order[first]
    min_keys = ks[first]
    cy_arr = (min_keys // gnx).astype(np.intp)
    cx_arr = (min_keys % gnx).astype(np.intp)

    grid_z = np.full((gny, gnx), np.inf)
    grid_idx = np.full((gny, gnx), -1, dtype=np.intp)
    grid_z[cy_arr, cx_arr] = z[min_pts]
    grid_idx[cy_arr, cx_arr] = min_pts

    # Morphological opening (~30 m window) to estimate bare ground
    open_wc = max(3, int(np.ceil(30.0 / stride)))
    eroded = minimum_filter(grid_z, size=open_wc,
                            mode="constant", cval=np.inf)
    eroded_safe = np.where(np.isfinite(eroded), eroded, -np.inf)
    opened = maximum_filter(eroded_safe, size=open_wc,
                            mode="constant", cval=-np.inf)

    seed_tol = max(height_threshold * 5, 2.0)
    is_gnd = (np.isfinite(grid_z)
              & np.isfinite(opened)
              & ((grid_z - opened) <= seed_tol))
    seed_raw = grid_idx[is_gnd]
    seed_indices = seed_raw[seed_raw >= 0]

    if len(seed_indices) < 3:
        seed_raw = grid_idx[np.isfinite(grid_z)]
        seed_indices = seed_raw[seed_raw >= 0]

    ground = np.zeros(N, dtype=bool)
    ground[seed_indices] = True

    if progress_fn:
        progress_fn("FPTD: {:,} seeds".format(len(seed_indices)))
    if len(seed_indices) < 3:
        return ground

    # ==================================================================
    # Step 2 — Pre-filter & spatial sort of candidates
    # ==================================================================
    z_ceiling = float(z[seed_indices].max()) + max(height_threshold, 10.0)
    candidates = np.where((~ground) & (z <= z_ceiling))[0]
    if len(candidates) == 0:
        return ground

    sort_cell = max(stride, 5.0)
    sny = max(1, int(np.ceil((y_max - y_min) / sort_cell)) + 1)
    sk = (np.floor((x[candidates] - x_min) / sort_cell).astype(np.int64)
          * sny
          + np.floor((y[candidates] - y_min) / sort_cell).astype(np.int64))
    candidates = candidates[np.argsort(sk, kind="mergesort")]

    if progress_fn:
        progress_fn("FPTD: {:,} candidates".format(len(candidates)))

    # ==================================================================
    # Step 3 — Global iterative TIN densification (max 5 passes)
    # ==================================================================
    MAX_TIN = 12000

    for it in range(5):
        gnd_idx = np.where(ground)[0]
        tin_idx = _grid_thin(points[:, :2], gnd_idx, MAX_TIN, stride)
        tin_pts = points[tin_idx]

        try:
            tri = Delaunay(tin_pts[:, :2])
        except Exception:
            break

        simp = tri.find_simplex(points[candidates, :2])
        inside = simp >= 0
        if not inside.any():
            break

        ci = np.where(inside)[0]
        tv = tri.simplices[simp[ci]]
        Ap = tin_pts[tv[:, 0]]
        Bp = tin_pts[tv[:, 1]]
        Cp = tin_pts[tv[:, 2]]

        z_max_v = np.maximum(np.maximum(Ap[:, 2], Bp[:, 2]), Cp[:, 2])
        cand_z = z[candidates[ci]]
        rel_elev = cand_z - z_max_v

        qxyz = points[candidates[ci]]
        AB = Bp - Ap
        AC = Cp - Ap
        nrm = np.cross(AB, AC)
        nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm /= np.maximum(nrm_len, 1e-12)
        nrm[nrm[:, 2] < 0] *= -1
        signed_d = ((qxyz - Ap) * nrm).sum(axis=1)

        dA_sq = ((qxyz - Ap) ** 2).sum(axis=1)
        dB_sq = ((qxyz - Bp) ** 2).sum(axis=1)
        dC_sq = ((qxyz - Cp) ** 2).sum(axis=1)
        l_sq = np.maximum(np.minimum(np.minimum(dA_sq, dB_sq), dC_sq),
                          1e-24)
        angle_ok = (signed_d <= 0) | (signed_d * signed_d <= max_sin_sq * l_sq)

        ok = (rel_elev <= height_threshold) & angle_ok
        accept = ci[ok]
        if len(accept) == 0:
            break

        ground[candidates[accept]] = True

        keep = np.ones(len(candidates), dtype=bool)
        keep[accept] = False
        candidates = candidates[keep]

        if progress_fn:
            progress_fn("FPTD: iter {} — {:,} ground".format(
                it + 1, int(ground.sum())))
        if len(candidates) == 0:
            break

    if progress_fn:
        n_g = int(ground.sum())
        progress_fn("FPTD: done – {:,} ground / {:,} non-ground".format(
            n_g, N - n_g))
    return ground

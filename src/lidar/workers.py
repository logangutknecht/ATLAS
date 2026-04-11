"""Background worker threads for LiDAR file loading and processing."""

import os
import time

import numpy as np
import laspy

from PyQt6.QtCore import QThread, pyqtSignal

from src.lidar.processing.geo import (
    extract_crs_wkt,
    extract_epsg_from_geokeys,
    build_crs,
    describe_coord_ranges,
)
from src.lidar.processing.filters import (
    statistical_outlier_removal,
    voxel_downsample,
    elevation_clip,
    ground_segmentation,
    detect_ground_anomalies,
)
from src.lidar.processing.satellite import generate_satellite_colors


class LASLoaderThread(QThread):
    """Reads a LAS/LAZ file and emits a cloud data dict when done."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath, max_points=None):
        super().__init__()
        self.filepath = filepath
        self.max_points = max_points

    def run(self):
        try:
            self._load()
        except MemoryError:
            self.error.emit("Not enough memory — reduce Max Points and retry.")
        except Exception as exc:
            self.error.emit(str(exc))

    def _load(self):
        name = os.path.basename(self.filepath)
        self.progress.emit("Reading {}...".format(name))
        t0 = time.time()

        las = laspy.read(self.filepath)
        total = las.header.point_count
        self.progress.emit("Processing {:,} points...".format(total))

        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        indices = self._subsample_indices(total)
        if indices is not None:
            xyz = xyz[indices]

        centroid = xyz.mean(axis=0)
        pts = (xyz - centroid).astype(np.float32)
        sub = lambda a: a[indices] if indices is not None else a

        crs_wkt = extract_crs_wkt(las.header)
        epsg_code = extract_epsg_from_geokeys(las.header)
        resolved = build_crs(las.header)
        resolved_epsg, crs_info = self._resolve_crs_info(resolved)
        coord_hint = "" if resolved else describe_coord_ranges(
            float(xyz[:, 0].min()), float(xyz[:, 0].max()),
            float(xyz[:, 1].min()), float(xyz[:, 1].max()),
        )

        data = {
            "points": pts,
            "centroid": centroid,
            "total_points": total,
            "displayed_points": len(pts),
            "filepath": self.filepath,
            "load_time": time.time() - t0,
            "attributes": {},
            "available_coloring": ["Elevation"],
            "crs_wkt": crs_wkt,
            "crs_epsg": resolved_epsg or epsg_code,
            "crs_info": crs_info,
            "coord_hint": coord_hint,
        }
        self._extract_attributes(las, sub, data)
        self.finished.emit(data)

    def _subsample_indices(self, total):
        if self.max_points and total > self.max_points:
            return np.sort(np.random.choice(total, self.max_points, replace=False))
        return None

    def _resolve_crs_info(self, resolved):
        if resolved is None:
            return None, ""
        try:
            epsg = resolved.to_epsg()
        except Exception:
            epsg = None
        name = "{} (EPSG:{})".format(resolved.name, epsg) if epsg else resolved.name
        return epsg, name

    def _extract_attributes(self, las, sub, data):
        if hasattr(las, "intensity"):
            data["attributes"]["intensity"] = sub(np.asarray(las.intensity, dtype=np.float32))
            data["available_coloring"].append("Intensity")

        if hasattr(las, "classification"):
            data["attributes"]["classification"] = sub(np.asarray(las.classification, dtype=np.float32))
            data["available_coloring"].append("Classification")

        if hasattr(las, "return_number"):
            data["attributes"]["return_number"] = sub(np.asarray(las.return_number, dtype=np.float32))
            data["available_coloring"].append("Return Number")

        if all(hasattr(las, c) for c in ("red", "green", "blue")):
            rgb = np.vstack([las.red, las.green, las.blue]).T
            rgb = sub(rgb).astype(np.float64)
            if rgb.max() > 255:
                rgb = rgb / 65535.0 * 255.0
            data["attributes"]["rgb"] = rgb.astype(np.uint8)
            data["available_coloring"].append("RGB")


class ProcessingThread(QThread):
    """Dispatches a processing operation to the filters module."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, operation, points, **kwargs):
        super().__init__()
        self.op = operation
        self.pts = points
        self.kw = kwargs

    def run(self):
        try:
            result = self._dispatch()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def _dispatch(self):
        pfn = self.progress.emit
        kw = self.kw
        if self.op == "sor":
            return statistical_outlier_removal(self.pts, kw["k"], kw["std_ratio"], pfn)
        if self.op == "voxel":
            return voxel_downsample(self.pts, kw["voxel_size"], pfn)
        if self.op == "elevation":
            return elevation_clip(self.pts, kw["z_min"], kw["z_max"])
        if self.op == "ground":
            return ground_segmentation(self.pts, kw["cell_size"], kw["height_threshold"], pfn)
        if self.op == "anomaly":
            return detect_ground_anomalies(
                self.pts, kw["ground_mask"],
                kw["reference_spacing"], kw["threshold"], kw["min_area"], pfn,
            )
        raise ValueError("Unknown operation: " + self.op)


class SatelliteThread(QThread):
    """Fetches satellite imagery and colorizes every point in the cloud."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cloud_data, epsg_override=None):
        super().__init__()
        self.cloud = cloud_data
        self.epsg_override = epsg_override

    def run(self):
        try:
            result = generate_satellite_colors(
                self.cloud["points"],
                self.cloud["centroid"],
                self.cloud.get("crs_wkt"),
                self.cloud.get("crs_epsg"),
                self.epsg_override,
                progress_fn=self.progress.emit,
            )
            if result["error"]:
                self.error.emit(result["error"])
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

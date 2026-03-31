"""
LiDAR Point Cloud Viewer

High-performance GPU-accelerated point cloud viewer built on vispy (OpenGL)
and PyQt6.  Processing algorithms live in the ProcessingTechniques package.

Mouse controls (3-D view):
    Left drag       Rotate (orbit)
    Scroll          Zoom
    Shift+Left drag Pan
    Middle drag     Pan
"""

import sys
import os
import time

import numpy as np
import laspy

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QFileDialog, QGroupBox,
    QDoubleSpinBox, QProgressBar, QStatusBar, QSplitter, QScrollArea,
    QMessageBox, QSpinBox, QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction

from vispy import scene
from vispy.color import Colormap, get_colormap

from ProcessingTechniques.geo import (
    extract_crs_wkt, extract_epsg_from_geokeys, build_crs,
    describe_coord_ranges,
)
from ProcessingTechniques.filters import (
    statistical_outlier_removal, voxel_downsample,
    elevation_clip, ground_segmentation,
)
from ProcessingTechniques.satellite import generate_satellite_colors

# ---------------------------------------------------------------------------
# Colour constants
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
    0: (0.50, 0.50, 0.50, 1.0),  1: (0.60, 0.60, 0.60, 1.0),
    2: (0.65, 0.45, 0.20, 1.0),  3: (0.45, 0.75, 0.30, 1.0),
    4: (0.20, 0.65, 0.15, 1.0),  5: (0.05, 0.45, 0.05, 1.0),
    6: (0.90, 0.25, 0.20, 1.0),  7: (0.55, 0.15, 0.55, 1.0),
    8: (0.70, 0.70, 0.70, 1.0),  9: (0.15, 0.40, 0.90, 1.0),
    10: (0.35, 0.70, 0.85, 1.0), 11: (0.40, 0.40, 0.40, 1.0),
    17: (0.90, 0.80, 0.20, 1.0), 18: (0.80, 0.50, 0.20, 1.0),
}

RETURN_PALETTE = np.array([
    [0.12, 0.47, 0.71, 1.0], [1.00, 0.50, 0.05, 1.0],
    [0.17, 0.63, 0.17, 1.0], [0.84, 0.15, 0.16, 1.0],
    [0.58, 0.40, 0.74, 1.0], [0.55, 0.34, 0.29, 1.0],
    [0.89, 0.47, 0.76, 1.0], [0.50, 0.50, 0.50, 1.0],
], dtype=np.float32)

DARK_STYLE = """
QMainWindow { background-color: #1e1e1e; }
QWidget { background-color: #2d2d2d; color: #cccccc; font-size: 12px; }
QGroupBox {
    border: 1px solid #555; border-radius: 4px;
    margin-top: 10px; padding-top: 18px;
    font-weight: bold; color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 2px 8px;
}
QPushButton {
    background-color: #0078d4; color: white; border: none;
    border-radius: 4px; padding: 6px 14px; font-weight: bold;
}
QPushButton:hover { background-color: #1a8ae8; }
QPushButton:pressed { background-color: #005a9e; }
QPushButton:disabled { background-color: #444; color: #777; }
QComboBox {
    background-color: #3c3c3c; border: 1px solid #555;
    border-radius: 4px; padding: 4px 8px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #3c3c3c; selection-background-color: #0078d4; color: #ccc;
}
QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #0078d4; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #0078d4; border-radius: 3px; }
QScrollArea { border: none; }
QStatusBar { background-color: #1e1e1e; color: #999; border-top: 1px solid #333; }
QProgressBar {
    background-color: #3c3c3c; border: 1px solid #555;
    border-radius: 4px; text-align: center; color: white;
}
QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c; border: 1px solid #555;
    border-radius: 4px; padding: 2px 6px;
}
QLineEdit {
    background-color: #3c3c3c; border: 1px solid #555;
    border-radius: 4px; padding: 2px 6px;
}
QLabel { background-color: transparent; }
QMenuBar { background-color: #1e1e1e; color: #ccc; border-bottom: 1px solid #333; }
QMenuBar::item:selected { background-color: #0078d4; }
QMenu { background-color: #2d2d2d; border: 1px solid #555; }
QMenu::item:selected { background-color: #0078d4; }
QSplitter::handle { background-color: #333; width: 2px; }
"""


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _normalize(values):
    lo, hi = values.min(), values.max()
    rng = hi - lo
    return (values - lo) / rng if rng > 0 else np.zeros_like(values)


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

    return TURBO_CMAP.map(_normalize(points[:, 2])).astype(np.float32)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class LASLoaderThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath, max_points=None):
        super().__init__()
        self.filepath = filepath
        self.max_points = max_points

    def run(self):
        try:
            name = os.path.basename(self.filepath)
            self.progress.emit("Reading {}...".format(name))
            t0 = time.time()

            las = laspy.read(self.filepath)
            total = las.header.point_count
            self.progress.emit("Processing {:,} points...".format(total))

            xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

            indices = None
            if self.max_points and total > self.max_points:
                indices = np.sort(
                    np.random.choice(total, self.max_points, replace=False)
                )
                xyz = xyz[indices]

            centroid = xyz.mean(axis=0)
            pts = (xyz - centroid).astype(np.float32)
            sub = (lambda a: a[indices] if indices is not None else a)

            crs_wkt = extract_crs_wkt(las.header)
            epsg_code = extract_epsg_from_geokeys(las.header)

            resolved = build_crs(las.header)
            resolved_epsg = None
            crs_info = ""
            if resolved is not None:
                try:
                    resolved_epsg = resolved.to_epsg()
                except Exception:
                    pass
                if resolved_epsg:
                    crs_info = "{} (EPSG:{})".format(resolved.name, resolved_epsg)
                else:
                    crs_info = resolved.name

            coord_hint = ""
            if resolved is None:
                coord_hint = describe_coord_ranges(
                    float(xyz[:, 0].min()), float(xyz[:, 0].max()),
                    float(xyz[:, 1].min()), float(xyz[:, 1].max()),
                )

            data = {
                "points": pts, "centroid": centroid,
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

            self.finished.emit(data)
        except MemoryError:
            self.error.emit("Not enough memory - reduce Max Points and retry.")
        except Exception as exc:
            self.error.emit(str(exc))


class ProcessingThread(QThread):
    """Dispatches to ProcessingTechniques.filters functions."""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, operation, points, **kwargs):
        super().__init__()
        self.op = operation
        self.pts = points
        self.kw = kwargs

    def run(self):
        try:
            pfn = self.progress.emit
            if self.op == "sor":
                mask = statistical_outlier_removal(self.pts, self.kw["k"], self.kw["std_ratio"], pfn)
            elif self.op == "voxel":
                mask = voxel_downsample(self.pts, self.kw["voxel_size"], pfn)
            elif self.op == "elevation":
                mask = elevation_clip(self.pts, self.kw["z_min"], self.kw["z_max"])
            elif self.op == "ground":
                mask = ground_segmentation(self.pts, self.kw["cell_size"], self.kw["height_threshold"], pfn)
            else:
                raise ValueError("Unknown op: " + self.op)
            self.finished.emit(mask)
        except Exception as exc:
            self.error.emit(str(exc))


class SatelliteThread(QThread):
    """Fetch satellite imagery and colorize every point in the cloud."""
    finished = pyqtSignal(object)   # dict: {colors, info}
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


# ---------------------------------------------------------------------------
# Custom camera: right-click drag = pan
# ---------------------------------------------------------------------------

class _PanCamera(scene.TurntableCamera):
    """TurntableCamera with right-click drag mapped to pan.

    Default vispy mapping:  left=orbit, right=zoom, shift+left=pan
    This camera remaps:     left=orbit, right=pan,  scroll=zoom
    """

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        if (event.type == 'mouse_move'
                and event.press_event is not None
                and 2 in event.buttons
                and 1 not in event.buttons):
            p1 = event.last_event.pos[:2]
            p2 = event.pos[:2]
            d = np.array(p2, dtype=np.float64) - np.array(p1, dtype=np.float64)

            vb = self._viewbox.size
            norm = max(vb[0], vb[1], 1.0)
            dist = getattr(self, '_actual_distance', None)
            if dist is None:
                dist = self.distance if self.distance else 500.0
            speed = dist / norm

            az = np.radians(self.azimuth)
            el = np.radians(self.elevation)

            right = np.array([np.cos(az), np.sin(az), 0.0])
            up = np.array([
                -np.sin(el) * np.sin(az),
                 np.sin(el) * np.cos(az),
                 np.cos(el),
            ])

            shift = (-d[0] * right + d[1] * up) * speed
            c = np.array(self.center, dtype=np.float64)
            self.center = tuple(c + shift)
            self.view_changed()
            event.handled = True
            return

        super().viewbox_mouse_event(event)


# ---------------------------------------------------------------------------
# 3-D viewer widget
# ---------------------------------------------------------------------------

class PointCloudView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=False, bgcolor="#1a1a2e",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = _PanCamera(fov=45, distance=500)
        self._scatter = scene.visuals.Markers(parent=self.view.scene)
        self._axis = scene.visuals.XYZAxis(parent=self.view.scene)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas.native)
        self._pts = self._colors = None
        self._size = 2

    def display(self, points, colors, size=2):
        self._pts, self._colors, self._size = points, colors, size
        self._scatter.set_data(pos=points, face_color=colors, size=size, edge_width=0)
        self.view.camera.set_range()
        self.canvas.update()

    def update_size(self, size):
        if self._pts is None:
            return
        self._size = size
        self._scatter.set_data(pos=self._pts, face_color=self._colors, size=size, edge_width=0)
        self.canvas.update()

    def set_bg(self, color):
        self.canvas.bgcolor = color

    def reset_camera(self):
        self.view.camera.set_range()
        self.canvas.update()

    def set_camera_angles(self, elevation, azimuth):
        self.view.camera.elevation = elevation
        self.view.camera.azimuth = azimuth
        self.canvas.update()


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    file_requested = pyqtSignal()
    color_changed = pyqtSignal(str)
    point_size_changed = pyqtSignal(int)
    max_points_changed = pyqtSignal(int)
    bg_toggled = pyqtSignal()
    reset_camera = pyqtSignal()
    view_preset = pyqtSignal(str)
    sor_requested = pyqtSignal(int, float)
    voxel_requested = pyqtSignal(float)
    elev_requested = pyqtSignal(float, float)
    ground_requested = pyqtSignal(float, float)
    hide_nonground_toggled = pyqtSignal(bool)
    reset_processing = pyqtSignal()
    satellite_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(290)
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setSpacing(6)
        col.setContentsMargins(8, 8, 8, 8)

        col.addWidget(self._file_section())
        col.addWidget(self._display_section())
        col.addWidget(self._perf_section())
        col.addWidget(self._camera_section())
        col.addWidget(self._processing_section())
        col.addWidget(self._satellite_section())
        col.addStretch()

        scroll.setWidget(inner)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    # -- sections -----------------------------------------------------------

    def _file_section(self):
        g = QGroupBox("File")
        lay = QVBoxLayout(g)
        self.load_btn = QPushButton("Open LAS / LAZ File")
        self.load_btn.clicked.connect(self.file_requested.emit)
        lay.addWidget(self.load_btn)
        self.file_info = QLabel("No file loaded")
        self.file_info.setWordWrap(True)
        self.file_info.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(self.file_info)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        lay.addWidget(self.progress)
        return g

    def _display_section(self):
        g = QGroupBox("Display")
        lay = QVBoxLayout(g)
        lay.addWidget(QLabel("Color By:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("Elevation")
        self.color_combo.currentTextChanged.connect(self.color_changed.emit)
        lay.addWidget(self.color_combo)

        lay.addWidget(QLabel("Point Size:"))
        row = QHBoxLayout()
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 15)
        self.size_slider.setValue(2)
        self.size_label = QLabel("2")
        self.size_slider.valueChanged.connect(self.point_size_changed.emit)
        self.size_slider.valueChanged.connect(lambda v: self.size_label.setText(str(v)))
        row.addWidget(self.size_slider)
        row.addWidget(self.size_label)
        lay.addLayout(row)

        self.bg_btn = QPushButton("Toggle Background")
        self.bg_btn.clicked.connect(self.bg_toggled.emit)
        lay.addWidget(self.bg_btn)
        return g

    def _perf_section(self):
        g = QGroupBox("Performance")
        lay = QVBoxLayout(g)
        lay.addWidget(QLabel("Max Points (millions):"))
        row = QHBoxLayout()
        self.max_pts_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_pts_slider.setRange(1, 100)
        self.max_pts_slider.setValue(10)
        self.max_pts_label = QLabel("10 M")
        self.max_pts_slider.valueChanged.connect(lambda v: self.max_pts_label.setText("{} M".format(v)))
        row.addWidget(self.max_pts_slider)
        row.addWidget(self.max_pts_label)
        lay.addLayout(row)
        self.reload_btn = QPushButton("Reload with New Limit")
        self.reload_btn.setEnabled(False)
        self.reload_btn.clicked.connect(lambda: self.max_points_changed.emit(self.max_pts_slider.value() * 1_000_000))
        lay.addWidget(self.reload_btn)
        return g

    def _camera_section(self):
        g = QGroupBox("Camera")
        lay = QVBoxLayout(g)
        btn = QPushButton("Reset Camera")
        btn.clicked.connect(self.reset_camera.emit)
        lay.addWidget(btn)
        row = QHBoxLayout()
        for name in ("Top", "Front", "Right"):
            b = QPushButton(name)
            b.setFixedWidth(78)
            b.clicked.connect(lambda _, n=name: self.view_preset.emit(n))
            row.addWidget(b)
        lay.addLayout(row)
        return g

    def _processing_section(self):
        g = QGroupBox("Post-Processing")
        lay = QVBoxLayout(g)

        # SOR
        lay.addWidget(QLabel("Statistical Outlier Removal"))
        r = QHBoxLayout()
        r.addWidget(QLabel("K:"))
        self.sor_k = QSpinBox(); self.sor_k.setRange(5, 200); self.sor_k.setValue(20)
        r.addWidget(self.sor_k)
        r.addWidget(QLabel("Std:"))
        self.sor_std = QDoubleSpinBox()
        self.sor_std.setRange(0.1, 10.0); self.sor_std.setValue(2.0); self.sor_std.setSingleStep(0.1)
        r.addWidget(self.sor_std)
        lay.addLayout(r)
        self.sor_btn = QPushButton("Remove Outliers")
        self.sor_btn.setEnabled(False)
        self.sor_btn.clicked.connect(lambda: self.sor_requested.emit(self.sor_k.value(), self.sor_std.value()))
        lay.addWidget(self.sor_btn)

        # Voxel
        lay.addWidget(QLabel("Voxel Downsampling"))
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Size:"))
        self.voxel_size = QDoubleSpinBox()
        self.voxel_size.setRange(0.001, 50.0); self.voxel_size.setValue(0.10)
        self.voxel_size.setSingleStep(0.01); self.voxel_size.setDecimals(3)
        r2.addWidget(self.voxel_size)
        lay.addLayout(r2)
        self.voxel_btn = QPushButton("Downsample")
        self.voxel_btn.setEnabled(False)
        self.voxel_btn.clicked.connect(lambda: self.voxel_requested.emit(self.voxel_size.value()))
        lay.addWidget(self.voxel_btn)

        # Elevation clip
        lay.addWidget(QLabel("Elevation Clip"))
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Z min:"))
        self.z_min = QDoubleSpinBox(); self.z_min.setRange(-100000, 100000); self.z_min.setDecimals(2)
        r3.addWidget(self.z_min)
        r3.addWidget(QLabel("Z max:"))
        self.z_max = QDoubleSpinBox(); self.z_max.setRange(-100000, 100000); self.z_max.setValue(1000); self.z_max.setDecimals(2)
        r3.addWidget(self.z_max)
        lay.addLayout(r3)
        self.elev_btn = QPushButton("Clip Elevation")
        self.elev_btn.setEnabled(False)
        self.elev_btn.clicked.connect(lambda: self.elev_requested.emit(self.z_min.value(), self.z_max.value()))
        lay.addWidget(self.elev_btn)

        # Ground segmentation
        lay.addWidget(QLabel("Ground Segmentation"))
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Cell:"))
        self.ground_cell = QDoubleSpinBox()
        self.ground_cell.setRange(0.1, 20.0); self.ground_cell.setValue(1.0)
        self.ground_cell.setSingleStep(0.1); self.ground_cell.setDecimals(1)
        r4.addWidget(self.ground_cell)
        r4.addWidget(QLabel("Ht:"))
        self.ground_thresh = QDoubleSpinBox()
        self.ground_thresh.setRange(0.05, 10.0); self.ground_thresh.setValue(0.3)
        self.ground_thresh.setSingleStep(0.05); self.ground_thresh.setDecimals(2)
        r4.addWidget(self.ground_thresh)
        lay.addLayout(r4)
        self.ground_btn = QPushButton("Segment Ground")
        self.ground_btn.setEnabled(False)
        self.ground_btn.clicked.connect(lambda: self.ground_requested.emit(self.ground_cell.value(), self.ground_thresh.value()))
        lay.addWidget(self.ground_btn)

        self.ground_info = QLabel("")
        self.ground_info.setStyleSheet("color:#888; font-size:11px;")
        self.ground_info.setWordWrap(True)
        lay.addWidget(self.ground_info)

        from PyQt6.QtWidgets import QCheckBox
        self.hide_nonground_cb = QCheckBox("Hide Non-Ground Points")
        self.hide_nonground_cb.setEnabled(False)
        self.hide_nonground_cb.toggled.connect(self.hide_nonground_toggled.emit)
        lay.addWidget(self.hide_nonground_cb)

        # Reset
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setStyleSheet("background-color:#c42b1c; color:white;")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_processing.emit)
        lay.addWidget(self.reset_btn)
        return g

    def _satellite_section(self):
        g = QGroupBox("Satellite Texture")
        lay = QVBoxLayout(g)

        self.crs_label = QLabel("CRS: unknown")
        self.crs_label.setStyleSheet("color:#888; font-size:11px;")
        self.crs_label.setWordWrap(True)
        lay.addWidget(self.crs_label)

        r = QHBoxLayout()
        r.addWidget(QLabel("EPSG:"))
        self.epsg_input = QLineEdit()
        self.epsg_input.setPlaceholderText("e.g. 32617")
        self.epsg_input.setFixedWidth(100)
        r.addWidget(self.epsg_input)
        lay.addLayout(r)

        self.sat_btn = QPushButton("Apply Satellite Texture")
        self.sat_btn.setEnabled(False)
        self.sat_btn.setToolTip(
            "Fetch Esri World Imagery and colorize each point\n"
            "with its satellite photo colour for a GIS terrain view."
        )
        self.sat_btn.clicked.connect(self.satellite_requested.emit)
        lay.addWidget(self.sat_btn)

        self.sat_info = QLabel("")
        self.sat_info.setStyleSheet("color:#888; font-size:11px;")
        self.sat_info.setWordWrap(True)
        lay.addWidget(self.sat_info)
        return g

    # -- helpers ------------------------------------------------------------

    def set_loading(self, busy):
        self.load_btn.setEnabled(not busy)
        self.progress.setVisible(busy)

    def update_for_data(self, data):
        self.file_info.setText(
            "{}\nTotal: {:,}\nDisplayed: {:,}\nLoaded in {:.1f}s".format(
                os.path.basename(data["filepath"]),
                data["total_points"], data["displayed_points"], data["load_time"],
            )
        )
        self.file_info.setStyleSheet("color:#bbb; font-size:11px;")

        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        for opt in data["available_coloring"]:
            self.color_combo.addItem(opt)
        self.color_combo.blockSignals(False)
        self.color_combo.setCurrentText(
            "RGB" if "RGB" in data["available_coloring"] else "Elevation"
        )

        for b in (self.sor_btn, self.voxel_btn, self.elev_btn,
                   self.reset_btn, self.reload_btn, self.ground_btn,
                   self.sat_btn):
            b.setEnabled(True)

        z = data["points"][:, 2]
        self.z_min.setValue(float(z.min()))
        self.z_max.setValue(float(z.max()))

        crs_info = data.get("crs_info", "")
        crs_epsg = data.get("crs_epsg")
        coord_hint = data.get("coord_hint", "")

        if crs_info:
            self.crs_label.setText("CRS: " + crs_info)
            self.crs_label.setStyleSheet("color:#6c6; font-size:11px;")
            if crs_epsg:
                self.epsg_input.setText(str(crs_epsg))
        elif coord_hint:
            self.crs_label.setText("CRS: not in file\n" + coord_hint)
            self.crs_label.setStyleSheet("color:#da5; font-size:11px;")
            self.epsg_input.clear()
        else:
            self.crs_label.setText("CRS: not found (set EPSG)")
            self.crs_label.setStyleSheet("color:#c66; font-size:11px;")
            self.epsg_input.clear()

        self.ground_info.setText("")
        self.sat_info.setText("")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class LiDARViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Point Cloud Viewer")
        self.setMinimumSize(1100, 700)
        self.resize(1440, 900)

        self._cloud = None
        self._mask = None
        self._ground_mask = None
        self._hide_nonground = False
        self._sat_colors = None          # (N,4) satellite RGBA for full cloud
        self._dark_bg = True
        self._filepath = None
        self._loader = None
        self._worker = None
        self._sat_worker = None

        self._build_ui()
        self._build_menu()
        self._wire()

        self._las_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiDAR_LAS")
        if not os.path.isdir(self._las_dir):
            self._las_dir = os.path.dirname(os.path.abspath(__file__))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.controls = ControlPanel()
        splitter.addWidget(self.controls)
        self.viewer = PointCloudView()
        splitter.addWidget(self.viewer)
        splitter.setSizes([290, 1150])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        self.status = QLabel("Ready  -  Ctrl+O to open a LAS file")
        bar = QStatusBar()
        bar.addPermanentWidget(self.status)
        self.setStatusBar(bar)

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        a = QAction("&Open LAS/LAZ...", self); a.setShortcut("Ctrl+O")
        a.triggered.connect(self._open_dialog); fm.addAction(a)
        fm.addSeparator()
        a2 = QAction("E&xit", self); a2.setShortcut("Ctrl+Q")
        a2.triggered.connect(self.close); fm.addAction(a2)

        vm = mb.addMenu("&View")
        ra = QAction("&Reset Camera", self); ra.setShortcut("R")
        ra.triggered.connect(self.viewer.reset_camera); vm.addAction(ra)
        vm.addSeparator()
        for label, key, el, az in [("Top","Ctrl+1",90,0),("Front","Ctrl+2",0,0),("Right","Ctrl+3",0,90)]:
            va = QAction("{} View".format(label), self); va.setShortcut(key)
            va.triggered.connect(lambda _, e=el, a=az: self.viewer.set_camera_angles(e, a))
            vm.addAction(va)

    def _wire(self):
        c = self.controls
        c.file_requested.connect(self._open_dialog)
        c.color_changed.connect(lambda _: self._render())
        c.point_size_changed.connect(self.viewer.update_size)
        c.bg_toggled.connect(self._toggle_bg)
        c.reset_camera.connect(self.viewer.reset_camera)
        c.view_preset.connect(self._view_preset)
        c.max_points_changed.connect(self._reload)
        c.sor_requested.connect(lambda k, s: self._run_proc("sor", k=k, std_ratio=s))
        c.voxel_requested.connect(lambda v: self._run_proc("voxel", voxel_size=v))
        c.elev_requested.connect(lambda lo, hi: self._run_proc("elevation", z_min=lo, z_max=hi))
        c.ground_requested.connect(self._run_ground_seg)
        c.hide_nonground_toggled.connect(self._toggle_nonground)
        c.reset_processing.connect(self._reset_proc)
        c.satellite_requested.connect(self._run_satellite)

    # -- file I/O -----------------------------------------------------------

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", self._las_dir,
            "LAS files (*.las *.laz);;All files (*)",
        )
        if path:
            self._las_dir = os.path.dirname(path)
            self._load(path, self.controls.max_pts_slider.value() * 1_000_000)

    def _load(self, path, max_pts):
        if self._loader and self._loader.isRunning():
            return
        self._filepath = path
        self.controls.set_loading(True)
        self.status.setText("Loading {}...".format(os.path.basename(path)))
        self._loader = LASLoaderThread(path, max_pts)
        self._loader.progress.connect(self.status.setText)
        self._loader.finished.connect(self._on_loaded)
        self._loader.error.connect(self._on_load_err)
        self._loader.start()

    def _on_loaded(self, data):
        self._cloud = data
        self._mask = None
        self._ground_mask = None
        self._sat_colors = None
        self.controls.set_loading(False)
        self.controls.update_for_data(data)
        self._render()
        self.status.setText("{:,} / {:,} pts | {:.1f}s".format(
            data["displayed_points"], data["total_points"], data["load_time"],
        ))

    def _on_load_err(self, msg):
        self.controls.set_loading(False)
        self.status.setText("Load failed")
        QMessageBox.critical(self, "Error", msg)

    def _reload(self, max_pts):
        if self._filepath:
            self._load(self._filepath, max_pts)

    # -- rendering ----------------------------------------------------------

    def _active_data(self):
        pts = self._cloud["points"]
        attrs = self._cloud["attributes"]
        if self._mask is not None:
            pts = pts[self._mask]
            attrs = {k: v[self._mask] for k, v in attrs.items()}
        return pts, attrs

    def _render(self):
        if not self._cloud:
            return
        pts, attrs = self._active_data()
        mode = self.controls.color_combo.currentText()

        if mode == "Satellite" and self._sat_colors is not None:
            colors = self._sat_colors
            if self._mask is not None:
                colors = colors[self._mask]
        elif mode == "Ground" and self._ground_mask is not None:
            gm = self._ground_mask
            if self._mask is not None:
                gm = gm[self._mask]
            colors = np.zeros((len(pts), 4), dtype=np.float32)
            colors[gm] = [0.55, 0.35, 0.17, 1.0]
            colors[~gm] = [0.15, 0.65, 0.15, 1.0]
        else:
            colors = compute_colors(pts, attrs, mode)

        # Hide non-ground points when the toggle is active
        if self._hide_nonground and self._ground_mask is not None:
            gm = self._ground_mask
            if self._mask is not None:
                gm = gm[self._mask]
            pts = pts[gm]
            colors = colors[gm]

        self.viewer.display(pts, colors, self.controls.size_slider.value())

    def _toggle_bg(self):
        self._dark_bg = not self._dark_bg
        self.viewer.set_bg("#1a1a2e" if self._dark_bg else "#e0e0e0")

    def _view_preset(self, name):
        angles = {"Top": (90, 0), "Front": (0, 0), "Right": (0, 90)}
        el, az = angles.get(name, (0, 0))
        self.viewer.set_camera_angles(el, az)

    # -- processing ---------------------------------------------------------

    def _run_proc(self, op, **kw):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        self.controls.set_loading(True)
        self.status.setText("Running {}...".format(op))
        self._worker = ProcessingThread(op, pts, **kw)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._on_proc_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_proc_done(self, new_mask):
        self.controls.set_loading(False)
        if self._mask is not None:
            full = self._mask.copy()
            active_idx = np.where(full)[0]
            full[active_idx[~new_mask]] = False
            self._mask = full
        else:
            self._mask = new_mask
        kept = int(self._mask.sum())
        removed = self._cloud["displayed_points"] - kept
        self._render()
        self.status.setText("Done - {:,} kept, {:,} removed".format(kept, removed))

    def _on_proc_err(self, msg):
        self.controls.set_loading(False)
        self.status.setText("Processing failed")
        QMessageBox.critical(self, "Processing Error", msg)

    # -- ground seg ---------------------------------------------------------

    def _toggle_nonground(self, hide):
        self._hide_nonground = hide
        self._render()

    def _run_ground_seg(self, cell_size, height_threshold):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        self.controls.set_loading(True)
        self.status.setText("Running ground segmentation...")
        self._worker = ProcessingThread("ground", pts, cell_size=cell_size, height_threshold=height_threshold)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._on_ground_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_ground_done(self, ground_mask):
        self.controls.set_loading(False)
        if self._mask is not None:
            full_ground = np.zeros(self._cloud["displayed_points"], dtype=bool)
            active_idx = np.where(self._mask)[0]
            full_ground[active_idx[ground_mask]] = True
            self._ground_mask = full_ground
        else:
            self._ground_mask = ground_mask

        n_g = int(ground_mask.sum())
        n_t = len(ground_mask)
        self.controls.ground_info.setText(
            "Ground: {:,} ({:.1f}%) | Non-ground: {:,}".format(n_g, 100.0 * n_g / max(n_t, 1), n_t - n_g)
        )
        self.status.setText("Ground seg done - {:,} ground, {:,} non-ground".format(n_g, n_t - n_g))

        self.controls.hide_nonground_cb.setEnabled(True)
        self._add_color_mode("Ground")
        self.controls.color_combo.setCurrentText("Ground")
        self._render()

    # -- satellite texture --------------------------------------------------

    def _run_satellite(self):
        if not self._cloud:
            return
        if self._sat_worker and self._sat_worker.isRunning():
            return

        epsg_text = self.controls.epsg_input.text().strip()
        epsg_override = None
        if epsg_text:
            try:
                epsg_override = int(epsg_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid EPSG", "EPSG must be an integer.")
                return

        self.controls.set_loading(True)
        self.status.setText("Fetching satellite imagery...")
        self.controls.sat_info.setText("Working...")

        self._sat_worker = SatelliteThread(self._cloud, epsg_override)
        self._sat_worker.progress.connect(self.status.setText)
        self._sat_worker.finished.connect(self._on_satellite_done)
        self._sat_worker.error.connect(self._on_satellite_err)
        self._sat_worker.start()

    def _on_satellite_done(self, result):
        self.controls.set_loading(False)
        colors = result["colors"]
        info = result.get("info", {})
        self._sat_colors = colors

        crs_name = info.get("crs_name", "?")
        crs_epsg = info.get("crs_epsg")
        lat_r = info.get("lat_range", (0, 0))
        lon_r = info.get("lon_range", (0, 0))

        diag = "CRS: {}".format(crs_name)
        if crs_epsg:
            diag += " (EPSG:{})".format(crs_epsg)
        diag += "\nLat: {:.5f} to {:.5f}".format(lat_r[0], lat_r[1])
        diag += "\nLon: {:.5f} to {:.5f}".format(lon_r[0], lon_r[1])
        self.controls.sat_info.setText(diag)
        self.controls.sat_info.setStyleSheet("color:#6c6; font-size:11px;")

        self.status.setText("Satellite texture ready - {:,} pts | EPSG:{}".format(
            len(colors), crs_epsg or "?",
        ))

        self._add_color_mode("Satellite")
        self.controls.color_combo.setCurrentText("Satellite")
        self._render()

    def _on_satellite_err(self, msg):
        self.controls.set_loading(False)
        self.controls.sat_info.setText("Failed")
        self.status.setText("Satellite texture failed")
        QMessageBox.warning(self, "Satellite Error", msg)

    # -- reset --------------------------------------------------------------

    def _reset_proc(self):
        self._mask = None
        self._ground_mask = None
        self._hide_nonground = False
        self._sat_colors = None
        self.controls.hide_nonground_cb.blockSignals(True)
        self.controls.hide_nonground_cb.setChecked(False)
        self.controls.hide_nonground_cb.setEnabled(False)
        self.controls.hide_nonground_cb.blockSignals(False)
        self.controls.ground_info.setText("")
        self.controls.sat_info.setText("")

        self.controls.color_combo.blockSignals(True)
        self.controls.color_combo.clear()
        for opt in self._cloud["available_coloring"]:
            self.controls.color_combo.addItem(opt)
        self.controls.color_combo.blockSignals(False)
        self.controls.color_combo.setCurrentText("Elevation")

        self._render()
        self.status.setText("Reset - {:,} points".format(self._cloud["displayed_points"]))

    # -- helpers ------------------------------------------------------------

    def _add_color_mode(self, name):
        items = [self.controls.color_combo.itemText(i)
                 for i in range(self.controls.color_combo.count())]
        if name not in items:
            self.controls.color_combo.addItem(name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = LiDARViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

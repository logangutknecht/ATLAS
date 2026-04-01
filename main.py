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
    QMessageBox, QSpinBox, QLineEdit, QToolBar, QDialog,
    QDialogButtonBox, QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QAction, QActionGroup, QPainter, QPen, QFont, QColor

from vispy import scene
from vispy.color import Colormap, get_colormap

from ProcessingTechniques.geo import (
    extract_crs_wkt, extract_epsg_from_geokeys, build_crs,
    describe_coord_ranges,
)
from ProcessingTechniques.filters import (
    statistical_outlier_removal, voxel_downsample,
    elevation_clip, ground_segmentation, detect_ground_anomalies,
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
QToolBar { background-color: #252525; border-bottom: 1px solid #333; }
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
            elif self.op == "anomaly":
                mask = detect_ground_anomalies(
                    self.pts, self.kw["ground_mask"], self.kw["cell_size"],
                    self.kw["threshold"], self.kw["radius"], pfn)
            else:
                raise ValueError("Unknown op: " + self.op)
            self.finished.emit(mask)
        except Exception as exc:
            self.error.emit(str(exc))


class SatelliteThread(QThread):
    """Fetch satellite imagery and colorize every point in the cloud."""
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


# ---------------------------------------------------------------------------
# Custom camera: right-click drag = pan
# ---------------------------------------------------------------------------

class _PanCamera(scene.TurntableCamera):
    """TurntableCamera with right-click drag mapped to pan."""

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
# Processing parameter dialogs
# ---------------------------------------------------------------------------

class _SORDialog(QDialog):
    def __init__(self, parent=None, k=20, std=2.0):
        super().__init__(parent)
        self.setWindowTitle("Statistical Outlier Removal")
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("K neighbours:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(5, 200)
        self.k_spin.setValue(k)
        row.addWidget(self.k_spin)
        row.addWidget(QLabel("Std ratio:"))
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(0.1, 10.0)
        self.std_spin.setValue(std)
        self.std_spin.setSingleStep(0.1)
        row.addWidget(self.std_spin)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class _VoxelDialog(QDialog):
    def __init__(self, parent=None, size=0.10):
        super().__init__(parent)
        self.setWindowTitle("Voxel Downsample")
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Voxel size:"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.001, 50.0)
        self.size_spin.setValue(size)
        self.size_spin.setSingleStep(0.01)
        self.size_spin.setDecimals(3)
        row.addWidget(self.size_spin)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class _ElevClipDialog(QDialog):
    def __init__(self, parent=None, z_min=0.0, z_max=1000.0):
        super().__init__(parent)
        self.setWindowTitle("Elevation Clip")
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Z min:"))
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(-100000, 100000)
        self.z_min_spin.setDecimals(2)
        self.z_min_spin.setValue(z_min)
        row.addWidget(self.z_min_spin)
        row.addWidget(QLabel("Z max:"))
        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(-100000, 100000)
        self.z_max_spin.setDecimals(2)
        self.z_max_spin.setValue(z_max)
        row.addWidget(self.z_max_spin)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class _GroundSegDialog(QDialog):
    def __init__(self, parent=None, cell=1.0, ht=0.3):
        super().__init__(parent)
        self.setWindowTitle("Ground Segmentation")
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell size:"))
        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(0.1, 20.0)
        self.cell_spin.setValue(cell)
        self.cell_spin.setSingleStep(0.1)
        self.cell_spin.setDecimals(1)
        row.addWidget(self.cell_spin)
        row.addWidget(QLabel("Height threshold:"))
        self.ht_spin = QDoubleSpinBox()
        self.ht_spin.setRange(0.05, 10.0)
        self.ht_spin.setValue(ht)
        self.ht_spin.setSingleStep(0.05)
        self.ht_spin.setDecimals(2)
        row.addWidget(self.ht_spin)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class _AnomalyDialog(QDialog):
    def __init__(self, parent=None, cell=0.5, threshold=0.10, radius=3.0):
        super().__init__(parent)
        self.setWindowTitle("Detect Ground Anomalies")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Detects bumps (rocks, mounds) and dips (potholes, craters)\n"
            "by comparing ground points to a smoothed reference surface."
        ))
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Cell size:"))
        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(0.05, 20.0)
        self.cell_spin.setValue(cell)
        self.cell_spin.setSingleStep(0.1)
        self.cell_spin.setDecimals(2)
        r1.addWidget(self.cell_spin)
        r1.addWidget(QLabel("Threshold:"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 50.0)
        self.thresh_spin.setValue(threshold)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setDecimals(2)
        self.thresh_spin.setToolTip("Minimum deviation from the reference surface")
        r1.addWidget(self.thresh_spin)
        lay.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Smoothing radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.5, 100.0)
        self.radius_spin.setValue(radius)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setToolTip(
            "Features smaller than ~2x this radius are flagged as anomalies"
        )
        r2.addWidget(self.radius_spin)
        lay.addLayout(r2)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class _SatelliteDialog(QDialog):
    def __init__(self, parent=None, info="", epsg=""):
        super().__init__(parent)
        self.setWindowTitle("Apply Satellite Texture")
        lay = QVBoxLayout(self)
        self.info_label = QLabel(info or "CRS: unknown")
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("EPSG:"))
        self.epsg_input = QLineEdit()
        self.epsg_input.setPlaceholderText("e.g. 32617")
        self.epsg_input.setText(epsg)
        row.addWidget(self.epsg_input)
        lay.addLayout(row)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


# ---------------------------------------------------------------------------
# Viewport overlays
# ---------------------------------------------------------------------------

class _ViewCube(QWidget):
    """Camera orientation widget overlaid in the top-right corner."""
    view_angle = pyqtSignal(float, float)

    _BTN = (
        "QPushButton { background:rgba(30,30,30,180); color:#ccc;"
        " border:1px solid #555; border-radius:3px;"
        " font:bold 9px; padding:0; min-width:0; }"
        "QPushButton:hover { background:rgba(0,120,212,200); color:white; }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(134, 100)
        self.setStyleSheet(self._BTN)
        grid = QGridLayout(self)
        grid.setSpacing(2)
        grid.setContentsMargins(2, 2, 2, 2)
        for label, row, col, el, az in [
            ("Top",    0, 1,  90,   0),
            ("Left",   1, 0,   0, -90),
            ("Front",  1, 1,   0,   0),
            ("Right",  1, 2,   0,  90),
            ("Bottom", 2, 1, -90,   0),
        ]:
            b = QPushButton(label)
            b.setFixedSize(40, 24)
            b.clicked.connect(lambda _, e=el, a=az: self.view_angle.emit(float(e), float(a)))
            grid.addWidget(b, row, col, Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()


class _ScaleBar(QWidget):
    """Distance reference bar painted in the bottom-left corner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 40)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._text = ""
        self._bar_px = 0

    def refresh(self, camera, viewport_w):
        dist = getattr(camera, "distance", None) or 500.0
        fov = getattr(camera, "fov", None) or 45.0
        half_span = dist * np.tan(np.radians(fov / 2.0))
        m_per_px = (2.0 * half_span) / max(viewport_w, 1)
        target = m_per_px * 140
        nices = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
                 100, 200, 500, 1000, 2000, 5000, 10000]
        bar_m = min(nices, key=lambda v: abs(v - target))
        self._bar_px = max(20, min(int(bar_m / max(m_per_px, 1e-12)), 180))
        if bar_m >= 1:
            self._text = "{:g} m".format(bar_m)
        else:
            self._text = "{:g} cm".format(bar_m * 100)
        self.update()

    def paintEvent(self, _event):
        if not self._bar_px:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, y = 10, 28
        x1 = x0 + self._bar_px
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(x0, y, x1, y)
        p.drawLine(x0, y - 5, x0, y + 5)
        p.drawLine(x1, y - 5, x1, y + 5)
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(x0, y - 8, self._text)
        p.end()


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

        self.view_cube = _ViewCube(self)
        self.scale_bar = _ScaleBar(self)
        self.canvas.events.draw.connect(self._on_draw)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_overlays()

    def _place_overlays(self):
        w, h = self.width(), self.height()
        self.view_cube.move(w - self.view_cube.width() - 10, 10)
        self.scale_bar.move(10, h - self.scale_bar.height() - 10)
        self.view_cube.raise_()
        self.scale_bar.raise_()

    def _on_draw(self, _ev=None):
        self.scale_bar.refresh(self.view.camera, self.canvas.size[0])

    def display(self, points, colors, size=2):
        self._pts, self._colors, self._size = points, colors, size
        self._scatter.set_data(pos=points, face_color=colors, size=size, edge_width=0)
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
# Side panel  (file section only)
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    file_requested = pyqtSignal()
    max_points_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setSpacing(6)
        col.setContentsMargins(8, 8, 8, 8)

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

        lay.addWidget(QLabel("Max Points (millions):"))
        row = QHBoxLayout()
        self.max_pts_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_pts_slider.setRange(1, 100)
        self.max_pts_slider.setValue(10)
        self.max_pts_label = QLabel("10 M")
        self.max_pts_slider.valueChanged.connect(
            lambda v: self.max_pts_label.setText("{} M".format(v))
        )
        row.addWidget(self.max_pts_slider)
        row.addWidget(self.max_pts_label)
        lay.addLayout(row)

        self.reload_btn = QPushButton("Reload with New Limit")
        self.reload_btn.setEnabled(False)
        self.reload_btn.clicked.connect(
            lambda: self.max_points_changed.emit(
                self.max_pts_slider.value() * 1_000_000
            )
        )
        lay.addWidget(self.reload_btn)

        col.addWidget(g)
        col.addStretch()
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def set_loading(self, busy):
        self.load_btn.setEnabled(not busy)
        self.progress.setVisible(busy)

    def update_file_info(self, data):
        self.file_info.setText(
            "{}\nTotal: {:,}\nDisplayed: {:,}\nLoaded in {:.1f}s".format(
                os.path.basename(data["filepath"]),
                data["total_points"],
                data["displayed_points"],
                data["load_time"],
            )
        )
        self.file_info.setStyleSheet("color:#bbb; font-size:11px;")
        self.reload_btn.setEnabled(True)


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
        self._anomaly_labels = None        # int8: +1 bump, -1 dip, 0 normal
        self._view_mode = "all"            # "all" | "ground" | "ground_anom"
        self._sat_colors = None
        self._dark_bg = True
        self._filepath = None
        self._loader = None
        self._worker = None
        self._sat_worker = None

        self._sor_k = 20
        self._sor_std = 2.0
        self._voxel_size = 0.10
        self._z_min = 0.0
        self._z_max = 1000.0
        self._ground_cell = 1.0
        self._ground_ht = 0.3
        self._anom_cell = 0.5
        self._anom_thresh = 0.10
        self._anom_radius = 3.0
        self._epsg_text = ""
        self._crs_display = ""

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._wire()

        self._las_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "LiDAR_LAS"
        )
        if not os.path.isdir(self._las_dir):
            self._las_dir = os.path.dirname(os.path.abspath(__file__))

    # -- layout -------------------------------------------------------------

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
        splitter.setSizes([220, 1220])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        self.status = QLabel("Ready  \u2014  Ctrl+O to open a LAS file")
        bar = QStatusBar()
        bar.addPermanentWidget(self.status)
        self.setStatusBar(bar)

    def _build_menus(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        a = QAction("&Open LAS/LAZ\u2026", self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self._open_dialog)
        fm.addAction(a)
        fm.addSeparator()
        a2 = QAction("E&xit", self)
        a2.setShortcut("Ctrl+Q")
        a2.triggered.connect(self.close)
        fm.addAction(a2)

        # View
        vm = mb.addMenu("&View")
        ra = QAction("&Reset Camera", self)
        ra.setShortcut("R")
        ra.triggered.connect(self.viewer.reset_camera)
        vm.addAction(ra)
        vm.addSeparator()
        for label, key, el, az in [
            ("Top", "Ctrl+1", 90, 0),
            ("Front", "Ctrl+2", 0, 0),
            ("Right", "Ctrl+3", 0, 90),
        ]:
            va = QAction("{} View".format(label), self)
            va.setShortcut(key)
            va.triggered.connect(
                lambda _, e=el, a=az: self.viewer.set_camera_angles(e, a)
            )
            vm.addAction(va)

        # Processing
        pm = mb.addMenu("&Processing")
        self._act_sor = QAction("Statistical Outlier Removal\u2026", self)
        self._act_sor.setEnabled(False)
        self._act_sor.triggered.connect(self._dlg_sor)
        pm.addAction(self._act_sor)

        self._act_voxel = QAction("Voxel Downsample\u2026", self)
        self._act_voxel.setEnabled(False)
        self._act_voxel.triggered.connect(self._dlg_voxel)
        pm.addAction(self._act_voxel)

        self._act_elev = QAction("Elevation Clip\u2026", self)
        self._act_elev.setEnabled(False)
        self._act_elev.triggered.connect(self._dlg_elev)
        pm.addAction(self._act_elev)

        pm.addSeparator()

        self._act_ground = QAction("Ground Segmentation\u2026", self)
        self._act_ground.setEnabled(False)
        self._act_ground.triggered.connect(self._dlg_ground)
        pm.addAction(self._act_ground)

        self._act_anomaly = QAction("Detect Ground Anomalies\u2026", self)
        self._act_anomaly.setEnabled(False)
        self._act_anomaly.triggered.connect(self._dlg_anomaly)
        pm.addAction(self._act_anomaly)

        pm.addSeparator()

        self._view_group = QActionGroup(self)
        self._view_group.setExclusive(True)

        self._act_view_all = QAction("Show All Points", self)
        self._act_view_all.setCheckable(True)
        self._act_view_all.setChecked(True)
        self._act_view_all.triggered.connect(lambda: self._set_view_mode("all"))
        self._view_group.addAction(self._act_view_all)
        pm.addAction(self._act_view_all)

        self._act_view_ground = QAction("Show Ground Only", self)
        self._act_view_ground.setCheckable(True)
        self._act_view_ground.setEnabled(False)
        self._act_view_ground.triggered.connect(lambda: self._set_view_mode("ground"))
        self._view_group.addAction(self._act_view_ground)
        pm.addAction(self._act_view_ground)

        self._act_view_anom = QAction("Show Ground + Anomalies", self)
        self._act_view_anom.setCheckable(True)
        self._act_view_anom.setEnabled(False)
        self._act_view_anom.triggered.connect(lambda: self._set_view_mode("ground_anom"))
        self._view_group.addAction(self._act_view_anom)
        pm.addAction(self._act_view_anom)

        pm.addSeparator()

        self._act_sat = QAction("Apply Satellite Texture\u2026", self)
        self._act_sat.setEnabled(False)
        self._act_sat.triggered.connect(self._dlg_satellite)
        pm.addAction(self._act_sat)

        pm.addSeparator()

        self._act_reset = QAction("Reset to Original", self)
        self._act_reset.setEnabled(False)
        self._act_reset.triggered.connect(self._reset_proc)
        pm.addAction(self._act_reset)

    def _build_toolbar(self):
        tb = QToolBar("Display")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setStyleSheet(
            "QToolBar { spacing:6px; padding:2px 6px; }"
            "QToolBar QLabel { color:#aaa; font-size:11px; background:transparent; }"
        )
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("Elevation")
        self.color_combo.setMinimumWidth(110)
        self.color_combo.currentTextChanged.connect(lambda _: self._render())
        tb.addWidget(self.color_combo)

        tb.addSeparator()

        tb.addWidget(QLabel(" Size:"))
        self.size_combo = QComboBox()
        for i in range(1, 11):
            self.size_combo.addItem(str(i))
        self.size_combo.setCurrentIndex(1)
        self.size_combo.setFixedWidth(50)
        self.size_combo.currentTextChanged.connect(self._on_size_change)
        tb.addWidget(self.size_combo)

        tb.addSeparator()

        bg_btn = QPushButton("BG")
        bg_btn.setToolTip("Toggle Background")
        bg_btn.setFixedSize(32, 24)
        bg_btn.setStyleSheet(
            "QPushButton { background:#3c3c3c; border:1px solid #555;"
            " border-radius:3px; color:#ccc; font:bold 10px; }"
            "QPushButton:hover { background:#0078d4; color:white; }"
        )
        bg_btn.clicked.connect(self._toggle_bg)
        tb.addWidget(bg_btn)

        tb.addSeparator()

        self.ground_label = QLabel("")
        self.ground_label.setStyleSheet(
            "color:#999; font-size:11px; background:transparent;"
        )
        tb.addWidget(self.ground_label)

    def _wire(self):
        self.controls.file_requested.connect(self._open_dialog)
        self.controls.max_points_changed.connect(self._reload)
        self.viewer.view_cube.view_angle.connect(self.viewer.set_camera_angles)

    # -- helpers ------------------------------------------------------------

    def _on_size_change(self, text):
        if text.isdigit():
            self.viewer.update_size(int(text))

    def _current_size(self):
        t = self.size_combo.currentText()
        return int(t) if t.isdigit() else 2

    # -- dialog launchers ---------------------------------------------------

    def _dlg_sor(self):
        d = _SORDialog(self, self._sor_k, self._sor_std)
        if d.exec():
            self._sor_k = d.k_spin.value()
            self._sor_std = d.std_spin.value()
            self._run_proc("sor", k=self._sor_k, std_ratio=self._sor_std)

    def _dlg_voxel(self):
        d = _VoxelDialog(self, self._voxel_size)
        if d.exec():
            self._voxel_size = d.size_spin.value()
            self._run_proc("voxel", voxel_size=self._voxel_size)

    def _dlg_elev(self):
        d = _ElevClipDialog(self, self._z_min, self._z_max)
        if d.exec():
            self._z_min = d.z_min_spin.value()
            self._z_max = d.z_max_spin.value()
            self._run_proc("elevation", z_min=self._z_min, z_max=self._z_max)

    def _dlg_ground(self):
        d = _GroundSegDialog(self, self._ground_cell, self._ground_ht)
        if d.exec():
            self._ground_cell = d.cell_spin.value()
            self._ground_ht = d.ht_spin.value()
            self._run_ground_seg(self._ground_cell, self._ground_ht)

    def _dlg_satellite(self):
        d = _SatelliteDialog(self, self._crs_display, self._epsg_text)
        if d.exec():
            self._epsg_text = d.epsg_input.text().strip()
            self._run_satellite()

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
        self._anomaly_labels = None
        self._view_mode = "all"
        self._sat_colors = None
        self.controls.set_loading(False)
        self.controls.update_file_info(data)

        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        for opt in data["available_coloring"]:
            self.color_combo.addItem(opt)
        self.color_combo.blockSignals(False)
        self.color_combo.setCurrentText(
            "RGB" if "RGB" in data["available_coloring"] else "Elevation"
        )

        for act in (self._act_sor, self._act_voxel, self._act_elev,
                     self._act_ground, self._act_sat, self._act_reset):
            act.setEnabled(True)

        z = data["points"][:, 2]
        self._z_min = float(z.min())
        self._z_max = float(z.max())

        crs_info = data.get("crs_info", "")
        crs_epsg = data.get("crs_epsg")
        coord_hint = data.get("coord_hint", "")
        if crs_info:
            self._crs_display = "CRS: " + crs_info
            if crs_epsg:
                self._epsg_text = str(crs_epsg)
        elif coord_hint:
            self._crs_display = "CRS: not in file\n" + coord_hint
            self._epsg_text = ""
        else:
            self._crs_display = "CRS: not found"
            self._epsg_text = ""

        self.ground_label.setText("")
        self._render()
        self.viewer.reset_camera()
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
        mode = self.color_combo.currentText()

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

            if (self._anomaly_labels is not None
                    and self._view_mode == "ground_anom"):
                al = self._anomaly_labels
                if self._mask is not None:
                    al = al[self._mask]
                colors[al > 0] = [0.95, 0.55, 0.10, 1.0]   # bump = orange
                colors[al < 0] = [0.20, 0.40, 0.95, 1.0]   # dip  = blue
        else:
            colors = compute_colors(pts, attrs, mode)

        if self._view_mode in ("ground", "ground_anom") and self._ground_mask is not None:
            gm = self._ground_mask
            if self._mask is not None:
                gm = gm[self._mask]
            pts = pts[gm]
            colors = colors[gm]

        self.viewer.display(pts, colors, self._current_size())

    def _toggle_bg(self):
        self._dark_bg = not self._dark_bg
        self.viewer.set_bg("#1a1a2e" if self._dark_bg else "#e0e0e0")

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
        self.status.setText("Done \u2014 {:,} kept, {:,} removed".format(kept, removed))

    def _on_proc_err(self, msg):
        self.controls.set_loading(False)
        self.status.setText("Processing failed")
        QMessageBox.critical(self, "Processing Error", msg)

    # -- ground seg & anomaly -----------------------------------------------

    def _set_view_mode(self, mode):
        self._view_mode = mode
        self._render()

    def _run_ground_seg(self, cell_size, height_threshold):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        self.controls.set_loading(True)
        self.status.setText("Running ground segmentation...")
        self._worker = ProcessingThread(
            "ground", pts,
            cell_size=cell_size, height_threshold=height_threshold,
        )
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
        self.ground_label.setText(
            "Ground: {:,} ({:.1f}%) | Non-ground: {:,}".format(
                n_g, 100.0 * n_g / max(n_t, 1), n_t - n_g
            )
        )
        self.status.setText(
            "Ground seg done \u2014 {:,} ground, {:,} non-ground".format(
                n_g, n_t - n_g
            )
        )

        self._act_anomaly.setEnabled(True)
        self._act_view_ground.setEnabled(True)
        self._act_view_anom.setEnabled(True)
        self._add_color_mode("Ground")
        self.color_combo.setCurrentText("Ground")
        self._render()

    # -- ground anomaly detection -------------------------------------------

    def _dlg_anomaly(self):
        if self._ground_mask is None:
            QMessageBox.information(
                self, "Ground Required",
                "Run Ground Segmentation first."
            )
            return
        d = _AnomalyDialog(self, self._anom_cell, self._anom_thresh,
                            self._anom_radius)
        if d.exec():
            self._anom_cell = d.cell_spin.value()
            self._anom_thresh = d.thresh_spin.value()
            self._anom_radius = d.radius_spin.value()
            self._run_anomaly(self._anom_cell, self._anom_thresh,
                              self._anom_radius)

    def _run_anomaly(self, cell_size, threshold, radius):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        gm = self._ground_mask
        if self._mask is not None:
            gm = gm[self._mask]
        self.controls.set_loading(True)
        self.status.setText("Detecting ground anomalies...")
        self._worker = ProcessingThread(
            "anomaly", pts,
            ground_mask=gm, cell_size=cell_size,
            threshold=threshold, radius=radius,
        )
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._on_anomaly_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_anomaly_done(self, labels):
        self.controls.set_loading(False)
        if self._mask is not None:
            full = np.zeros(self._cloud["displayed_points"], dtype=np.int8)
            active_idx = np.where(self._mask)[0]
            full[active_idx] = labels
            self._anomaly_labels = full
        else:
            self._anomaly_labels = labels

        n_bump = int((labels == 1).sum())
        n_dip = int((labels == -1).sum())
        n_g = int(self._ground_mask.sum()) if self._ground_mask is not None else 0
        self.ground_label.setText(
            "Ground: {:,} | Bumps: {:,} | Dips: {:,}".format(
                n_g, n_bump, n_dip
            )
        )
        self.status.setText(
            "Anomaly detection done \u2014 {:,} bumps + {:,} dips".format(
                n_bump, n_dip
            )
        )

        self._act_view_anom.setEnabled(True)
        self._view_mode = "ground_anom"
        self._act_view_anom.setChecked(True)
        self._add_color_mode("Ground")
        self.color_combo.setCurrentText("Ground")
        self._render()

    # -- satellite texture --------------------------------------------------

    def _run_satellite(self):
        if not self._cloud:
            return
        if self._sat_worker and self._sat_worker.isRunning():
            return

        epsg_override = None
        if self._epsg_text:
            try:
                epsg_override = int(self._epsg_text)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid EPSG", "EPSG must be an integer."
                )
                return

        self.controls.set_loading(True)
        self.status.setText("Fetching satellite imagery...")

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

        crs_epsg = info.get("crs_epsg")
        self.status.setText(
            "Satellite texture ready \u2014 {:,} pts | EPSG:{}".format(
                len(colors), crs_epsg or "?"
            )
        )

        self._add_color_mode("Satellite")
        self.color_combo.setCurrentText("Satellite")
        self._render()

    def _on_satellite_err(self, msg):
        self.controls.set_loading(False)
        self.status.setText("Satellite texture failed")
        QMessageBox.warning(self, "Satellite Error", msg)

    # -- reset --------------------------------------------------------------

    def _reset_proc(self):
        self._mask = None
        self._ground_mask = None
        self._anomaly_labels = None
        self._view_mode = "all"
        self._sat_colors = None

        self._act_view_all.setChecked(True)
        self._act_view_ground.setEnabled(False)
        self._act_view_anom.setEnabled(False)
        self._act_anomaly.setEnabled(False)

        self.ground_label.setText("")

        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        for opt in self._cloud["available_coloring"]:
            self.color_combo.addItem(opt)
        self.color_combo.blockSignals(False)
        self.color_combo.setCurrentText("Elevation")

        self._render()
        self.status.setText(
            "Reset \u2014 {:,} points".format(self._cloud["displayed_points"])
        )

    # -- helpers ------------------------------------------------------------

    def _add_color_mode(self, name):
        items = [self.color_combo.itemText(i)
                 for i in range(self.color_combo.count())]
        if name not in items:
            self.color_combo.addItem(name)


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

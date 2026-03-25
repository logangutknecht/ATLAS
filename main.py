"""
LiDAR Point Cloud Viewer

High-performance GPU-accelerated point cloud viewer built on vispy (OpenGL)
and PyQt6.  Designed for interactive exploration and post-processing of
LAS / LAZ files with tens of millions of points.

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
from scipy.spatial import cKDTree

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QFileDialog, QGroupBox,
    QDoubleSpinBox, QProgressBar, QStatusBar, QSplitter, QScrollArea,
    QMessageBox, QSpinBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction

from vispy import scene
from vispy.color import Colormap, get_colormap

# ---------------------------------------------------------------------------
# Constants
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
    0: (0.50, 0.50, 0.50, 1.0),   # Never classified
    1: (0.60, 0.60, 0.60, 1.0),   # Unclassified
    2: (0.65, 0.45, 0.20, 1.0),   # Ground
    3: (0.45, 0.75, 0.30, 1.0),   # Low vegetation
    4: (0.20, 0.65, 0.15, 1.0),   # Medium vegetation
    5: (0.05, 0.45, 0.05, 1.0),   # High vegetation
    6: (0.90, 0.25, 0.20, 1.0),   # Building
    7: (0.55, 0.15, 0.55, 1.0),   # Low point (noise)
    8: (0.70, 0.70, 0.70, 1.0),   # Reserved / Model key
    9: (0.15, 0.40, 0.90, 1.0),   # Water
    10: (0.35, 0.70, 0.85, 1.0),  # Rail
    11: (0.40, 0.40, 0.40, 1.0),  # Road surface
    17: (0.90, 0.80, 0.20, 1.0),  # Bridge deck
    18: (0.80, 0.50, 0.20, 1.0),  # High noise
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

def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    rng = hi - lo
    return (values - lo) / rng if rng > 0 else np.zeros_like(values)


def compute_colors(
    points: np.ndarray,
    attrs: dict,
    mode: str,
) -> np.ndarray:
    """Return an (N, 4) float32 RGBA array for the requested colouring mode."""

    if mode == "RGB" and "rgb" in attrs:
        rgb = attrs["rgb"].astype(np.float32) / 255.0
        alpha = np.ones((len(rgb), 1), dtype=np.float32)
        return np.hstack([rgb, alpha])

    if mode == "Classification" and "classification" in attrs:
        cls = attrs["classification"].astype(np.int32)
        out = np.full((len(cls), 4), 0.5, dtype=np.float32)
        out[:, 3] = 1.0
        for code, rgba in ASPRS_COLORS.items():
            mask = cls == code
            if mask.any():
                out[mask] = rgba
        return out

    if mode == "Return Number" and "return_number" in attrs:
        rn = attrs["return_number"].astype(np.int32)
        idx = np.clip(rn - 1, 0, len(RETURN_PALETTE) - 1)
        return RETURN_PALETTE[idx]

    if mode == "Intensity" and "intensity" in attrs:
        normed = _normalize(attrs["intensity"])
        cmap = get_colormap("grays")
        return cmap.map(normed).astype(np.float32)

    # Default: elevation
    normed = _normalize(points[:, 2])
    return TURBO_CMAP.map(normed).astype(np.float32)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class LASLoaderThread(QThread):
    """Read a LAS/LAZ file off the UI thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath: str, max_points=None):
        super().__init__()
        self.filepath = filepath
        self.max_points = max_points

    def run(self):
        try:
            name = os.path.basename(self.filepath)
            self.progress.emit(f"Reading {name}…")
            t0 = time.time()

            las = laspy.read(self.filepath)
            total = las.header.point_count

            self.progress.emit(f"Processing {total:,} points…")

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

            data: dict = {
                "points": pts,
                "centroid": centroid,
                "total_points": total,
                "displayed_points": len(pts),
                "filepath": self.filepath,
                "load_time": time.time() - t0,
                "attributes": {},
                "available_coloring": ["Elevation"],
            }

            if hasattr(las, "intensity"):
                data["attributes"]["intensity"] = sub(
                    np.asarray(las.intensity, dtype=np.float32)
                )
                data["available_coloring"].append("Intensity")

            if hasattr(las, "classification"):
                data["attributes"]["classification"] = sub(
                    np.asarray(las.classification, dtype=np.float32)
                )
                data["available_coloring"].append("Classification")

            if hasattr(las, "return_number"):
                data["attributes"]["return_number"] = sub(
                    np.asarray(las.return_number, dtype=np.float32)
                )
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
            self.error.emit(
                "Not enough memory – reduce the Max Points slider and retry."
            )
        except Exception as exc:
            self.error.emit(str(exc))


class ProcessingThread(QThread):
    """Run a post-processing filter off the UI thread."""

    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, operation: str, points: np.ndarray, **kwargs):
        super().__init__()
        self.op = operation
        self.pts = points
        self.kw = kwargs

    def run(self):
        try:
            mask = getattr(self, f"_{self.op}")()
            self.finished.emit(mask)
        except Exception as exc:
            self.error.emit(str(exc))

    def _sor(self):
        k, ratio = self.kw["k"], self.kw["std_ratio"]
        self.progress.emit(f"SOR: KD-tree on {len(self.pts):,} pts…")
        tree = cKDTree(self.pts)
        self.progress.emit("SOR: neighbour distances…")
        d, _ = tree.query(self.pts, k=k + 1)
        mean_d = d[:, 1:].mean(axis=1)
        return mean_d <= (mean_d.mean() + ratio * mean_d.std())

    def _voxel(self):
        vs = self.kw["voxel_size"]
        self.progress.emit("Voxel downsampling…")
        keys = np.floor(self.pts / vs).astype(np.int32)
        _, idx = np.unique(keys, axis=0, return_index=True)
        mask = np.zeros(len(self.pts), dtype=bool)
        mask[idx] = True
        return mask

    def _elevation(self):
        z = self.pts[:, 2]
        return (z >= self.kw["z_min"]) & (z <= self.kw["z_max"])


# ---------------------------------------------------------------------------
# 3-D viewer widget (vispy)
# ---------------------------------------------------------------------------

class PointCloudView(QWidget):
    """Thin wrapper around a vispy SceneCanvas for point-cloud display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=False, bgcolor="#1a1a2e",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=45, distance=500)

        self._scatter = scene.visuals.Markers(parent=self.view.scene)
        self._axis = scene.visuals.XYZAxis(parent=self.view.scene)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas.native)

        self._pts = None
        self._colors = None
        self._size = 2

    def display(self, points: np.ndarray, colors: np.ndarray, size: int = 2):
        self._pts = points
        self._colors = colors
        self._size = size
        self._scatter.set_data(
            pos=points,
            face_color=colors,
            size=size,
            edge_width=0,
        )
        self.view.camera.set_range()
        self.canvas.update()

    def update_size(self, size: int):
        if self._pts is None:
            return
        self._size = size
        self._scatter.set_data(
            pos=self._pts,
            face_color=self._colors,
            size=size,
            edge_width=0,
        )
        self.canvas.update()

    def set_bg(self, color: str):
        self.canvas.bgcolor = color

    def reset_camera(self):
        self.view.camera.set_range()
        self.canvas.update()

    def set_camera_angles(self, elevation: float, azimuth: float):
        self.view.camera.elevation = elevation
        self.view.camera.azimuth = azimuth
        self.canvas.update()

    def clear(self):
        self._pts = None
        self._colors = None
        self._scatter.set_data(pos=np.zeros((1, 3), dtype=np.float32))
        self.canvas.update()


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    """Sidebar with file-loading, display, and processing controls."""

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
    reset_processing = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(290)
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

        col.addWidget(self._file_section())
        col.addWidget(self._display_section())
        col.addWidget(self._perf_section())
        col.addWidget(self._camera_section())
        col.addWidget(self._processing_section())
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
        self.size_slider.valueChanged.connect(
            lambda v: self.size_label.setText(str(v))
        )
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
        self.max_pts_slider.valueChanged.connect(
            lambda v: self.max_pts_label.setText(f"{v} M")
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
        self.sor_std.setRange(0.1, 10.0); self.sor_std.setValue(2.0)
        self.sor_std.setSingleStep(0.1)
        r.addWidget(self.sor_std)
        lay.addLayout(r)
        self.sor_btn = QPushButton("Remove Outliers")
        self.sor_btn.setEnabled(False)
        self.sor_btn.clicked.connect(
            lambda: self.sor_requested.emit(
                self.sor_k.value(), self.sor_std.value()
            )
        )
        lay.addWidget(self.sor_btn)

        # Voxel
        lay.addWidget(QLabel("Voxel Downsampling"))
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Size:"))
        self.voxel_size = QDoubleSpinBox()
        self.voxel_size.setRange(0.001, 50.0)
        self.voxel_size.setValue(0.10); self.voxel_size.setSingleStep(0.01)
        self.voxel_size.setDecimals(3)
        r2.addWidget(self.voxel_size)
        lay.addLayout(r2)
        self.voxel_btn = QPushButton("Downsample")
        self.voxel_btn.setEnabled(False)
        self.voxel_btn.clicked.connect(
            lambda: self.voxel_requested.emit(self.voxel_size.value())
        )
        lay.addWidget(self.voxel_btn)

        # Elevation clip
        lay.addWidget(QLabel("Elevation Clip"))
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Z min:"))
        self.z_min = QDoubleSpinBox()
        self.z_min.setRange(-100000, 100000); self.z_min.setDecimals(2)
        r3.addWidget(self.z_min)
        r3.addWidget(QLabel("Z max:"))
        self.z_max = QDoubleSpinBox()
        self.z_max.setRange(-100000, 100000); self.z_max.setValue(1000)
        self.z_max.setDecimals(2)
        r3.addWidget(self.z_max)
        lay.addLayout(r3)
        self.elev_btn = QPushButton("Clip Elevation")
        self.elev_btn.setEnabled(False)
        self.elev_btn.clicked.connect(
            lambda: self.elev_requested.emit(
                self.z_min.value(), self.z_max.value()
            )
        )
        lay.addWidget(self.elev_btn)

        # Reset
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setStyleSheet("background-color:#c42b1c; color:white;")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_processing.emit)
        lay.addWidget(self.reset_btn)
        return g

    # -- public helpers -----------------------------------------------------

    def set_loading(self, busy: bool):
        self.load_btn.setEnabled(not busy)
        self.progress.setVisible(busy)

    def update_for_data(self, data: dict):
        self.file_info.setText(
            f"{os.path.basename(data['filepath'])}\n"
            f"Total: {data['total_points']:,}\n"
            f"Displayed: {data['displayed_points']:,}\n"
            f"Loaded in {data['load_time']:.1f}s"
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
                   self.reset_btn, self.reload_btn):
            b.setEnabled(True)

        z = data["points"][:, 2]
        self.z_min.setValue(float(z.min()))
        self.z_max.setValue(float(z.max()))


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
        self._dark_bg = True
        self._filepath = None
        self._loader = None
        self._worker = None

        self._build_ui()
        self._build_menu()
        self._wire_signals()

        self._las_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "LiDAR_LAS",
        )
        if not os.path.isdir(self._las_dir):
            self._las_dir = os.path.dirname(os.path.abspath(__file__))

    # -- construction -------------------------------------------------------

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

        self.status = QLabel("Ready  —  Ctrl+O to open a LAS file")
        bar = QStatusBar()
        bar.addPermanentWidget(self.status)
        self.setStatusBar(bar)

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        a = QAction("&Open LAS/LAZ…", self)
        a.setShortcut("Ctrl+O")
        a.triggered.connect(self._open_dialog)
        fm.addAction(a)
        fm.addSeparator()
        a2 = QAction("E&xit", self)
        a2.setShortcut("Ctrl+Q")
        a2.triggered.connect(self.close)
        fm.addAction(a2)

        vm = mb.addMenu("&View")
        ra = QAction("&Reset Camera", self)
        ra.setShortcut("R")
        ra.triggered.connect(lambda: self.viewer.reset_camera())
        vm.addAction(ra)
        vm.addSeparator()
        for label, key, el, az in [
            ("Top", "Ctrl+1", 90, 0),
            ("Front", "Ctrl+2", 0, 0),
            ("Right", "Ctrl+3", 0, 90),
        ]:
            va = QAction(f"{label} View", self)
            va.setShortcut(key)
            va.triggered.connect(
                lambda _, e=el, a=az: self.viewer.set_camera_angles(e, a)
            )
            vm.addAction(va)

    def _wire_signals(self):
        c = self.controls
        c.file_requested.connect(self._open_dialog)
        c.color_changed.connect(self._recolor)
        c.point_size_changed.connect(self.viewer.update_size)
        c.bg_toggled.connect(self._toggle_bg)
        c.reset_camera.connect(self.viewer.reset_camera)
        c.view_preset.connect(self._view_preset)
        c.max_points_changed.connect(self._reload)
        c.sor_requested.connect(
            lambda k, s: self._run_proc("sor", k=k, std_ratio=s)
        )
        c.voxel_requested.connect(
            lambda v: self._run_proc("voxel", voxel_size=v)
        )
        c.elev_requested.connect(
            lambda lo, hi: self._run_proc("elevation", z_min=lo, z_max=hi)
        )
        c.reset_processing.connect(self._reset_proc)

    # -- file I/O -----------------------------------------------------------

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", self._las_dir,
            "LAS files (*.las *.laz);;All files (*)",
        )
        if path:
            self._las_dir = os.path.dirname(path)
            max_pts = self.controls.max_pts_slider.value() * 1_000_000
            self._load(path, max_pts)

    def _load(self, path: str, max_pts: int):
        if self._loader and self._loader.isRunning():
            return
        self._filepath = path
        self.controls.set_loading(True)
        self.status.setText(f"Loading {os.path.basename(path)}…")

        self._loader = LASLoaderThread(path, max_pts)
        self._loader.progress.connect(self.status.setText)
        self._loader.finished.connect(self._on_loaded)
        self._loader.error.connect(self._on_load_err)
        self._loader.start()

    def _on_loaded(self, data: dict):
        self._cloud = data
        self._mask = None
        self.controls.set_loading(False)
        self.controls.update_for_data(data)
        self._render()
        self.status.setText(
            f"{data['displayed_points']:,} / {data['total_points']:,} pts  "
            f"| {data['load_time']:.1f}s"
        )

    def _on_load_err(self, msg: str):
        self.controls.set_loading(False)
        self.status.setText("Load failed")
        QMessageBox.critical(self, "Error", msg)

    def _reload(self, max_pts: int):
        if self._filepath:
            self._load(self._filepath, max_pts)

    # -- rendering ----------------------------------------------------------

    def _active_data(self):
        """Return (points, attributes) respecting the current mask."""
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
        colors = compute_colors(pts, attrs, mode)
        size = self.controls.size_slider.value()
        self.viewer.display(pts, colors, size)

    def _recolor(self, _mode: str):
        self._render()

    def _toggle_bg(self):
        self._dark_bg = not self._dark_bg
        self.viewer.set_bg("#1a1a2e" if self._dark_bg else "#e0e0e0")

    def _view_preset(self, name: str):
        angles = {"Top": (90, 0), "Front": (0, 0), "Right": (0, 90)}
        el, az = angles.get(name, (0, 0))
        self.viewer.set_camera_angles(el, az)

    # -- processing ---------------------------------------------------------

    def _run_proc(self, op: str, **kw):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        self.controls.set_loading(True)
        self.status.setText(f"Running {op}…")

        self._worker = ProcessingThread(op, pts, **kw)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._on_proc_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_proc_done(self, new_mask: np.ndarray):
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
        self.status.setText(f"Done — {kept:,} kept, {removed:,} removed")

    def _on_proc_err(self, msg: str):
        self.controls.set_loading(False)
        self.status.setText("Processing failed")
        QMessageBox.critical(self, "Processing Error", msg)

    def _reset_proc(self):
        self._mask = None
        self._render()
        self.status.setText(
            f"Reset — {self._cloud['displayed_points']:,} points"
        )


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

"""Photogrammetry viewport widgets.

ThumbnailGrid   – scrollable, clickable thumbnail strip
ImageViewer     – full-resolution single-image display (fit to window)
FlightMapView   – 2-D overhead map of camera positions and footprints
OrthoView       – orthomosaic result display (reuses ImageViewer logic)
"""

import os

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QGridLayout, QPushButton, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsPolygonItem, QStackedWidget,
    QTabBar, QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QFont, QColor, QPen, QBrush, QPolygonF


_THUMB_BTN_STYLE = (
    "QPushButton {"
    "  background: #2a2a2a; border: 2px solid #444;"
    "  border-radius: 4px; padding: 2px;"
    "}"
    "QPushButton:hover { border-color: #0078d4; }"
    "QPushButton:checked { border-color: #0078d4; background: #1a3a5c; }"
)

_TAB_BAR_STYLE = (
    "QTabBar::tab {"
    "  background: #252525; color: #888; border: none;"
    "  padding: 6px 14px; font-size: 11px;"
    "}"
    "QTabBar::tab:selected { background: #2d2d2d; color: #fff; border-bottom: 2px solid #0078d4; }"
    "QTabBar::tab:hover { background: #333; color: #ccc; }"
)


# ---------------------------------------------------------------------------
# Thumbnail grid
# ---------------------------------------------------------------------------

class ThumbnailGrid(QScrollArea):
    """Scrollable grid of clickable image thumbnails."""

    image_selected = pyqtSignal(str)
    _COLUMNS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(4)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self.setWidget(self._container)

        self._buttons = {}
        self._active_path = None

    def clear(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._buttons.clear()
        self._active_path = None

    def add_thumbnail(self, path, pixmap):
        idx = len(self._buttons)
        row, col = divmod(idx, self._COLUMNS)

        btn = QPushButton()
        btn.setCheckable(True)
        btn.setFixedSize(QSize(120, 100))
        btn.setStyleSheet(_THUMB_BTN_STYLE)
        btn.setIcon(QIcon(pixmap) if isinstance(pixmap, QPixmap) else QIcon())
        btn.setIconSize(QSize(112, 84))
        btn.setToolTip(os.path.basename(path))
        btn.clicked.connect(lambda _, p=path: self._on_clicked(p))
        self._grid.addWidget(btn, row, col)
        self._buttons[path] = btn

    def _on_clicked(self, path):
        if self._active_path and self._active_path in self._buttons:
            self._buttons[self._active_path].setChecked(False)
        self._active_path = path
        self._buttons[path].setChecked(True)
        self.image_selected.emit(path)


# ---------------------------------------------------------------------------
# Single image viewer (fit to window)
# ---------------------------------------------------------------------------

class ImageViewer(QWidget):
    """Displays a single image scaled to fit the available space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #555; font-size: 14px; background: #1a1a2e;")
        self._label.setText("No image selected")
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._label, stretch=1)

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background: #252525; border-bottom: 1px solid #333;")
        row = QHBoxLayout(header)
        row.setContentsMargins(8, 0, 8, 0)
        self._filename_label = QLabel("")
        self._filename_label.setStyleSheet("color:#999; font-size:11px; background:transparent;")
        self._dim_label = QLabel("")
        self._dim_label.setStyleSheet("color:#666; font-size:11px; background:transparent;")
        row.addWidget(self._filename_label)
        row.addStretch()
        row.addWidget(self._dim_label)
        return header

    def show_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._label.setText("Failed to load: {}".format(os.path.basename(path)))
            return
        self._pixmap = pixmap
        self._filename_label.setText(os.path.basename(path))
        self._dim_label.setText("{} × {}".format(pixmap.width(), pixmap.height()))
        self._fit_image()

    def show_pixmap(self, pixmap, label="Orthomosaic"):
        self._pixmap = pixmap
        self._filename_label.setText(label)
        self._dim_label.setText("{} × {}".format(pixmap.width(), pixmap.height()))
        self._fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def _fit_image(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._label.width(), self._label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)


# ---------------------------------------------------------------------------
# Flight map (2-D overhead camera positions + footprints)
# ---------------------------------------------------------------------------

class FlightMapView(QGraphicsView):
    """Overhead map showing camera GPS positions, flight path, and image footprints."""

    camera_selected = pyqtSignal(str)  # emits image path when a camera dot is clicked

    _BG_COLOR         = QColor("#1a1a2e")
    _FOOTPRINT_COLOR  = QColor(255, 255, 255, 18)
    _FOOTPRINT_PEN    = QPen(QColor(255, 255, 255, 50), 0)
    _PATH_PEN         = QPen(QColor(0, 120, 212, 160), 0)
    _DOT_COLOR        = QColor(0, 120, 212)
    _DOT_SELECTED     = QColor(255, 140, 0)
    _DOT_RADIUS       = 4.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(self._BG_COLOR))
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)

        self._metadata   = []
        self._poses      = []
        self._dot_items  = {}   # path -> QGraphicsEllipseItem
        self._meta_by_path = {}  # path -> metadata dict
        self._active_path = None
        self._hover_path  = None
        self._thumb_cache = {}   # path -> QPixmap

        self._preview = self._build_preview_widget()

    def _build_preview_widget(self):
        w = QWidget(self)
        w.setStyleSheet(
            "QWidget { background: #1e1e2e; border: 1px solid #444; border-radius: 4px; }"
        )
        w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        w.hide()

        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        self._preview_thumb = QLabel()
        self._preview_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_thumb.setFixedSize(180, 135)
        self._preview_thumb.setStyleSheet("border: none; background: transparent;")
        lay.addWidget(self._preview_thumb)

        self._preview_caption = QLabel()
        self._preview_caption.setStyleSheet(
            "color: #ccc; font-size: 10px; border: none; background: transparent;"
        )
        self._preview_caption.setWordWrap(True)
        lay.addWidget(self._preview_caption)

        w.setFixedWidth(192)
        return w

    def _show_preview(self, path, viewport_pos):
        meta = self._meta_by_path.get(path)
        if meta is None:
            self._preview.hide()
            return

        if path not in self._thumb_cache:
            pix = QPixmap(path)
            if pix.isNull():
                self._preview.hide()
                return
            self._thumb_cache[path] = pix.scaled(
                180, 135,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._preview_thumb.setPixmap(self._thumb_cache[path])

        gps = meta["gps"]
        gimbal = meta["gimbal"]
        self._preview_caption.setText(
            "{name}\n"
            "{lat:.6f}°, {lon:.6f}°\n"
            "Alt {alt:.0f}m  |  Hdg {hdg:.1f}°".format(
                name=os.path.basename(path),
                lat=gps["latitude"], lon=gps["longitude"],
                alt=gps["altitude_relative"],
                hdg=gimbal["yaw"],
            )
        )
        self._preview.adjustSize()

        x = viewport_pos.x() + 16
        y = viewport_pos.y() + 16
        vw, vh = self.viewport().width(), self.viewport().height()
        pw, ph = self._preview.width(), self._preview.height()
        if x + pw > vw:
            x = viewport_pos.x() - pw - 8
        if y + ph > vh:
            y = viewport_pos.y() - ph - 8
        self._preview.move(x, y)
        self._preview.show()
        self._preview.raise_()

    def _hide_preview(self):
        self._preview.hide()
        self._hover_path = None

    def load(self, all_metadata: list, poses: list):
        self._scene.clear()
        self._dot_items.clear()
        self._thumb_cache.clear()
        self._meta_by_path = {m["path"]: m for m in all_metadata}
        self._metadata = all_metadata
        self._poses    = poses
        self._hide_preview()

        if not poses:
            return

        self._draw_footprints()
        self._draw_flight_path()
        self._draw_camera_dots()
        self.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def highlight_camera(self, path: str):
        """Highlight the dot for the given image path."""
        if self._active_path and self._active_path in self._dot_items:
            self._dot_items[self._active_path].setBrush(QBrush(self._DOT_COLOR))
        self._active_path = path
        if path in self._dot_items:
            self._dot_items[path].setBrush(QBrush(self._DOT_SELECTED))

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        path = item.data(0) if item else None
        if path and path != self._hover_path:
            self._hover_path = path
            self._show_preview(path, event.pos())
        elif path and path == self._hover_path:
            self._show_preview(path, event.pos())
        elif not path and self._hover_path:
            self._hide_preview()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hide_preview()

    # -- drawing helpers ----------------------------------------------------

    def _draw_footprints(self):
        from src.photogrammetry.processing.georef import compute_image_footprint
        for meta, pose in zip(self._metadata, self._poses):
            fp = compute_image_footprint(
                pose, meta["calibration"], meta["image_width"], meta["image_height"]
            )
            poly = QPolygonF([QPointF(pt[0], -pt[1]) for pt in fp])
            item = self._scene.addPolygon(poly, self._FOOTPRINT_PEN, QBrush(self._FOOTPRINT_COLOR))
            item.setZValue(0)

    def _draw_flight_path(self):
        positions = [p["position"] for p in self._poses]
        for i in range(len(positions) - 1):
            x0, y0 = positions[i][0],     -positions[i][1]
            x1, y1 = positions[i + 1][0], -positions[i + 1][1]
            line = self._scene.addLine(x0, y0, x1, y1, self._PATH_PEN)
            line.setZValue(1)

    def _draw_camera_dots(self):
        r = self._DOT_RADIUS
        for meta, pose in zip(self._metadata, self._poses):
            x = pose["position"][0]
            y = -pose["position"][1]   # flip Y so North is up
            dot = self._scene.addEllipse(
                x - r, y - r, r * 2, r * 2,
                QPen(Qt.PenStyle.NoPen), QBrush(self._DOT_COLOR),
            )
            dot.setZValue(2)
            dot.setToolTip(os.path.basename(meta["path"]))
            dot.setData(0, meta["path"])
            self._dot_items[meta["path"]] = dot

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        item = self.itemAt(event.pos())
        if item and item.data(0):
            self.camera_selected.emit(item.data(0))


# ---------------------------------------------------------------------------
# Tabbed viewport (Image | Flight Map | Ortho)
# ---------------------------------------------------------------------------

class PhotogrammetryViewport(QWidget):
    """Tab bar that switches between image viewer, flight map, and ortho output."""

    TABS = ["Image", "Flight Map", "Ortho"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_bar = QTabBar()
        self._tab_bar.setStyleSheet(_TAB_BAR_STYLE)
        self._tab_bar.setExpanding(False)
        for tab in self.TABS:
            self._tab_bar.addTab(tab)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tab_bar)

        self._stack = QStackedWidget()
        self.image_view   = ImageViewer()
        self.flight_map   = FlightMapView()
        self.ortho_view   = ImageViewer()
        self._stack.addWidget(self.image_view)
        self._stack.addWidget(self.flight_map)
        self._stack.addWidget(self.ortho_view)
        layout.addWidget(self._stack, stretch=1)

    def _on_tab_changed(self, index):
        self._stack.setCurrentIndex(index)

    def switch_to(self, tab_name: str):
        idx = self.TABS.index(tab_name)
        self._tab_bar.setCurrentIndex(idx)

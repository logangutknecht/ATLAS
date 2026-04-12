"""Photogrammetry viewport widgets.

ThumbnailGrid   – scrollable, clickable thumbnail strip
ImageViewer     – full-resolution single-image display (fit to window)
FlightMapView   – 2-D overhead map of camera positions and footprints
OrthoViewer     – interactive ortho display with zoom, pan, overlays
"""

import math
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
from PyQt6.QtGui import (
    QPixmap, QIcon, QFont, QColor, QPen, QBrush, QPolygonF,
    QPainter, QPainterPath,
)


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
# Nice scale values for the ortho scale bar
# ---------------------------------------------------------------------------

_SCALE_NICES = [
    0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000,
]


# ---------------------------------------------------------------------------
# Interactive ortho viewer with zoom, pan, lat/lon cursor, compass, scale bar
# ---------------------------------------------------------------------------

class OrthoViewer(QWidget):
    """Zoomable, pannable orthomosaic viewer with map overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #1a1a2e;")
        self.setMouseTracking(True)

        self._pixmap = None
        self._title_text = ""
        self._dim_text = ""

        # Geo-referencing info (set via set_geo)
        self._origin = None      # (lat, lon, alt)
        self._bounds = None      # (xmin, xmax, ymin, ymax) in ENU metres
        self._resolution = None  # m/px

        # View transform state
        self._zoom = 1.0
        self._pan = QPointF(0, 0)   # pixel offset in widget coords
        self._dragging = False
        self._drag_start = QPointF()
        self._pan_start = QPointF()

        # Cursor coordinate label (floating over the viewport)
        self._coord_label = QLabel(self)
        self._coord_label.setStyleSheet(
            "background: rgba(0,0,0,170); color: #ddd; font-size: 11px;"
            " padding: 2px 6px; border-radius: 3px;"
        )
        self._coord_label.hide()

    # -- public API ---------------------------------------------------------

    def show_pixmap(self, pixmap, label="Orthomosaic"):
        self._pixmap = pixmap
        self._title_text = label
        self._dim_text = "{} × {}".format(pixmap.width(), pixmap.height())
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def set_geo(self, origin, bounds, resolution):
        """Store georeferencing info for coordinate display.

        origin     – (lat, lon, alt) ENU origin in WGS84
        bounds     – (xmin, xmax, ymin, ymax) in ENU metres
        resolution – metres per ortho pixel
        """
        self._origin = origin
        self._bounds = bounds
        self._resolution = resolution

    # -- coordinate helpers -------------------------------------------------

    def _widget_to_enu(self, widget_pos):
        """Convert a widget-space position to ENU (east, north) or None."""
        if self._pixmap is None or self._bounds is None:
            return None
        img_rect = self._image_rect()
        if img_rect is None or img_rect.width() == 0 or img_rect.height() == 0:
            return None

        fx = (widget_pos.x() - img_rect.x()) / img_rect.width()
        fy = (widget_pos.y() - img_rect.y()) / img_rect.height()
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            return None

        xmin, xmax, ymin, ymax = self._bounds
        east = xmin + fx * (xmax - xmin)
        north = ymax - fy * (ymax - ymin)
        return east, north

    def _enu_to_latlon(self, east, north):
        """Convert local ENU offset back to approximate WGS84 lat/lon.

        The ortho builder negates E/N positions before computing bounds,
        so we invert them here to recover true geographic coordinates.
        """
        if self._origin is None:
            return None
        lat0, lon0, _alt0 = self._origin
        lat = lat0 + (-north) / 111_320.0
        lon = lon0 + (-east) / (111_320.0 * math.cos(math.radians(lat0)))
        return lat, lon

    # -- image layout -------------------------------------------------------

    def _image_rect(self):
        """Return QRectF of the ortho image in widget coordinates."""
        if self._pixmap is None:
            return None
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw = self.width()
        vh = self.height() - 28  # subtract header
        if pw == 0 or ph == 0 or vw == 0 or vh == 0:
            return None

        # Fit scale (before zoom)
        fit = min(vw / pw, vh / ph)
        w = pw * fit * self._zoom
        h = ph * fit * self._zoom
        x = (vw - w) / 2.0 + self._pan.x()
        y = 28 + (vh - h) / 2.0 + self._pan.y()
        return QRectF(x, y, w, h)

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1a1a2e"))

        # Header
        p.fillRect(0, 0, self.width(), 28, QColor("#252525"))
        p.setPen(QColor("#999"))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(8, 18, self._title_text)
        p.setPen(QColor("#666"))
        tw = p.fontMetrics().horizontalAdvance(self._dim_text)
        p.drawText(self.width() - tw - 8, 18, self._dim_text)

        if self._pixmap is None:
            p.setPen(QColor("#555"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No orthomosaic")
            p.end()
            return

        rect = self._image_rect()
        if rect is not None:
            p.drawPixmap(rect.toRect(), self._pixmap)

        self._draw_compass(p)
        self._draw_scale_bar(p)
        p.end()

    def _draw_compass(self, p):
        """Draw a cardinal-direction compass rose in the bottom-right corner."""
        cx = self.width() - 40
        cy = self.height() - 50
        r = 22

        # Circle background
        p.setBrush(QBrush(QColor(30, 30, 30, 180)))
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.drawEllipse(QPointF(cx, cy), r + 6, r + 6)

        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        p.setFont(font)

        directions = [
            ("N", 0, QColor(220, 60, 60)),
            ("E", 90, QColor(200, 200, 200)),
            ("S", 180, QColor(200, 200, 200)),
            ("W", 270, QColor(200, 200, 200)),
        ]
        for label, angle_deg, color in directions:
            rad = math.radians(angle_deg - 90)
            tx = cx + (r + 1) * math.cos(rad)
            ty = cy + (r + 1) * math.sin(rad)
            p.setPen(color)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            p.drawText(int(tx - tw / 2), int(ty + th / 3), label)

        # Needle
        p.setPen(Qt.PenStyle.NoPen)
        # North (red) half
        north_path = QPainterPath()
        north_path.moveTo(cx, cy - r + 4)
        north_path.lineTo(cx - 3, cy)
        north_path.lineTo(cx + 3, cy)
        north_path.closeSubpath()
        p.setBrush(QBrush(QColor(220, 60, 60)))
        p.drawPath(north_path)
        # South (white) half
        south_path = QPainterPath()
        south_path.moveTo(cx, cy + r - 4)
        south_path.lineTo(cx - 3, cy)
        south_path.lineTo(cx + 3, cy)
        south_path.closeSubpath()
        p.setBrush(QBrush(QColor(200, 200, 200)))
        p.drawPath(south_path)

    def _draw_scale_bar(self, p):
        """Draw a distance scale bar in the bottom-left corner."""
        if self._resolution is None or self._pixmap is None:
            return
        rect = self._image_rect()
        if rect is None or rect.width() == 0:
            return

        pw = self._pixmap.width()
        m_per_widget_px = (pw * self._resolution) / rect.width()

        target = m_per_widget_px * 140
        bar_m = min(_SCALE_NICES, key=lambda v: abs(v - target))
        bar_px = max(20, min(int(bar_m / max(m_per_widget_px, 1e-12)), 180))

        if bar_m >= 1:
            text = "{:g} m".format(bar_m)
        else:
            text = "{:g} cm".format(bar_m * 100)

        x0 = 14
        y = self.height() - 16
        x1 = x0 + bar_px

        # Background
        p.setBrush(QBrush(QColor(0, 0, 0, 140)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(x0 - 6, y - 22, bar_px + 12, 30), 4, 4)

        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(x0, y, x1, y)
        p.drawLine(x0, y - 5, x0, y + 5)
        p.drawLine(x1, y - 5, x1, y + 5)

        p.setFont(QFont("Segoe UI", 9))
        p.drawText(x0, y - 8, text)

    # -- input events -------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        old_zoom = self._zoom
        self._zoom = max(0.1, min(self._zoom * factor, 50.0))
        actual_factor = self._zoom / old_zoom

        # Zoom towards cursor
        mouse = event.position()
        cx = self.width() / 2.0 + self._pan.x()
        cy = (self.height() - 28) / 2.0 + 28 + self._pan.y()
        self._pan.setX(self._pan.x() - (mouse.x() - cx) * (actual_factor - 1))
        self._pan.setY(self._pan.y() - (mouse.y() - cy) * (actual_factor - 1))

        self.update()
        self._update_coord_label(event.position())

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._dragging = True
            self._drag_start = event.position()
            self._pan_start = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.position() - self._drag_start
            self._pan = QPointF(
                self._pan_start.x() + delta.x(),
                self._pan_start.y() + delta.y(),
            )
            self.update()
        self._update_coord_label(event.position())

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event):
        self._coord_label.hide()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def _update_coord_label(self, pos):
        enu = self._widget_to_enu(pos)
        if enu is None:
            self._coord_label.hide()
            return
        east, north = enu
        latlon = self._enu_to_latlon(east, north)
        if latlon is None:
            self._coord_label.hide()
            return
        lat, lon = latlon
        self._coord_label.setText("{:.6f}°, {:.6f}°".format(lat, lon))
        self._coord_label.adjustSize()
        lx = int(pos.x()) + 16
        ly = int(pos.y()) - 10
        if lx + self._coord_label.width() > self.width():
            lx = int(pos.x()) - self._coord_label.width() - 8
        if ly < 28:
            ly = 28
        self._coord_label.move(lx, ly)
        self._coord_label.show()
        self._coord_label.raise_()


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
        self.ortho_view   = OrthoViewer()
        self._stack.addWidget(self.image_view)
        self._stack.addWidget(self.flight_map)
        self._stack.addWidget(self.ortho_view)
        layout.addWidget(self._stack, stretch=1)

    def _on_tab_changed(self, index):
        self._stack.setCurrentIndex(index)

    def switch_to(self, tab_name: str):
        idx = self.TABS.index(tab_name)
        self._tab_bar.setCurrentIndex(idx)

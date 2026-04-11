"""Viewport overlay widgets: orientation cube and scale bar."""

import numpy as np

from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QFont, QColor


_CUBE_BTN_STYLE = (
    "QPushButton { background:rgba(30,30,30,180); color:#ccc;"
    " border:1px solid #555; border-radius:3px;"
    " font:bold 9px; padding:0; min-width:0; }"
    "QPushButton:hover { background:rgba(0,120,212,200); color:white; }"
)

_VIEWS = [
    ("Top",    0, 1,  90,   0),
    ("Left",   1, 0,   0, -90),
    ("Front",  1, 1,   0,   0),
    ("Right",  1, 2,   0,  90),
    ("Bottom", 2, 1, -90,   0),
]

_SCALE_NICES = [
    0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000,
]


class ViewCube(QWidget):
    """Camera orientation widget overlaid in the top-right corner."""

    view_angle = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(134, 100)
        self.setStyleSheet(_CUBE_BTN_STYLE)
        self._build()

    def _build(self):
        grid = QGridLayout(self)
        grid.setSpacing(2)
        grid.setContentsMargins(2, 2, 2, 2)
        for label, row, col, el, az in _VIEWS:
            btn = QPushButton(label)
            btn.setFixedSize(40, 24)
            btn.clicked.connect(
                lambda _, e=el, a=az: self.view_angle.emit(float(e), float(a))
            )
            grid.addWidget(btn, row, col, Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()


class ScaleBar(QWidget):
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
        bar_m = min(_SCALE_NICES, key=lambda v: abs(v - target))
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

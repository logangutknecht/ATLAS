"""Photogrammetry viewport: thumbnail grid and full-image viewer."""

import os

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QGridLayout, QPushButton, QSizePolicy,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon, QFont


_THUMB_BTN_STYLE = (
    "QPushButton {"
    "  background: #2a2a2a; border: 2px solid #444;"
    "  border-radius: 4px; padding: 2px;"
    "}"
    "QPushButton:hover { border-color: #0078d4; }"
    "QPushButton:checked { border-color: #0078d4; background: #1a3a5c; }"
)

_NO_IMAGE_STYLE = "color: #555; font-size: 14px; background: #1a1a2e;"


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

        self._buttons = {}   # path -> QPushButton
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


class ImageViewer(QWidget):
    """Displays a single image scaled to fit the available space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = self._build_header()
        layout.addWidget(self._header)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(_NO_IMAGE_STYLE)
        self._image_label.setText("No image selected")
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, stretch=1)

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet("background: #252525; border-bottom: 1px solid #333;")
        row = QHBoxLayout(header)
        row.setContentsMargins(8, 0, 8, 0)
        self._filename_label = QLabel("")
        self._filename_label.setStyleSheet("color:#999; font-size:11px; background:transparent;")
        row.addWidget(self._filename_label)
        row.addStretch()
        self._dim_label = QLabel("")
        self._dim_label.setStyleSheet("color:#666; font-size:11px; background:transparent;")
        row.addWidget(self._dim_label)
        return header

    def show_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._image_label.setText("Failed to load: {}".format(os.path.basename(path)))
            return
        self._pixmap = pixmap
        self._filename_label.setText(os.path.basename(path))
        self._dim_label.setText("{}  ×  {}".format(pixmap.width(), pixmap.height()))
        self._fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def _fit_image(self):
        if self._pixmap is None:
            return
        label = self._image_label
        scaled = self._pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

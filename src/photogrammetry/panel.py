"""Photogrammetry panel: image browser + viewer, self-contained QWidget."""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox, QScrollArea,
    QProgressBar, QFileDialog, QToolBar, QSizePolicy,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction

from src.photogrammetry.workers import ImageLoaderThread
from src.photogrammetry.viewport import ThumbnailGrid, ImageViewer


class PhotogrammetryPanel(QWidget):
    """Full photogrammetry tab: controls sidebar + image viewer."""

    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loader = None
        self._image_paths = []
        self._image_dir = self._default_image_dir()

        self._build_ui()

    # -- Layout -------------------------------------------------------------

    def _default_image_dir(self):
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "..", "CAMERA"))
        return candidate if os.path.isdir(candidate) else os.path.expanduser("~")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        self._viewer = ImageViewer()
        splitter.addWidget(self._viewer)
        splitter.setSizes([250, 1190])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_toolbar(self):
        tb = QToolBar("Photogrammetry")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setStyleSheet(
            "QToolBar { spacing:6px; padding:2px 6px; }"
            "QToolBar QLabel { color:#aaa; font-size:11px; background:transparent; }"
        )

        open_btn = QPushButton("Open Image Folder")
        open_btn.clicked.connect(self._open_folder_dialog)
        open_btn.setFixedHeight(26)
        tb.addWidget(open_btn)

        tb.addSeparator()

        self._folder_label = QLabel("No folder loaded")
        self._folder_label.setStyleSheet("color:#888; font-size:11px; background:transparent;")
        tb.addWidget(self._folder_label)

        return tb

    def _build_controls(self):
        container = QWidget()
        container.setFixedWidth(250)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_info_group())
        layout.addWidget(self._build_thumbnail_area(), stretch=1)
        return container

    def _build_info_group(self):
        g = QGroupBox("Images")
        lay = QVBoxLayout(g)

        self._count_label = QLabel("0 images")
        self._count_label.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(self._count_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.hide()
        lay.addWidget(self._progress)

        return g

    def _build_thumbnail_area(self):
        wrapper = QWidget()
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Thumbnails")
        label.setStyleSheet("color:#666; font-size:10px;")
        lay.addWidget(label)

        self._thumb_grid = ThumbnailGrid()
        self._thumb_grid.image_selected.connect(self._on_image_selected)
        lay.addWidget(self._thumb_grid, stretch=1)

        return wrapper

    # -- Menu setup (called by ATLASWindow when this tab is activated) ------

    def setup_menus(self, menubar):
        fm = menubar.addMenu("&File")
        open_act = QAction("Open Image Folder\u2026", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_folder_dialog)
        fm.addAction(open_act)

    # -- Folder loading -----------------------------------------------------

    def _open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", self._image_dir
        )
        if folder:
            self._image_dir = folder
            self._load_folder(folder)

    def _load_folder(self, folder):
        if self._loader and self._loader.isRunning():
            self._loader.terminate()
            self._loader.wait()

        self._thumb_grid.clear()
        self._image_paths = []
        self._progress.setValue(0)
        self._progress.show()
        self.status_message.emit("Loading images from {}...".format(os.path.basename(folder)))

        self._loader = ImageLoaderThread(folder)
        self._loader.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._loader.finished.connect(self._on_load_finished)
        self._loader.progress.connect(self._on_load_progress)
        self._loader.error.connect(self._on_load_error)
        self._loader.start()

    def _on_thumbnail_ready(self, path, pixmap):
        self._thumb_grid.add_thumbnail(path, pixmap)

    def _on_load_finished(self, paths):
        self._image_paths = paths
        self._progress.hide()
        n = len(paths)
        self._count_label.setText("{:,} image{}".format(n, "s" if n != 1 else ""))
        self._folder_label.setText(os.path.basename(self._image_dir))
        self.status_message.emit(
            "Loaded {:,} image{} from {}".format(
                n, "s" if n != 1 else "", os.path.basename(self._image_dir)
            )
        )

    def _on_load_progress(self, message):
        # Parse "X/Y" from the progress message to update the bar
        try:
            parts = message.split()
            frac = parts[-1]
            cur, total = map(int, frac.split("/"))
            self._progress.setRange(0, total)
            self._progress.setValue(cur)
        except (ValueError, IndexError):
            pass
        self.status_message.emit(message)

    def _on_load_error(self, msg):
        self._progress.hide()
        self.status_message.emit("Image load error")
        QMessageBox.critical(self, "Error", msg)

    def _on_image_selected(self, path):
        self._viewer.show_image(path)
        self.status_message.emit(os.path.basename(path))

    # -- Auto-load default CAMERA dir on first show -------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._image_paths and os.path.isdir(self._image_dir):
            self._load_folder(self._image_dir)

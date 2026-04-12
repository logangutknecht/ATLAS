"""Photogrammetry panel: image browser, flight map, and orthomosaic generation."""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox, QScrollArea,
    QProgressBar, QFileDialog, QToolBar, QDoubleSpinBox,
    QMessageBox, QCheckBox,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction, QPixmap

from src.photogrammetry.workers import ImageLoaderThread, MetadataLoaderThread, OrthoThread
from src.photogrammetry.viewport import ThumbnailGrid, PhotogrammetryViewport
from src.photogrammetry.processing.ortho import OrthoConfig


class PhotogrammetryPanel(QWidget):
    """Full photogrammetry tab: thumbnail sidebar + tabbed viewport."""

    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loader      = None
        self._meta_loader = None
        self._ortho_worker = None

        self._image_paths = []
        self._all_metadata = []
        self._poses = []
        self._image_dir = self._default_image_dir()

        self._build_ui()

    # -- Layout -------------------------------------------------------------

    def _default_image_dir(self):
        here      = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "..", "CAMERA"))
        return candidate if os.path.isdir(candidate) else os.path.expanduser("~")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        self._viewport = PhotogrammetryViewport()
        self._viewport.flight_map.camera_selected.connect(self._on_flight_map_camera_selected)
        splitter.addWidget(self._viewport)
        splitter.setSizes([260, 1180])
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

        open_folder_btn = QPushButton("Open Image Folder")
        open_folder_btn.setFixedHeight(26)
        open_folder_btn.clicked.connect(self._open_folder_dialog)
        tb.addWidget(open_folder_btn)

        open_files_btn = QPushButton("Open Images")
        open_files_btn.setFixedHeight(26)
        open_files_btn.clicked.connect(self._open_files_dialog)
        tb.addWidget(open_files_btn)

        tb.addSeparator()

        self._build_ortho_btn = QPushButton("Build Orthomosaic")
        self._build_ortho_btn.setFixedHeight(26)
        self._build_ortho_btn.setEnabled(False)
        self._build_ortho_btn.clicked.connect(self._run_ortho)
        tb.addWidget(self._build_ortho_btn)

        tb.addWidget(QLabel("  Resolution (m/px):"))
        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.05, 2.0)
        self._res_spin.setSingleStep(0.05)
        self._res_spin.setDecimals(2)
        self._res_spin.setValue(0.25)
        self._res_spin.setFixedWidth(72)
        tb.addWidget(self._res_spin)

        tb.addWidget(QLabel("  Min altitude (m):"))
        self._min_alt_spin = QDoubleSpinBox()
        self._min_alt_spin.setRange(0.0, 200.0)
        self._min_alt_spin.setSingleStep(5.0)
        self._min_alt_spin.setDecimals(0)
        self._min_alt_spin.setValue(65.0)
        self._min_alt_spin.setFixedWidth(60)
        self._min_alt_spin.setToolTip(
            "Exclude images below this relative altitude (m).\n"
            "Raise to skip ascent/descent frames."
        )
        tb.addWidget(self._min_alt_spin)

        tb.addSeparator()

        self._source_label = QLabel("No images loaded")
        self._source_label.setStyleSheet("color:#888; font-size:11px; background:transparent;")
        tb.addWidget(self._source_label)

        return tb

    def _build_sidebar(self):
        container = QWidget()
        container.setFixedWidth(260)
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

        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet("color:#777; font-size:10px;")
        self._meta_label.setWordWrap(True)
        lay.addWidget(self._meta_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.hide()
        lay.addWidget(self._progress)

        return g

    def _build_thumbnail_area(self):
        wrapper = QWidget()
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel("Thumbnails")
        lbl.setStyleSheet("color:#666; font-size:10px;")
        lay.addWidget(lbl)
        self._thumb_grid = ThumbnailGrid()
        self._thumb_grid.image_selected.connect(self._on_image_selected)
        lay.addWidget(self._thumb_grid, stretch=1)
        return wrapper

    # -- Menu setup ---------------------------------------------------------

    def setup_menus(self, menubar):
        fm = menubar.addMenu("&File")
        act_folder = QAction("Open Image Folder\u2026", self)
        act_folder.setShortcut("Ctrl+O")
        act_folder.triggered.connect(self._open_folder_dialog)
        fm.addAction(act_folder)

        act_files = QAction("Open Images\u2026", self)
        act_files.setShortcut("Ctrl+Shift+O")
        act_files.triggered.connect(self._open_files_dialog)
        fm.addAction(act_files)

        pm = menubar.addMenu("&Processing")
        self._act_ortho = QAction("Build Orthomosaic\u2026", self)
        self._act_ortho.setEnabled(bool(self._all_metadata))
        self._act_ortho.triggered.connect(self._run_ortho)
        pm.addAction(self._act_ortho)

    # -- Image loading ------------------------------------------------------

    def _open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", self._image_dir
        )
        if folder:
            self._image_dir = folder
            self._source_label.setText(os.path.basename(folder))
            self._load_images(folder)

    def _open_files_dialog(self):
        from src.photogrammetry.workers import IMAGE_EXTENSIONS
        exts = " ".join("*" + ext for ext in sorted(IMAGE_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", self._image_dir,
            "Images ({});;All files (*)".format(exts),
        )
        if paths:
            self._image_dir = os.path.dirname(paths[0])
            self._source_label.setText("{} image{}".format(
                len(paths), "s" if len(paths) != 1 else "",
            ))
            self._load_images(paths)

    def _load_images(self, source):
        """Start loading from *source* (a directory path or list of file paths)."""
        if self._loader and self._loader.isRunning():
            self._loader.terminate()
            self._loader.wait()

        self._thumb_grid.clear()
        self._image_paths  = []
        self._all_metadata = []
        self._poses        = []
        self._build_ortho_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.show()

        if isinstance(source, list):
            self.status_message.emit("Loading {} image{}…".format(
                len(source), "s" if len(source) != 1 else "",
            ))
        else:
            self.status_message.emit("Loading images from {}…".format(
                os.path.basename(source),
            ))

        self._loader = ImageLoaderThread(source)
        self._loader.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._loader.finished.connect(self._on_images_found)
        self._loader.progress.connect(self._on_image_load_progress)
        self._loader.error.connect(self._on_error)
        self._loader.start()

    def _on_images_found(self, paths):
        self._image_paths = paths
        n = len(paths)
        self._count_label.setText("{:,} image{}".format(n, "s" if n != 1 else ""))
        self.status_message.emit("Found {:,} images — reading metadata…".format(n))

        # Start metadata loading right after image discovery
        self._meta_loader = MetadataLoaderThread(paths)
        self._meta_loader.progress.connect(self._on_meta_progress)
        self._meta_loader.finished.connect(self._on_metadata_loaded)
        self._meta_loader.error.connect(self._on_error)
        self._meta_loader.start()

    def _on_metadata_loaded(self, metadata_list):
        self._progress.hide()
        self._all_metadata = metadata_list
        self._compute_poses()
        self._build_ortho_btn.setEnabled(True)

        n = len(metadata_list)
        min_alt = self._min_alt_spin.value()
        above_thresh = sum(
            1 for m in metadata_list
            if m["gps"]["altitude_relative"] >= min_alt
        )
        self._meta_label.setText(
            "{} images with GPS\n{} above {:.0f}m  (turn filtering happens at build time)".format(
                n, above_thresh, min_alt
            )
        )
        self.status_message.emit(
            "Metadata ready — {} images, {} above {:.0f}m — click 'Build Orthomosaic'".format(
                n, above_thresh, min_alt
            )
        )
        self._viewport.flight_map.load(self._all_metadata, self._poses)
        self._viewport.switch_to("Flight Map")

    def _compute_poses(self):
        from src.photogrammetry.processing.georef import build_camera_pose, compute_origin
        if not self._all_metadata:
            return
        origin = compute_origin(self._all_metadata)
        self._poses = [build_camera_pose(m, origin) for m in self._all_metadata]

    # -- Thumbnail / image view ---------------------------------------------

    def _on_thumbnail_ready(self, path, pixmap):
        self._thumb_grid.add_thumbnail(path, pixmap)

    def _on_image_selected(self, path):
        self._viewport.image_view.show_image(path)
        self._viewport.switch_to("Image")
        self._viewport.flight_map.highlight_camera(path)
        self.status_message.emit(os.path.basename(path))

    def _on_flight_map_camera_selected(self, path):
        self._viewport.image_view.show_image(path)
        self._viewport.switch_to("Image")
        if path in self._thumb_grid._buttons:
            self._thumb_grid._on_clicked(path)
        self.status_message.emit(os.path.basename(path))

    # -- Orthomosaic --------------------------------------------------------

    def _run_ortho(self):
        if not self._all_metadata:
            QMessageBox.information(self, "No Metadata", "Load a folder with GPS-tagged images first.")
            return
        if self._ortho_worker and self._ortho_worker.isRunning():
            return

        config = OrthoConfig(
            resolution=self._res_spin.value(),
            image_scale=0.25,
            min_altitude=self._min_alt_spin.value(),
        )
        self._build_ortho_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.show()
        self.status_message.emit("Building orthomosaic…")

        self._ortho_worker = OrthoThread(self._all_metadata, config)
        self._ortho_worker.progress.connect(self._on_ortho_progress)
        self._ortho_worker.finished.connect(self._on_ortho_done)
        self._ortho_worker.error.connect(self._on_ortho_error)
        self._ortho_worker.start()

    def _on_ortho_done(self, pil_image, info):
        self._progress.hide()
        self._build_ortho_btn.setEnabled(True)

        # Convert PIL image to QPixmap
        import io
        from PyQt6.QtCore import QByteArray
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        qpix = QPixmap()
        qpix.loadFromData(buf.read(), "PNG")

        dirs = info.get("flight_directions", [])
        dir_str = "  |  axes {:.0f}° / {:.0f}°".format(*dirs) if dirs else ""
        rejected = info.get("images_rejected", 0)
        reject_str = "  |  {} turn imgs excluded".format(rejected) if rejected else ""

        self._viewport.ortho_view.show_pixmap(
            qpix,
            "Orthomosaic  {w}×{h} px  |  {res:.2f} m/px  |  {n}/{tot} images{dirs}{rej}  |  {aw:.0f}m × {ah:.0f}m".format(
                w=info["width_px"], h=info["height_px"],
                res=info["resolution"],
                n=info["images_used"], tot=info["images_total"],
                dirs=dir_str, rej=reject_str,
                aw=(info["bounds"][1] - info["bounds"][0]),
                ah=(info["bounds"][3] - info["bounds"][2]),
            )
        )
        self._viewport.switch_to("Ortho")
        self.status_message.emit(
            "Orthomosaic complete — {} × {} px  |  {}/{} images used{}".format(
                info["width_px"], info["height_px"],
                info["images_used"], info["images_total"],
                "  ({} turn images excluded)".format(rejected) if rejected else "",
            )
        )

    def _on_ortho_error(self, msg):
        self._progress.hide()
        self._build_ortho_btn.setEnabled(True)
        self.status_message.emit("Orthomosaic failed")
        QMessageBox.critical(self, "Orthomosaic Error", msg)

    def _on_ortho_progress(self, message, cur, total):
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(cur)
        self.status_message.emit(message)

    # -- Misc progress / error handlers ------------------------------------

    def _on_image_load_progress(self, message):
        try:
            frac = message.split()[-1]
            cur, total = map(int, frac.split("/"))
            self._progress.setRange(0, total)
            self._progress.setValue(cur)
        except (ValueError, IndexError):
            pass
        self.status_message.emit(message)

    def _on_meta_progress(self, message, cur, total):
        self._progress.setRange(0, total)
        self._progress.setValue(cur)
        self.status_message.emit(message)

    def _on_error(self, msg):
        self._progress.hide()
        self.status_message.emit("Error")
        QMessageBox.critical(self, "Error", msg)


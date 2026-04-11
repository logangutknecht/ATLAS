"""LiDAR panel: controls sidebar + 3-D viewport, self-contained QWidget."""

import os

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QSlider, QLabel, QComboBox, QFileDialog,
    QGroupBox, QProgressBar, QScrollArea, QMessageBox,
    QToolBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup

from src.lidar.workers import LASLoaderThread, ProcessingThread, SatelliteThread
from src.lidar.dialogs import (
    SORDialog, VoxelDialog, ElevClipDialog,
    GroundSegDialog, AnomalyDialog, SatelliteDialog,
)
from src.lidar.viewport import PointCloudView
from src.lidar.colors import (
    compute_colors,
    DEV_COLOR_GROUND, DEV_COLOR_DIP, DEV_COLOR_BUMP,
)


class LiDARPanel(QWidget):
    """Full LiDAR tab: toolbar + controls sidebar + 3-D point cloud viewer."""

    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Processing state
        self._cloud = None
        self._mask = None
        self._ground_mask = None
        self._deviations = None
        self._features = None
        self._view_mode = "all"
        self._sat_colors = None
        self._dark_bg = True
        self._filepath = None

        # Remembered parameters
        self._sor_k = 20
        self._sor_std = 2.0
        self._voxel_size = 0.10
        self._z_min = 0.0
        self._z_max = 1000.0
        self._ground_cell = 1.0
        self._ground_ht = 0.3
        self._anom_spacing = 3.0
        self._anom_thresh = 0.05
        self._anom_min_area = 0.1
        self._epsg_text = ""
        self._crs_display = ""

        # Worker threads (kept as attributes to prevent GC)
        self._loader = None
        self._worker = None
        self._sat_worker = None

        # Default directory for open dialog
        self._las_dir = self._default_las_dir()

        self._build_ui()
        self._wire()

    # -- Layout -------------------------------------------------------------

    def _default_las_dir(self):
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "..", "..", "LiDAR_LAS")
        candidate = os.path.normpath(candidate)
        return candidate if os.path.isdir(candidate) else os.path.dirname(here)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toolbar = self._build_toolbar()
        root.addWidget(self._toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        self.viewport = PointCloudView()
        splitter.addWidget(self.viewport)
        splitter.setSizes([220, 1220])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_controls(self):
        container = QWidget()
        container.setFixedWidth(220)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setSpacing(6)
        col.setContentsMargins(8, 8, 8, 8)
        col.addWidget(self._build_file_group())
        col.addStretch()
        scroll.setWidget(inner)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        return container

    def _build_file_group(self):
        g = QGroupBox("File")
        lay = QVBoxLayout(g)

        self._load_btn = QPushButton("Open LAS / LAZ File")
        self._load_btn.clicked.connect(self._open_dialog)
        lay.addWidget(self._load_btn)

        self._file_info = QLabel("No file loaded")
        self._file_info.setWordWrap(True)
        self._file_info.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(self._file_info)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
        lay.addWidget(self._progress)

        lay.addWidget(QLabel("Max Points (millions):"))
        row = QHBoxLayout()
        self._max_pts_slider = QSlider(Qt.Orientation.Horizontal)
        self._max_pts_slider.setRange(1, 100)
        self._max_pts_slider.setValue(10)
        self._max_pts_label = QLabel("10 M")
        self._max_pts_slider.valueChanged.connect(
            lambda v: self._max_pts_label.setText("{} M".format(v))
        )
        row.addWidget(self._max_pts_slider)
        row.addWidget(self._max_pts_label)
        lay.addLayout(row)

        self._reload_btn = QPushButton("Reload with New Limit")
        self._reload_btn.setEnabled(False)
        self._reload_btn.clicked.connect(
            lambda: self._reload(self._max_pts_slider.value() * 1_000_000)
        )
        lay.addWidget(self._reload_btn)

        return g

    def _build_toolbar(self):
        tb = QToolBar("Display")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setStyleSheet(
            "QToolBar { spacing:6px; padding:2px 6px; }"
            "QToolBar QLabel { color:#aaa; font-size:11px; background:transparent; }"
        )

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

        self._ground_label = QLabel("")
        self._ground_label.setStyleSheet("color:#999; font-size:11px; background:transparent;")
        tb.addWidget(self._ground_label)

        return tb

    def _wire(self):
        self.viewport.view_cube.view_angle.connect(self.viewport.set_camera_angles)

    # -- Menu setup (called by ATLASWindow when this tab is activated) -------

    def setup_menus(self, menubar):
        self._build_file_menu(menubar)
        self._build_view_menu(menubar)
        self._build_processing_menu(menubar)

    def _build_file_menu(self, menubar):
        fm = menubar.addMenu("&File")
        open_act = QAction("&Open LAS/LAZ\u2026", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_dialog)
        fm.addAction(open_act)

    def _build_view_menu(self, menubar):
        vm = menubar.addMenu("&View")
        reset_act = QAction("&Reset Camera", self)
        reset_act.setShortcut("R")
        reset_act.triggered.connect(self.viewport.reset_camera)
        vm.addAction(reset_act)
        vm.addSeparator()
        for label, shortcut, el, az in [
            ("Top",   "Ctrl+1",  90, 0),
            ("Front", "Ctrl+2",   0, 0),
            ("Right", "Ctrl+3",   0, 90),
        ]:
            act = QAction("{} View".format(label), self)
            act.setShortcut(shortcut)
            act.triggered.connect(
                lambda _, e=el, a=az: self.viewport.set_camera_angles(e, a)
            )
            vm.addAction(act)

    def _build_processing_menu(self, menubar):
        pm = menubar.addMenu("&Processing")
        has_cloud = self._cloud is not None

        self._act_sor = self._proc_action(pm, "Statistical Outlier Removal\u2026", self._dlg_sor, has_cloud)
        self._act_voxel = self._proc_action(pm, "Voxel Downsample\u2026", self._dlg_voxel, has_cloud)
        self._act_elev = self._proc_action(pm, "Elevation Clip\u2026", self._dlg_elev, has_cloud)
        pm.addSeparator()
        self._act_ground = self._proc_action(pm, "Ground Segmentation\u2026", self._dlg_ground, has_cloud)
        self._act_anomaly = self._proc_action(pm, "Detect Ground Anomalies\u2026", self._dlg_anomaly, self._ground_mask is not None)
        pm.addSeparator()

        self._view_group = QActionGroup(self)
        self._view_group.setExclusive(True)
        self._act_view_all = self._view_action(pm, "Show All Points", "all", checked=True, enabled=True)
        self._act_view_ground = self._view_action(pm, "Show Ground Only", "ground", enabled=self._ground_mask is not None)
        self._act_view_anom = self._view_action(pm, "Show Ground + Anomalies", "ground_anom", enabled=self._deviations is not None)
        pm.addSeparator()
        self._act_sat = self._proc_action(pm, "Apply Satellite Texture\u2026", self._dlg_satellite, has_cloud)
        pm.addSeparator()
        self._act_reset = self._proc_action(pm, "Reset to Original", self._reset_proc, has_cloud)

    def _proc_action(self, menu, label, slot, enabled):
        act = QAction(label, self)
        act.setEnabled(enabled)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _view_action(self, menu, label, mode, checked=False, enabled=True):
        act = QAction(label, self)
        act.setCheckable(True)
        act.setChecked(checked)
        act.setEnabled(enabled)
        act.triggered.connect(lambda: self._set_view_mode(mode))
        self._view_group.addAction(act)
        menu.addAction(act)
        return act

    # -- Helpers ------------------------------------------------------------

    def _on_size_change(self, text):
        if text.isdigit():
            self.viewport.update_size(int(text))

    def _current_size(self):
        t = self.size_combo.currentText()
        return int(t) if t.isdigit() else 2

    def _set_busy(self, busy):
        self._load_btn.setEnabled(not busy)
        self._progress.setVisible(busy)

    def _add_color_mode(self, name):
        items = [self.color_combo.itemText(i) for i in range(self.color_combo.count())]
        if name not in items:
            self.color_combo.addItem(name)

    # -- Dialog launchers ---------------------------------------------------

    def _dlg_sor(self):
        d = SORDialog(self, self._sor_k, self._sor_std)
        if d.exec():
            self._sor_k, self._sor_std = d.k_spin.value(), d.std_spin.value()
            self._run_proc("sor", k=self._sor_k, std_ratio=self._sor_std)

    def _dlg_voxel(self):
        d = VoxelDialog(self, self._voxel_size)
        if d.exec():
            self._voxel_size = d.size_spin.value()
            self._run_proc("voxel", voxel_size=self._voxel_size)

    def _dlg_elev(self):
        d = ElevClipDialog(self, self._z_min, self._z_max)
        if d.exec():
            self._z_min, self._z_max = d.z_min_spin.value(), d.z_max_spin.value()
            self._run_proc("elevation", z_min=self._z_min, z_max=self._z_max)

    def _dlg_ground(self):
        d = GroundSegDialog(self, self._ground_cell, self._ground_ht)
        if d.exec():
            self._ground_cell, self._ground_ht = d.cell_spin.value(), d.ht_spin.value()
            self._run_ground_seg(self._ground_cell, self._ground_ht)

    def _dlg_anomaly(self):
        if self._ground_mask is None:
            QMessageBox.information(self, "Ground Required", "Run Ground Segmentation first.")
            return
        d = AnomalyDialog(self, self._anom_spacing, self._anom_thresh, self._anom_min_area)
        if d.exec():
            self._anom_spacing = d.spacing_spin.value()
            self._anom_thresh = d.thresh_spin.value()
            self._anom_min_area = d.area_spin.value()
            self._run_anomaly(self._anom_spacing, self._anom_thresh, self._anom_min_area)

    def _dlg_satellite(self):
        d = SatelliteDialog(self, self._crs_display, self._epsg_text)
        if d.exec():
            self._epsg_text = d.epsg_input.text().strip()
            self._run_satellite()

    # -- File I/O -----------------------------------------------------------

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", self._las_dir,
            "LAS files (*.las *.laz);;All files (*)",
        )
        if path:
            self._las_dir = os.path.dirname(path)
            self._load(path, self._max_pts_slider.value() * 1_000_000)

    def _load(self, path, max_pts):
        if self._loader and self._loader.isRunning():
            return
        self._filepath = path
        self._set_busy(True)
        self.status_message.emit("Loading {}...".format(os.path.basename(path)))
        self._loader = LASLoaderThread(path, max_pts)
        self._loader.progress.connect(self.status_message.emit)
        self._loader.finished.connect(self._on_loaded)
        self._loader.error.connect(self._on_load_err)
        self._loader.start()

    def _on_loaded(self, data):
        self._cloud = data
        self._mask = None
        self._ground_mask = None
        self._deviations = None
        self._features = None
        self._view_mode = "all"
        self._sat_colors = None
        self._set_busy(False)
        self._update_file_info(data)
        self._refresh_color_combo(data["available_coloring"], default="RGB" if "RGB" in data["available_coloring"] else "Elevation")
        self._enable_processing_actions(True)

        z = data["points"][:, 2]
        self._z_min, self._z_max = float(z.min()), float(z.max())
        self._update_crs_state(data)

        self._ground_label.setText("")
        self._render()
        self.viewport.reset_camera()
        self.status_message.emit("{:,} / {:,} pts | {:.1f}s".format(
            data["displayed_points"], data["total_points"], data["load_time"],
        ))

    def _on_load_err(self, msg):
        self._set_busy(False)
        self.status_message.emit("Load failed")
        QMessageBox.critical(self, "Error", msg)

    def _reload(self, max_pts):
        if self._filepath:
            self._load(self._filepath, max_pts)

    def _update_file_info(self, data):
        self._file_info.setText(
            "{}\nTotal: {:,}\nDisplayed: {:,}\nLoaded in {:.1f}s".format(
                os.path.basename(data["filepath"]),
                data["total_points"],
                data["displayed_points"],
                data["load_time"],
            )
        )
        self._file_info.setStyleSheet("color:#bbb; font-size:11px;")
        self._reload_btn.setEnabled(True)

    def _update_crs_state(self, data):
        crs_info = data.get("crs_info", "")
        crs_epsg = data.get("crs_epsg")
        coord_hint = data.get("coord_hint", "")
        if crs_info:
            self._crs_display = "CRS: " + crs_info
            self._epsg_text = str(crs_epsg) if crs_epsg else ""
        elif coord_hint:
            self._crs_display = "CRS: not in file\n" + coord_hint
            self._epsg_text = ""
        else:
            self._crs_display = "CRS: not found"
            self._epsg_text = ""

    def _refresh_color_combo(self, options, default="Elevation"):
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        for opt in options:
            self.color_combo.addItem(opt)
        self.color_combo.blockSignals(False)
        self.color_combo.setCurrentText(default)

    def _enable_processing_actions(self, enabled):
        for act in (self._act_sor, self._act_voxel, self._act_elev,
                    self._act_ground, self._act_sat, self._act_reset):
            act.setEnabled(enabled)

    # -- Rendering ----------------------------------------------------------

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
        colors = self._compute_render_colors(pts, attrs, mode)
        pts, colors = self._apply_view_mode_filter(pts, colors)
        self.viewport.display(pts, colors, self._current_size())

    def _compute_render_colors(self, pts, attrs, mode):
        if mode == "Satellite" and self._sat_colors is not None:
            colors = self._sat_colors
            return colors[self._mask] if self._mask is not None else colors

        if mode == "Deviation" and self._deviations is not None:
            return self._deviation_colors(pts)

        if mode == "Ground" and self._ground_mask is not None:
            return self._ground_colors(pts)

        return compute_colors(pts, attrs, mode)

    def _deviation_colors(self, pts):
        dev = self._deviations
        if self._mask is not None:
            dev = dev[self._mask]
        colors = np.tile(np.array(DEV_COLOR_GROUND, dtype=np.float32), (len(pts), 1))
        colors[dev < 0] = DEV_COLOR_DIP
        colors[dev > 0] = DEV_COLOR_BUMP
        return colors

    def _ground_colors(self, pts):
        gm = self._ground_mask
        if self._mask is not None:
            gm = gm[self._mask]
        colors = np.zeros((len(pts), 4), dtype=np.float32)
        colors[gm] = [0.55, 0.35, 0.17, 1.0]
        colors[~gm] = [0.15, 0.65, 0.15, 1.0]
        return colors

    def _apply_view_mode_filter(self, pts, colors):
        if self._view_mode in ("ground", "ground_anom") and self._ground_mask is not None:
            gm = self._ground_mask
            if self._mask is not None:
                gm = gm[self._mask]
            return pts[gm], colors[gm]
        return pts, colors

    def _toggle_bg(self):
        self._dark_bg = not self._dark_bg
        self.viewport.set_bg("#1a1a2e" if self._dark_bg else "#e0e0e0")

    # -- Processing ---------------------------------------------------------

    def _run_proc(self, op, **kw):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        self._set_busy(True)
        self.status_message.emit("Running {}...".format(op))
        self._worker = ProcessingThread(op, pts, **kw)
        self._worker.progress.connect(self.status_message.emit)
        self._worker.finished.connect(self._on_proc_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_proc_done(self, new_mask):
        self._set_busy(False)
        if self._mask is not None:
            full = self._mask.copy()
            full[np.where(full)[0][~new_mask]] = False
            self._mask = full
        else:
            self._mask = new_mask
        kept = int(self._mask.sum())
        removed = self._cloud["displayed_points"] - kept
        self._render()
        self.status_message.emit("Done \u2014 {:,} kept, {:,} removed".format(kept, removed))

    def _on_proc_err(self, msg):
        self._set_busy(False)
        self.status_message.emit("Processing failed")
        QMessageBox.critical(self, "Processing Error", msg)

    # -- Ground segmentation ------------------------------------------------

    def _set_view_mode(self, mode):
        self._view_mode = mode
        self._render()

    def _run_ground_seg(self, cell_size, height_threshold):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        self._set_busy(True)
        self.status_message.emit("Running ground segmentation...")
        self._worker = ProcessingThread("ground", pts, cell_size=cell_size, height_threshold=height_threshold)
        self._worker.progress.connect(self.status_message.emit)
        self._worker.finished.connect(self._on_ground_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_ground_done(self, ground_mask):
        self._set_busy(False)
        self._ground_mask = self._expand_mask_to_full(ground_mask)

        n_g, n_t = int(ground_mask.sum()), len(ground_mask)
        self._ground_label.setText(
            "Ground: {:,} ({:.1f}%) | Non-ground: {:,}".format(
                n_g, 100.0 * n_g / max(n_t, 1), n_t - n_g
            )
        )
        self.status_message.emit(
            "Ground seg done \u2014 {:,} ground, {:,} non-ground".format(n_g, n_t - n_g)
        )
        self._act_anomaly.setEnabled(True)
        self._act_view_ground.setEnabled(True)
        self._act_view_anom.setEnabled(True)
        self._add_color_mode("Ground")
        self.color_combo.setCurrentText("Ground")
        self._render()

    # -- Anomaly detection --------------------------------------------------

    def _run_anomaly(self, reference_spacing, threshold, min_area):
        if not self._cloud:
            return
        pts, _ = self._active_data()
        gm = self._ground_mask if self._mask is None else self._ground_mask[self._mask]
        self._set_busy(True)
        self.status_message.emit("Detecting ground anomalies...")
        self._worker = ProcessingThread(
            "anomaly", pts,
            ground_mask=gm, reference_spacing=reference_spacing,
            threshold=threshold, min_area=min_area,
        )
        self._worker.progress.connect(self.status_message.emit)
        self._worker.finished.connect(self._on_anomaly_done)
        self._worker.error.connect(self._on_proc_err)
        self._worker.start()

    def _on_anomaly_done(self, result):
        self._set_busy(False)
        devs, feats = result["deviations"], result["features"]

        if self._mask is not None:
            full = np.zeros(self._cloud["displayed_points"], dtype=np.float64)
            full[np.where(self._mask)[0]] = devs
            self._deviations = full
        else:
            self._deviations = devs
        self._features = feats

        n_bumps = sum(1 for f in feats if f["type"] == "bump")
        n_dips = sum(1 for f in feats if f["type"] == "dip")
        n_g = int(self._ground_mask.sum()) if self._ground_mask is not None else 0

        parts = ["Ground: {:,}".format(n_g)]
        if n_bumps:
            biggest = max((f for f in feats if f["type"] == "bump"), key=lambda f: abs(f["max_dev"]))
            parts.append("{:,} bumps (max {:.0f}cm, {:.1f}m\u00b2)".format(
                n_bumps, abs(biggest["max_dev"]) * 100, biggest["area"]))
        if n_dips:
            biggest = max((f for f in feats if f["type"] == "dip"), key=lambda f: abs(f["max_dev"]))
            parts.append("{:,} dips (max {:.0f}cm, {:.1f}m\u00b2)".format(
                n_dips, abs(biggest["max_dev"]) * 100, biggest["area"]))
        self._ground_label.setText(" | ".join(parts))
        self.status_message.emit(
            "Anomaly detection done \u2014 {:,} bumps + {:,} dips".format(n_bumps, n_dips)
        )
        self._act_view_anom.setEnabled(True)
        self._view_mode = "ground_anom"
        self._act_view_anom.setChecked(True)
        self._add_color_mode("Deviation")
        self.color_combo.setCurrentText("Deviation")
        self._render()

    # -- Satellite texture --------------------------------------------------

    def _run_satellite(self):
        if not self._cloud or (self._sat_worker and self._sat_worker.isRunning()):
            return
        epsg_override = self._parse_epsg_override()
        if epsg_override is False:
            return
        self._set_busy(True)
        self.status_message.emit("Fetching satellite imagery...")
        self._sat_worker = SatelliteThread(self._cloud, epsg_override)
        self._sat_worker.progress.connect(self.status_message.emit)
        self._sat_worker.finished.connect(self._on_satellite_done)
        self._sat_worker.error.connect(self._on_satellite_err)
        self._sat_worker.start()

    def _parse_epsg_override(self):
        if not self._epsg_text:
            return None
        try:
            return int(self._epsg_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid EPSG", "EPSG must be an integer.")
            return False

    def _on_satellite_done(self, result):
        self._set_busy(False)
        self._sat_colors = result["colors"]
        crs_epsg = result.get("info", {}).get("crs_epsg")
        self.status_message.emit(
            "Satellite texture ready \u2014 {:,} pts | EPSG:{}".format(
                len(self._sat_colors), crs_epsg or "?"
            )
        )
        self._add_color_mode("Satellite")
        self.color_combo.setCurrentText("Satellite")
        self._render()

    def _on_satellite_err(self, msg):
        self._set_busy(False)
        self.status_message.emit("Satellite texture failed")
        QMessageBox.warning(self, "Satellite Error", msg)

    # -- Reset --------------------------------------------------------------

    def _reset_proc(self):
        self._mask = None
        self._ground_mask = None
        self._deviations = None
        self._features = None
        self._view_mode = "all"
        self._sat_colors = None

        self._act_view_all.setChecked(True)
        self._act_view_ground.setEnabled(False)
        self._act_view_anom.setEnabled(False)
        self._act_anomaly.setEnabled(False)
        self._ground_label.setText("")

        self._refresh_color_combo(self._cloud["available_coloring"], default="Elevation")
        self._render()
        self.status_message.emit("Reset \u2014 {:,} points".format(self._cloud["displayed_points"]))

    # -- Utility ------------------------------------------------------------

    def _expand_mask_to_full(self, local_mask):
        """Map a mask over active points back to full point array size."""
        if self._mask is None:
            return local_mask
        full = np.zeros(self._cloud["displayed_points"], dtype=bool)
        full[np.where(self._mask)[0][local_mask]] = True
        return full

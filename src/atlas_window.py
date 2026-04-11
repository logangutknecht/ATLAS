"""ATLASWindow: main application window with tab sidebar and panel switcher."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QStatusBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction

from src.lidar.panel import LiDARPanel
from src.photogrammetry.panel import PhotogrammetryPanel


_TAB_BASE = (
    "QPushButton {"
    "  background: #252525;"
    "  color: #888;"
    "  border: none;"
    "  border-right: 2px solid transparent;"
    "  border-radius: 0;"
    "  padding: 12px 4px;"
    "  font-size: 10px;"
    "  font-weight: bold;"
    "}"
    "QPushButton:hover { background: #2d2d2d; color: #ccc; }"
)

_TAB_ACTIVE = (
    "QPushButton {"
    "  background: #2d2d2d;"
    "  color: #ffffff;"
    "  border: none;"
    "  border-right: 2px solid #0078d4;"
    "  border-radius: 0;"
    "  padding: 12px 4px;"
    "  font-size: 10px;"
    "  font-weight: bold;"
    "}"
)


class ATLASWindow(QMainWindow):
    """Top-level window: narrow tab sidebar on the left, panel content on the right."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATLAS — Aerial Terrain & LiDAR Analysis Suite")
        self.setMinimumSize(1100, 700)
        self.resize(1440, 900)

        self._panels = {}       # name -> QWidget panel
        self._tab_btns = {}     # name -> QPushButton

        self._build_ui()
        self._activate_tab("lidar")

    # -- Layout -------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._status_label = QLabel("Ready")
        bar = QStatusBar()
        bar.addPermanentWidget(self._status_label)
        self.setStatusBar(bar)

        self._register_panel("lidar", LiDARPanel(), "LiDAR")
        self._register_panel("photo", PhotogrammetryPanel(), "Photo")

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(52)
        sidebar.setStyleSheet("background: #252525; border-right: 1px solid #333;")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)

        self._tab_btns["lidar"] = self._make_tab_btn("LiDAR", "lidar")
        self._tab_btns["photo"] = self._make_tab_btn("Photo", "photo")

        layout.addWidget(self._tab_btns["lidar"])
        layout.addWidget(self._tab_btns["photo"])
        layout.addStretch()

        app_label = QLabel("ATLAS")
        app_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        app_label.setStyleSheet("color:#333; font-size:9px; font-weight:bold; background:transparent;")
        layout.addWidget(app_label)

        return sidebar

    def _make_tab_btn(self, label, tab_name):
        btn = QPushButton(label)
        btn.setFixedSize(QSize(52, 64))
        btn.setStyleSheet(_TAB_BASE)
        btn.clicked.connect(lambda: self._activate_tab(tab_name))
        return btn

    # -- Panel registration -------------------------------------------------

    def _register_panel(self, name, panel, label):
        self._panels[name] = panel
        self._stack.addWidget(panel)
        panel.status_message.connect(self._status_label.setText)

    # -- Tab switching ------------------------------------------------------

    def _activate_tab(self, name):
        panel = self._panels[name]
        self._stack.setCurrentWidget(panel)
        self._rebuild_menus(panel)
        for tab_name, btn in self._tab_btns.items():
            btn.setStyleSheet(_TAB_ACTIVE if tab_name == name else _TAB_BASE)
        self._status_label.setText("Ready")

    def _rebuild_menus(self, panel):
        self.menuBar().clear()
        panel.setup_menus(self.menuBar())
        self._add_app_menu()

    def _add_app_menu(self):
        app_menu = self.menuBar().addMenu("&App")
        exit_act = QAction("E&xit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        app_menu.addAction(exit_act)

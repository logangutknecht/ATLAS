"""Application-wide dark theme stylesheet."""

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

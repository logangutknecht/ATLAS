"""ATLAS — Aerial Terrain & LiDAR Analysis Suite

Entry point. All application logic lives in src/.
"""

import sys

from PyQt6.QtWidgets import QApplication

from src.atlas_window import ATLASWindow
from src.theme import DARK_STYLE


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = ATLASWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

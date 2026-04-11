"""Background workers for photogrammetry image loading."""

import os

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
THUMB_SIZE = (120, 90)


class ImageLoaderThread(QThread):
    """Scans a directory for images, loads thumbnails, and emits them one by one."""

    thumbnail_ready = pyqtSignal(str, QPixmap)   # (path, thumbnail pixmap)
    finished = pyqtSignal(list)                  # list of all image paths
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def run(self):
        try:
            paths = self._collect_image_paths()
            self.finished.emit(paths)
            for i, path in enumerate(paths):
                self.progress.emit("Loading thumbnails… {}/{}".format(i + 1, len(paths)))
                thumb = self._load_thumbnail(path)
                if thumb is not None:
                    self.thumbnail_ready.emit(path, thumb)
        except Exception as exc:
            self.error.emit(str(exc))

    def _collect_image_paths(self):
        paths = []
        for entry in sorted(os.scandir(self.directory), key=lambda e: e.name):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(entry.path)
        return paths

    def _load_thumbnail(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(
            THUMB_SIZE[0], THUMB_SIZE[1],
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

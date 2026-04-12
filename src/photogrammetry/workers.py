"""Background workers for photogrammetry image loading and processing."""

import os

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
THUMB_SIZE = (120, 90)


class ImageLoaderThread(QThread):
    """Loads thumbnails from a directory or an explicit list of image paths."""

    thumbnail_ready = pyqtSignal(str, QPixmap)   # (path, thumbnail pixmap)
    finished        = pyqtSignal(list)            # list of all image paths
    progress        = pyqtSignal(str)
    error           = pyqtSignal(str)

    def __init__(self, source):
        """*source* is either a directory path (str) or a list of file paths."""
        super().__init__()
        if isinstance(source, list):
            self._paths = sorted(source, key=lambda p: os.path.basename(p))
            self._directory = None
        else:
            self._paths = None
            self._directory = source

    def run(self):
        try:
            paths = self._paths if self._paths is not None else self._scan_directory()
            self.finished.emit(paths)
            for i, path in enumerate(paths):
                self.progress.emit("Loading thumbnails… {}/{}".format(i + 1, len(paths)))
                thumb = self._load_thumbnail(path)
                if thumb is not None:
                    self.thumbnail_ready.emit(path, thumb)
        except Exception as exc:
            self.error.emit(str(exc))

    def _scan_directory(self):
        paths = []
        for entry in sorted(os.scandir(self._directory), key=lambda e: e.name):
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


class MetadataLoaderThread(QThread):
    """Parses EXIF/XMP metadata from every image in a list."""

    finished = pyqtSignal(list)    # list of metadata dicts
    progress = pyqtSignal(str, int, int)  # message, current, total
    error    = pyqtSignal(str)

    def __init__(self, paths: list):
        super().__init__()
        self.paths = paths

    def run(self):
        from src.photogrammetry.processing.metadata import parse_image_metadata
        results = []
        total = len(self.paths)
        for i, path in enumerate(self.paths):
            self.progress.emit(
                "Reading metadata {}/{}…".format(i + 1, total), i + 1, total
            )
            try:
                results.append(parse_image_metadata(path))
            except Exception:
                pass
        self.finished.emit(results)


class OrthoThread(QThread):
    """Builds an orthomosaic from pre-loaded image metadata."""

    finished = pyqtSignal(object, dict)   # (PIL.Image, info_dict)
    progress = pyqtSignal(str, int, int)  # message, current, total
    error    = pyqtSignal(str)

    def __init__(self, all_metadata: list, config):
        super().__init__()
        self.all_metadata = all_metadata
        self.config       = config

    def run(self):
        from src.photogrammetry.processing.ortho import build_orthomosaic
        try:
            def _pfn(msg, cur, tot):
                self.progress.emit(msg, cur, tot)

            image, info = build_orthomosaic(
                self.all_metadata,
                self.config,
                progress_fn=_pfn,
            )
            self.finished.emit(image, info)
        except Exception as exc:
            self.error.emit(str(exc))

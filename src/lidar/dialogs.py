"""Parameter dialogs for LiDAR processing operations."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox, QLineEdit,
)


class SORDialog(QDialog):
    def __init__(self, parent=None, k=20, std=2.0):
        super().__init__(parent)
        self.setWindowTitle("Statistical Outlier Removal")
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("K neighbours:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(5, 200)
        self.k_spin.setValue(k)
        row.addWidget(self.k_spin)
        row.addWidget(QLabel("Std ratio:"))
        self.std_spin = QDoubleSpinBox()
        self.std_spin.setRange(0.1, 10.0)
        self.std_spin.setValue(std)
        self.std_spin.setSingleStep(0.1)
        row.addWidget(self.std_spin)
        lay.addLayout(row)

        lay.addWidget(_ok_cancel(self))


class VoxelDialog(QDialog):
    def __init__(self, parent=None, size=0.10):
        super().__init__(parent)
        self.setWindowTitle("Voxel Downsample")
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Voxel size:"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.001, 50.0)
        self.size_spin.setValue(size)
        self.size_spin.setSingleStep(0.01)
        self.size_spin.setDecimals(3)
        row.addWidget(self.size_spin)
        lay.addLayout(row)

        lay.addWidget(_ok_cancel(self))


class ElevClipDialog(QDialog):
    def __init__(self, parent=None, z_min=0.0, z_max=1000.0):
        super().__init__(parent)
        self.setWindowTitle("Elevation Clip")
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Z min:"))
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(-100000, 100000)
        self.z_min_spin.setDecimals(2)
        self.z_min_spin.setValue(z_min)
        row.addWidget(self.z_min_spin)
        row.addWidget(QLabel("Z max:"))
        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(-100000, 100000)
        self.z_max_spin.setDecimals(2)
        self.z_max_spin.setValue(z_max)
        row.addWidget(self.z_max_spin)
        lay.addLayout(row)

        lay.addWidget(_ok_cancel(self))


class GroundSegDialog(QDialog):
    def __init__(self, parent=None, cell=1.0, ht=0.3):
        super().__init__(parent)
        self.setWindowTitle("Ground Segmentation")
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cell size:"))
        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(0.1, 20.0)
        self.cell_spin.setValue(cell)
        self.cell_spin.setSingleStep(0.1)
        self.cell_spin.setDecimals(1)
        row.addWidget(self.cell_spin)
        row.addWidget(QLabel("Height threshold:"))
        self.ht_spin = QDoubleSpinBox()
        self.ht_spin.setRange(0.05, 10.0)
        self.ht_spin.setValue(ht)
        self.ht_spin.setSingleStep(0.05)
        self.ht_spin.setDecimals(2)
        row.addWidget(self.ht_spin)
        lay.addLayout(row)

        lay.addWidget(_ok_cancel(self))


class AnomalyDialog(QDialog):
    def __init__(self, parent=None, spacing=3.0, threshold=0.05, min_area=0.1):
        super().__init__(parent)
        self.setWindowTitle("Detect Ground Anomalies")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "TIN-based slope-corrected anomaly detection.\n"
            "Detects bumps and dips relative to a coarse reference surface."
        ))

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Reference spacing (m):"))
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.5, 50.0)
        self.spacing_spin.setValue(spacing)
        self.spacing_spin.setSingleStep(0.5)
        self.spacing_spin.setDecimals(1)
        self.spacing_spin.setToolTip(
            "Grid spacing for the coarse reference TIN. "
            "Features smaller than this are detectable."
        )
        r1.addWidget(self.spacing_spin)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Threshold (m):"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.005, 50.0)
        self.thresh_spin.setValue(threshold)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setDecimals(3)
        self.thresh_spin.setToolTip("Minimum |deviation| to flag as anomaly")
        r2.addWidget(self.thresh_spin)
        r2.addWidget(QLabel("Min area (m\u00b2):"))
        self.area_spin = QDoubleSpinBox()
        self.area_spin.setRange(0.01, 1000.0)
        self.area_spin.setValue(min_area)
        self.area_spin.setSingleStep(0.1)
        self.area_spin.setDecimals(2)
        self.area_spin.setToolTip("Minimum cluster area to report as a feature")
        r2.addWidget(self.area_spin)
        lay.addLayout(r2)

        lay.addWidget(_ok_cancel(self))


class SatelliteDialog(QDialog):
    def __init__(self, parent=None, info="", epsg=""):
        super().__init__(parent)
        self.setWindowTitle("Apply Satellite Texture")
        lay = QVBoxLayout(self)

        info_label = QLabel(info or "CRS: unknown")
        info_label.setWordWrap(True)
        lay.addWidget(info_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("EPSG:"))
        self.epsg_input = QLineEdit()
        self.epsg_input.setPlaceholderText("e.g. 32617")
        self.epsg_input.setText(epsg)
        row.addWidget(self.epsg_input)
        lay.addLayout(row)

        lay.addWidget(_ok_cancel(self))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_cancel(dialog):
    bb = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    bb.accepted.connect(dialog.accept)
    bb.rejected.connect(dialog.reject)
    return bb

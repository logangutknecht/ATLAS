"""3-D point cloud viewport: camera and render widget."""

import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from vispy import scene

from src.lidar.overlays import ViewCube, ScaleBar


class PanCamera(scene.TurntableCamera):
    """TurntableCamera extended so right-click drag pans instead of zooming."""

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        is_right_drag = (
            event.type == "mouse_move"
            and event.press_event is not None
            and 2 in event.buttons
            and 1 not in event.buttons
        )
        if is_right_drag:
            self._pan(event)
            event.handled = True
            return

        super().viewbox_mouse_event(event)

    def _pan(self, event):
        p1 = event.last_event.pos[:2]
        p2 = event.pos[:2]
        delta = np.array(p2, dtype=np.float64) - np.array(p1, dtype=np.float64)

        vb = self._viewbox.size
        norm = max(vb[0], vb[1], 1.0)
        dist = getattr(self, "_actual_distance", None) or self.distance or 500.0
        speed = dist / norm

        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        right = np.array([np.cos(az), np.sin(az), 0.0])
        up = np.array([
            -np.sin(el) * np.sin(az),
             np.sin(el) * np.cos(az),
             np.cos(el),
        ])

        shift = (-delta[0] * right + delta[1] * up) * speed
        self.center = tuple(np.array(self.center, dtype=np.float64) + shift)
        self.view_changed()


class PointCloudView(QWidget):
    """OpenGL viewport that renders a scatter of 3-D points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pts = None
        self._colors = None
        self._size = 2

        self._canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="#1a1a2e")
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = PanCamera(fov=45, distance=500)
        self._scatter = scene.visuals.Markers(parent=self._view.scene)
        self._axis = scene.visuals.XYZAxis(parent=self._view.scene)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

        self.view_cube = ViewCube(self)
        self.scale_bar = ScaleBar(self)
        self._canvas.events.draw.connect(self._on_draw)

    # -- overlays -----------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_overlays()

    def _place_overlays(self):
        w, h = self.width(), self.height()
        self.view_cube.move(w - self.view_cube.width() - 10, 10)
        self.scale_bar.move(10, h - self.scale_bar.height() - 10)
        self.view_cube.raise_()
        self.scale_bar.raise_()

    def _on_draw(self, _ev=None):
        self.scale_bar.refresh(self._view.camera, self._canvas.size[0])

    # -- public interface ---------------------------------------------------

    def display(self, points, colors, size=2):
        self._pts, self._colors, self._size = points, colors, size
        self._scatter.set_data(pos=points, face_color=colors, size=size, edge_width=0)
        self._canvas.update()

    def update_size(self, size):
        if self._pts is None:
            return
        self._size = size
        self._scatter.set_data(pos=self._pts, face_color=self._colors, size=size, edge_width=0)
        self._canvas.update()

    def set_bg(self, color):
        self._canvas.bgcolor = color

    def reset_camera(self):
        self._view.camera.set_range()
        self._canvas.update()

    def set_camera_angles(self, elevation, azimuth):
        self._view.camera.elevation = elevation
        self._view.camera.azimuth = azimuth
        self._canvas.update()

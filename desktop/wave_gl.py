from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath, QLinearGradient
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from bridge import PipelineState

DEEP = QColor(2, 8, 15)
SAMPLES = 96
# Incomplete circle: ~270° arc (classic spinner gap).
LOADER_SPAN = math.pi * 1.5


class WaveGLWidget(QOpenGLWidget):
    """Same stroke morphs between flat/sound-wave and a spinning arc loader."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state: PipelineState = "starting"
        self._amplitude = 0.0
        self._smooth_amp = 0.0
        self._morph = 1.0  # boot as loader until engine is ready
        self._message = ""
        self._t0 = time.monotonic()
        self._wave_phase = 0.0
        self._loader_angle = 0.0
        # Display points start on the loader arc (normalized 0..1 approximated later).
        self._display_x = [0.5] * SAMPLES
        self._display_y = [0.5] * SAMPLES

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self.update)
        self._timer.start()

        self.setMinimumSize(200, 200)

    def set_status(self, state: str, amplitude: float, message: str = "") -> None:
        self._state = state
        self._amplitude = max(0.0, min(1.0, float(amplitude)))
        if message:
            self._message = message

    def initializeGL(self) -> None:
        self.setAutoFillBackground(False)

    def resizeGL(self, w: int, h: int) -> None:
        pass

    def paintGL(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, DEEP)

        t = time.monotonic() - self._t0
        state = self._state
        error = state == "error"

        self._smooth_amp += (self._amplitude - self._smooth_amp) * 0.18
        if self._smooth_amp < 0.004:
            self._smooth_amp = 0.0
        amp = self._smooth_amp

        target_morph = 1.0 if state in ("processing", "starting") else 0.0
        # Smooth morph both ways (wave ↔ loader).
        morph_rate = 0.10 if abs(target_morph - self._morph) > 0.5 else 0.14
        self._morph += (target_morph - self._morph) * morph_rate
        if abs(self._morph - target_morph) < 0.002:
            self._morph = target_morph
        morph = self._morph

        if error:
            base = QColor(255, 64, 96)
            glow = QColor(255, 40, 70, int(25 + amp * 70))
        else:
            base = QColor(0, 212, 255)
            # Glow stronger while loading or speaking.
            glow_a = int(20 + amp * 90 + morph * 50)
            glow = QColor(0, 180, 255, glow_a)

        # Wave phase only when mostly in wave mode and speaking.
        wave_drive = (1.0 - morph) * amp
        self._wave_phase += (0.35 + wave_drive * 5.5) * 0.016 * (1.0 if wave_drive > 0.0 else 0.0)

        # Spinner keeps turning while morphing toward / in loader mode.
        if morph > 0.02 or state in ("processing", "starting"):
            self._loader_angle += 0.085 * (0.35 + morph * 0.65)

        mid_y = h * 0.5
        max_h = h * 0.38
        margin_x = w * 0.06

        band = QLinearGradient(0, mid_y - max_h, 0, mid_y + max_h)
        band.setColorAt(0.0, QColor(0, 0, 0, 0))
        band.setColorAt(0.5, glow)
        band.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, int(mid_y - max_h), w, int(max_h * 2), band)

        cx, cy = w * 0.5, h * 0.5
        loader_r = min(w, h) * 0.30

        path = QPainterPath()
        for i in range(SAMPLES):
            wx, wy = self._wave_point(w, h, margin_x, mid_y, max_h, t, amp, i)
            lx, ly = self._loader_point(cx, cy, loader_r, i)
            tx = wx * (1.0 - morph) + lx * morph
            ty = wy * (1.0 - morph) + ly * morph
            # Soft follow so morph + wave never snap.
            self._display_x[i] = self._display_x[i] * 0.70 + (tx / w) * 0.30
            self._display_y[i] = self._display_y[i] * 0.70 + (ty / h) * 0.30
            x = self._display_x[i] * w
            y = self._display_y[i] * h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        thickness = 1.8 + amp * 0.6 + morph * 0.8
        glow_pen = QPen(QColor(base.red(), base.green(), base.blue(), 70))
        glow_pen.setWidthF(4.0 + amp * 2.0 + morph * 1.5)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, glow_pen)

        pen = QPen(base)
        pen.setWidthF(thickness)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)

        # End caps only in wave mode (fade out as loader takes over).
        if morph < 0.85:
            alpha = int(90 * (1.0 - morph))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(base.red(), base.green(), base.blue(), alpha))
            painter.drawEllipse(QPointF(margin_x, mid_y), 2.0, 2.0)
            painter.drawEllipse(QPointF(w - margin_x, mid_y), 2.0, 2.0)

        if error and self._message:
            painter.setPen(QColor(255, 140, 160, 200))
            font = painter.font()
            font.setPixelSize(9)
            painter.setFont(font)
            painter.drawText(
                8,
                h - 10,
                self._message[:42] + ("…" if len(self._message) > 42 else ""),
            )

        painter.end()

    def _wave_point(
        self,
        w: float,
        h: float,
        margin_x: float,
        mid_y: float,
        max_h: float,
        t: float,
        amp: float,
        i: int,
    ) -> tuple[float, float]:
        u = i / (SAMPLES - 1)
        x = margin_x + (w - 2 * margin_x) * u
        y = mid_y + self._wave_sample(t, amp, i) * max_h
        return x, y

    def _wave_sample(self, t: float, amp: float, i: int) -> float:
        if amp <= 0.0:
            return 0.0
        x = i / (SAMPLES - 1)
        phase = self._wave_phase
        bright = 0.15 + 0.55 * amp
        return amp * (
            0.50 * math.sin(x * math.pi * 3.2 + phase * 1.1)
            + 0.28 * math.sin(x * math.pi * 7.0 + phase * 1.9 + t * 2.3)
            + bright * 0.18 * math.sin(x * math.pi * 13.0 + phase * 2.7 + t * 5.1)
            + bright * 0.10 * math.sin(x * math.pi * 21.0 + t * 9.0 + i * 0.15)
        )

    def _loader_point(
        self, cx: float, cy: float, radius: float, i: int
    ) -> tuple[float, float]:
        """Points along an incomplete spinning arc."""
        u = i / (SAMPLES - 1)
        angle = self._loader_angle + u * LOADER_SPAN
        return cx + math.cos(angle) * radius, cy + math.sin(angle) * radius

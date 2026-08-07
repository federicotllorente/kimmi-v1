from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath, QLinearGradient
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from bridge import PipelineState

DEEP = QColor(2, 8, 15)
SAMPLES = 96


class WaveGLWidget(QOpenGLWidget):
    """Horizontal oscilloscope-style sound wave on an OpenGL-backed surface."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state: PipelineState = "idle"
        self._amplitude = 0.08
        self._message = ""
        self._t0 = time.monotonic()
        self._phase = 0.0

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
        amp = self._amplitude
        state = self._state
        error = state == "error"

        if error:
            base = QColor(255, 64, 96)
            glow = QColor(255, 40, 70, int(40 + amp * 80))
        else:
            base = QColor(0, 212, 255)
            glow = QColor(0, 180, 255, int(35 + amp * 70))

        speed = {
            "idle": 0.0,
            "listening": 0.0,
            "processing": 0.0,
            "responding": 6.0,
            "error": 0.0,
        }.get(state, 0.0)
        self._phase += speed * 0.016

        mid_y = h * 0.5
        max_h = h * 0.38
        margin = w * 0.06

        # Soft under-glow along the wave band.
        band = QLinearGradient(0, mid_y - max_h, 0, mid_y + max_h)
        band.setColorAt(0.0, QColor(0, 0, 0, 0))
        band.setColorAt(0.5, glow)
        band.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, int(mid_y - max_h), w, int(max_h * 2), band)

        path = QPainterPath()
        for i in range(SAMPLES):
            x = margin + (w - 2 * margin) * (i / (SAMPLES - 1))
            y = mid_y + self._sample(state, t, amp, i) * max_h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # Glow stroke then crisp stroke.
        glow_pen = QPen(QColor(base.red(), base.green(), base.blue(), 70))
        glow_pen.setWidthF(5.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, glow_pen)

        pen = QPen(base)
        pen.setWidthF(2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, pen)

        # Center reference dots (subtle).
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(base.red(), base.green(), base.blue(), 90))
        painter.drawEllipse(QPointF(margin, mid_y), 2.0, 2.0)
        painter.drawEllipse(QPointF(w - margin, mid_y), 2.0, 2.0)

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

    def _sample(self, state: PipelineState, t: float, amp: float, i: int) -> float:
        """Normalized Y sample in [-1, 1]. Only animates while the assistant speaks."""
        if state != "responding":
            return 0.0

        x = i / (SAMPLES - 1)
        phase = self._phase
        return amp * (
            0.45 * math.sin(x * math.pi * 5 + phase * 1.7)
            + 0.30 * math.sin(x * math.pi * 11 + phase * 2.4 + t * 3)
            + 0.20 * math.sin(x * math.pi * 17 + t * 13)
            + 0.05 * math.sin(x * math.pi * 29 + phase * 5)
        )

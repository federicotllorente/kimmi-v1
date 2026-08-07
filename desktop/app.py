from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from bridge import EngineBridge
from wave_gl import WaveGLWidget


WINDOW_SIZE = 200


class KimmiWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KIMMI")
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowTitleHint
        )

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.wave = WaveGLWidget(central)
        layout.addWidget(self.wave)
        self.setCentralWidget(central)

        self.bridge = EngineBridge(self)
        self.bridge.status_updated.connect(self.wave.set_status)

    def start_engine(self) -> None:
        # Failures become visual "error" (red wave); window stays open.
        self.bridge.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.bridge.stop()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("KIMMI")

    window = KimmiWindow()
    window.start_engine()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from bridge import EngineBridge, ui_bus_mode
from mosquitto_supervisor import MosquittoSupervisor
from mqtt_client import MqttClient
from wave_gl import WaveGLWidget


WINDOW_SIZE = 200
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSETS_DIR / "kimmi.png"


def _ensure_icon() -> QIcon:
    """Load tray icon; synthesize a simple circle if the asset is missing."""
    if ICON_PATH.exists():
        return QIcon(str(ICON_PATH))
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#2EC4B6"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 56, 56)
    painter.setBrush(QColor("#0B132B"))
    painter.drawEllipse(22, 22, 20, 20)
    painter.end()
    pix.save(str(ICON_PATH), "PNG")
    return QIcon(pix)


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

        self.mosquitto = MosquittoSupervisor()
        self.mqtt = MqttClient(self)
        self.mqtt.state_received.connect(self.bridge.apply_mqtt_state)
        self.mqtt.notify_received.connect(self._on_notify)
        self.mqtt.tts_active_received.connect(self.bridge.apply_tts_active)
        self.mqtt.connection_changed.connect(self.bridge.set_mqtt_connected)

        self._icon = _ensure_icon()
        self.setWindowIcon(self._icon)
        self.tray = QSystemTrayIcon(self._icon, self)
        self.tray.setToolTip("Kimmi")
        quit_action = QAction("Quit Kimmi", self)
        quit_action.triggered.connect(self.close)
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def start_services(self) -> None:
        mode = ui_bus_mode()
        need_mqtt = mode in ("mqtt", "dual")

        if need_mqtt:
            ok, message = self.mosquitto.start()
            print(f"[desktop] Mosquitto: {message}", flush=True)
            if not ok and mode == "mqtt":
                self.bridge.fail(message)
                return
            if ok:
                if not self.mqtt.start():
                    if mode == "mqtt":
                        self.bridge.fail("MQTT client failed to start")
                        return
                else:
                    self.bridge.set_mqtt_connected(self.mqtt.connected)
            elif mode == "dual":
                print("[desktop] MQTT unavailable; dual mode will use stdout", flush=True)

        self.bridge.start()

    def _on_notify(self, title: str, body: str, level: str) -> None:
        icon = QSystemTrayIcon.MessageIcon.Information
        if level in ("error", "critical"):
            icon = QSystemTrayIcon.MessageIcon.Critical
        elif level in ("warn", "warning"):
            icon = QSystemTrayIcon.MessageIcon.Warning
        self.tray.showMessage(title or "Kimmi", body or "", icon, 8000)

    def closeEvent(self, event) -> None:  # noqa: N802
        # Order: MQTT client → engine → mosquitto child → tray (§0.1 / Fase A plan).
        try:
            self.mqtt.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.bridge.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.mosquitto.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.tray.hide()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("KIMMI")
    # Keep running if the window is hidden but tray is visible (optional later).
    app.setQuitOnLastWindowClosed(True)

    window = KimmiWindow()
    window.start_services()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

import theme
from aspect_lock import handle_sizing
from bridge import EngineBridge, ui_bus_mode
from chat_window import ChatWindow
from mosquitto_supervisor import MosquittoSupervisor
from mqtt_client import MqttClient
from wave_gl import WaveGLWidget


# Initial size; the window can be resized but always stays square, and the
# wave scales with it (see aspect_lock.py and WaveGLWidget)
WINDOW_SIZE = 200
MIN_WINDOW_SIZE = 140
WINDOW_ASPECT = 1.0
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
    painter.setBrush(theme.TEAL)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 56, 56)
    painter.setBrush(theme.DEEP)
    painter.drawEllipse(22, 22, 20, 20)
    painter.end()
    pix.save(str(ICON_PATH), "PNG")
    return QIcon(pix)


class KimmiWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KIMMI")
        self.resize(WINDOW_SIZE, WINDOW_SIZE)
        self.setMinimumSize(MIN_WINDOW_SIZE, MIN_WINDOW_SIZE)
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

        # Second window: conversation + tool cards (only fed over MQTT)
        self.chat = ChatWindow()
        self._icon = _ensure_icon()
        self.chat.setWindowIcon(self._icon)
        self.mqtt.chat_message_received.connect(self.chat.add_message)
        self.mqtt.intent_received.connect(self.chat.add_intent)
        self.mqtt.tool_call_received.connect(self.chat.update_tool)
        self.mqtt.connection_changed.connect(self.chat.set_connected)
        self.bridge.status_updated.connect(self.chat.set_status)
        self._chat_placed = False

        self.setWindowIcon(self._icon)
        self.tray = QSystemTrayIcon(self._icon, self)
        self.tray.setToolTip("Kimmi")
        chat_action = QAction("Show chat", self)
        chat_action.triggered.connect(self.show_chat)
        quit_action = QAction("Quit Kimmi", self)
        quit_action.triggered.connect(self.close)
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction(chat_action)
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def show_chat(self) -> None:
        if not self._chat_placed:
            # First time: open it just to the right of the wave window
            frame = self.frameGeometry()
            self.chat.move(frame.right() + 12, frame.top())
            self._chat_placed = True
        self.chat.show()
        self.chat.raise_()
        self.chat.activateWindow()

    def start_services(self) -> None:
        mode = ui_bus_mode()
        need_mqtt = mode in ("mqtt", "dual")
        if not need_mqtt:
            self.chat.set_connected(False)

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

    def nativeEvent(self, event_type, message):  # noqa: N802
        # Keep the window square while the user drags an edge or corner
        if event_type == b"windows_generic_MSG":
            min_width = int(MIN_WINDOW_SIZE * self.devicePixelRatioF())
            if handle_sizing(int(message), WINDOW_ASPECT, min_width):
                return True, 1
        return super().nativeEvent(event_type, message)

    def _on_notify(self, title: str, body: str, level: str) -> None:
        icon = QSystemTrayIcon.MessageIcon.Information
        if level in ("error", "critical"):
            icon = QSystemTrayIcon.MessageIcon.Critical
        elif level in ("warn", "warning"):
            icon = QSystemTrayIcon.MessageIcon.Warning
        self.tray.showMessage(title or "Kimmi", body or "", icon, 8000)

    def closeEvent(self, event) -> None:  # noqa: N802
        # Order: chat window → MQTT client → engine → mosquitto child → tray (§0.1 / Fase A plan).
        try:
            self.chat.close()
        except Exception:  # noqa: BLE001
            pass
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
    theme.load_fonts()
    # Keep running if the window is hidden but tray is visible (optional later).
    app.setQuitOnLastWindowClosed(True)

    window = KimmiWindow()
    window.start_services()
    window.show()
    window.show_chat()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

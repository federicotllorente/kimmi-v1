"""Qt-facing MQTT subscriber for Kimmi desktop (pipeline state, notify, chat, tools)."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


class MqttClient(QObject):
    """Subscribe to kimmi topics and emit Qt signals on the main thread."""

    state_received = Signal(str, str)  # state, message
    notify_received = Signal(str, str, str)  # title, body, level
    tts_active_received = Signal(bool)
    connection_changed = Signal(bool)  # connected?
    # Chat window: conversation lines, router decisions and tool calls
    chat_message_received = Signal(str, str, int)  # role, text, turn_id
    intent_received = Signal(dict)
    tool_call_received = Signal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.host = _env("KIMMI_MQTT_HOST", "127.0.0.1")
        self.port = int(_env("KIMMI_MQTT_PORT", "1883"))
        self.prefix = _env("KIMMI_MQTT_PREFIX", "kimmi").strip().strip("/")
        self._client: Any = None
        self._connected = False
        self._started = False
        self._stopping = False
        self._lock = threading.Lock()
        # Dedupe toasts from notify + jobs/done for the same job_id.
        self._recent_toast_keys: dict[str, float] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        if self._started:
            return True
        if mqtt is None:
            print("[desktop][MQTT]: paho-mqtt not installed", flush=True)
            return False

        self._stopping = False
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"kimmi-desktop-{os.getpid()}",
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        self._client = client

        try:
            client.connect_async(self.host, self.port, keepalive=30)
            client.loop_start()
            self._started = True
            # Wait briefly for first connect.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not self._connected:
                time.sleep(0.05)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[desktop][MQTT]: failed to start ({exc})", flush=True)
            self._client = None
            self._started = False
            return False

    def stop(self) -> None:
        self._stopping = True
        client = self._client
        self._client = None
        self._started = False
        was = self._connected
        self._connected = False
        if was:
            self.connection_changed.emit(False)
        if client is None:
            return
        try:
            client.loop_stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        rc = int(getattr(reason_code, "value", reason_code))
        if rc != 0:
            print(f"[desktop][MQTT]: connect refused (rc={rc})", flush=True)
            self._connected = False
            self.connection_changed.emit(False)
            return
        self._connected = True
        topics = [
            (f"{self.prefix}/pipeline/state", 0),
            (f"{self.prefix}/notify", 1),
            (f"{self.prefix}/tts/active", 0),
            (f"{self.prefix}/jobs/done", 1),
            (f"{self.prefix}/jobs/failed", 1),
            (f"{self.prefix}/chat/message", 1),
            (f"{self.prefix}/chat/intent", 1),
            (f"{self.prefix}/tools/call", 1),
        ]
        for topic, qos in topics:
            client.subscribe(topic, qos=qos)
        print(f"[desktop][MQTT]: connected to {self.host}:{self.port}", flush=True)
        self.connection_changed.emit(True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        self._connected = False
        if not self._stopping:
            print("[desktop][MQTT]: disconnected", flush=True)
        self.connection_changed.emit(False)

    def _should_emit_toast(self, key: str) -> bool:
        now = time.time()
        # Drop stale keys
        stale = [k for k, ts in self._recent_toast_keys.items() if now - ts > 8.0]
        for k in stale:
            self._recent_toast_keys.pop(k, None)
        if key in self._recent_toast_keys:
            return False
        self._recent_toast_keys[key] = now
        return True

    def _emit_notify(self, title: str, body: str, level: str, dedupe_key: str) -> None:
        if not self._should_emit_toast(dedupe_key):
            return
        self.notify_received.emit(title, body, level)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        topic = msg.topic or ""
        if topic.endswith("/pipeline/state"):
            state = str(payload.get("state") or "")
            message = str(payload.get("message") or "")
            if state:
                self.state_received.emit(state, message)
        elif topic.endswith("/notify"):
            title = str(payload.get("title") or "Kimmi")
            body = str(payload.get("body") or "")
            level = str(payload.get("level") or "info")
            job_id = str(payload.get("job_id") or "")
            key = f"job:{job_id}" if job_id else f"notify:{title}:{body}:{payload.get('ts')}"
            self._emit_notify(title, body, level, key)
        elif topic.endswith("/jobs/done") or topic.endswith("/jobs/failed"):
            job_id = str(payload.get("job_id") or "")
            title_job = str(payload.get("title") or "job")
            ok = bool(payload.get("ok", topic.endswith("/jobs/done")))
            body = (
                f"Finished: {title_job}"
                if ok
                else f"Failed: {title_job}"
            )
            level = "info" if ok else "error"
            key = f"job:{job_id}" if job_id else f"jobs:{title_job}:{payload.get('ts')}"
            self._emit_notify("Kimmi", body, level, key)
        elif topic.endswith("/tts/active"):
            self.tts_active_received.emit(bool(payload.get("active")))
        elif topic.endswith("/chat/message"):
            role = str(payload.get("role") or "")
            text = str(payload.get("text") or "")
            if role and text:
                self.chat_message_received.emit(role, text, int(payload.get("turn_id") or 0))
        elif topic.endswith("/chat/intent"):
            self.intent_received.emit(dict(payload))
        elif topic.endswith("/tools/call"):
            if payload.get("call_id"):
                self.tool_call_received.emit(dict(payload))

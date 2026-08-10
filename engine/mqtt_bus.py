"""MQTT control bus for Kimmi (pipeline state, notify, job stubs).

Transport only — no business logic. Safe if the broker is down: publishes
become no-ops after a clear log, and the voice loop keeps running.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


class MqttBus:
    """Publisher for kimmi/{pipeline,notify,tts,jobs} topics."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        self.host = host or _env("KIMMI_MQTT_HOST", "127.0.0.1")
        self.port = int(port if port is not None else _env("KIMMI_MQTT_PORT", "1883"))
        self.prefix = (prefix or _env("KIMMI_MQTT_PREFIX", "kimmi")).strip().strip("/")
        self._client: Any = None
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._stopping = False

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> bool:
        """Connect and start the network loop. Returns True if start was attempted."""
        if self._started:
            return True
        if mqtt is None:
            print("[MQTT]: paho-mqtt not installed; bus disabled", flush=True)
            return False

        self._stopping = False
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"kimmi-engine-{os.getpid()}",
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect

        self._client = client
        try:
            client.connect_async(self.host, self.port, keepalive=30)
            client.loop_start()
            self._started = True
            # Brief wait so READY can publish soon after boot.
            self._connected.wait(timeout=2.0)
            if not self.connected:
                print(
                    f"[MQTT]: waiting for broker at {self.host}:{self.port} "
                    "(will retry in background)",
                    flush=True,
                )
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[MQTT]: failed to start ({exc})", flush=True)
            self._client = None
            self._started = False
            return False

    def stop(self) -> None:
        """Disconnect and stop the network loop (idempotent)."""
        self._stopping = True
        client = self._client
        self._client = None
        self._started = False
        self._connected.clear()
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

    def publish_state(self, state: str, message: str = "") -> None:
        payload = {
            "state": state,
            "message": message or "",
            "ts": time.time(),
        }
        self._publish(f"{self.prefix}/pipeline/state", payload, qos=0, retain=True)

    def publish_notify(
        self,
        title: str,
        body: str,
        level: str = "info",
        job_id: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "title": title,
            "body": body,
            "level": level,
            "ts": time.time(),
        }
        if job_id:
            payload["job_id"] = job_id
        self._publish(f"{self.prefix}/notify", payload, qos=1, retain=False)

    def publish_tts_active(self, active: bool) -> None:
        payload = {"active": bool(active), "ts": time.time()}
        self._publish(f"{self.prefix}/tts/active", payload, qos=0, retain=True)

    # --- Job stubs (Phase B callers) ---

    def publish_job_queued(self, job_id: str, kind: str, title: str) -> None:
        self._publish_job("queued", job_id, kind, title, ok=True)

    def publish_job_progress(
        self,
        job_id: str,
        kind: str,
        title: str,
        stage: str = "",
        pct: Optional[float] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "title": title,
            "stage": stage,
            "ts": time.time(),
        }
        if pct is not None:
            payload["pct"] = pct
        self._publish(f"{self.prefix}/jobs/progress", payload, qos=0, retain=False)

    def publish_job_done(
        self,
        job_id: str,
        kind: str,
        title: str,
        *,
        ok: bool = True,
        summary_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        topic = "done" if ok else "failed"
        self._publish_job(topic, job_id, kind, title, ok=ok, summary_path=summary_path, error=error)

    def publish_job_failed(
        self,
        job_id: str,
        kind: str,
        title: str,
        error: str = "",
    ) -> None:
        self.publish_job_done(job_id, kind, title, ok=False, error=error)

    def _publish_job(
        self,
        topic_suffix: str,
        job_id: str,
        kind: str,
        title: str,
        *,
        ok: bool,
        summary_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "title": title,
            "ok": ok,
            "ts": time.time(),
        }
        if summary_path:
            payload["summary_path"] = summary_path
        if error:
            payload["error"] = error
        self._publish(f"{self.prefix}/jobs/{topic_suffix}", payload, qos=1, retain=False)

    def _publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        qos: int,
        retain: bool,
    ) -> None:
        client = self._client
        if client is None or self._stopping:
            return
        try:
            data = json.dumps(payload, ensure_ascii=False)
            with self._lock:
                client.publish(topic, data, qos=qos, retain=retain)
        except Exception as exc:  # noqa: BLE001
            print(f"[MQTT]: publish failed on {topic}: {exc}", flush=True)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        # paho v2: reason_code is ReasonCode; success is 0.
        rc = int(getattr(reason_code, "value", reason_code))
        if rc == 0:
            self._connected.set()
            print(f"[MQTT]: connected to {self.host}:{self.port}", flush=True)
        else:
            self._connected.clear()
            print(f"[MQTT]: connect refused (rc={rc})", flush=True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        self._connected.clear()
        if not self._stopping:
            print("[MQTT]: disconnected; reconnecting in background", flush=True)


# Process-wide bus used by assistant.py
bus = MqttBus()

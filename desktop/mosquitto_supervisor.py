"""Spawn / stop a local Mosquitto broker owned by the desktop app."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def mqtt_host() -> str:
    return _env("KIMMI_MQTT_HOST", "127.0.0.1")


def mqtt_port() -> int:
    return int(_env("KIMMI_MQTT_PORT", "1883"))


def port_is_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_mosquitto_bin() -> Optional[Path]:
    override = os.environ.get("KIMMI_MOSQUITTO_BIN", "").strip()
    if override:
        path = Path(override)
        return path if path.exists() else None

    which = shutil.which("mosquitto")
    if which:
        return Path(which)

    if sys.platform == "win32":
        candidates = [
            Path(r"C:\Program Files\mosquitto\mosquitto.exe"),
            Path(r"C:\Program Files (x86)\mosquitto\mosquitto.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


class MosquittoSupervisor:
    """Start Mosquitto as a child process when the port is free."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host = host or mqtt_host()
        self.port = port if port is not None else mqtt_port()
        self._proc: Optional[subprocess.Popen] = None
        self._conf_path: Optional[Path] = None
        self._owned = False
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None

    @property
    def owned(self) -> bool:
        return self._owned

    @property
    def running(self) -> bool:
        if self._owned and self._proc is not None:
            return self._proc.poll() is None
        return port_is_open(self.host, self.port)

    def start(self, ready_timeout_sec: float = 8.0) -> tuple[bool, str]:
        """Ensure a broker is listening. Returns (ok, message)."""
        if port_is_open(self.host, self.port):
            self._owned = False
            self._proc = None
            return True, f"Reusing existing MQTT broker at {self.host}:{self.port}"

        binary = find_mosquitto_bin()
        if binary is None:
            return (
                False,
                "mosquitto not found (install it or set KIMMI_MOSQUITTO_BIN)",
            )

        self._tmpdir = tempfile.TemporaryDirectory(prefix="kimmi-mosquitto-")
        conf = Path(self._tmpdir.name) / "mosquitto.conf"
        # Bind only to localhost; anonymous OK for local Kimmi bus.
        conf.write_text(
            "\n".join(
                [
                    f"listener {self.port} {self.host}",
                    "allow_anonymous true",
                    "persistence false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self._conf_path = conf

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )

        try:
            self._proc = subprocess.Popen(
                [str(binary), "-c", str(conf)],
                cwd=str(conf.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._cleanup_temp()
            return False, f"Failed to start mosquitto: {exc}"

        self._owned = True
        deadline = time.monotonic() + ready_timeout_sec
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                self._owned = False
                self._proc = None
                self._cleanup_temp()
                return False, f"mosquitto exited early (code {code})"
            if port_is_open(self.host, self.port):
                return True, f"Started mosquitto at {self.host}:{self.port}"
            time.sleep(0.1)

        self.stop()
        return False, f"Timed out waiting for mosquitto on {self.host}:{self.port}"

    def stop(self) -> None:
        """Stop the child broker only if we own it."""
        if not self._owned:
            self._proc = None
            self._cleanup_temp()
            return

        proc = self._proc
        self._proc = None
        self._owned = False
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            except OSError:
                pass
        self._cleanup_temp()

    def _cleanup_temp(self) -> None:
        self._conf_path = None
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:  # noqa: BLE001
                pass
            self._tmpdir = None

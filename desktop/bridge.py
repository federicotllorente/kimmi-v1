from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Signal

# idle | listening | processing | responding | starting | error
PipelineState = str

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "engine"
ENGINE_SCRIPT = ENGINE_DIR / "assistant.py"

if sys.platform == "win32":
    ENGINE_PYTHON = ENGINE_DIR / "proj-env" / "Scripts" / "python.exe"
else:
    ENGINE_PYTHON = ENGINE_DIR / "proj-env" / "bin" / "python"

_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "Error:",
    "Exception:",
    "CUDA out of memory",
    "ModuleNotFoundError",
    "FileNotFoundError",
    "OSError:",
    "RuntimeError:",
)

_VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
_SPEECH_MARKERS = ("[Kimmi]:", "[Confirm]:")

# The engine prints these in English regardless of the language it speaks in,
# so this table stays valid across locales.
READY_MARKER = "KIMMI READY"


def ui_bus_mode() -> str:
    """mqtt | stdout | dual (default)."""
    raw = (os.environ.get("KIMMI_UI_BUS") or "dual").strip().lower()
    if raw in ("mqtt", "stdout", "dual"):
        return raw
    return "dual"


def parse_engine_line(line: str) -> Optional[tuple[PipelineState, str]]:
    """Map engine stdout markers to pipeline state. Returns (state, message) or None."""
    text = line.strip()
    if not text:
        return None

    if any(marker in text for marker in _ERROR_MARKERS):
        return "error", text

    if READY_MARKER in text or "KIMMI IDLE" in text:
        return "idle", text

    if (
        "[Wake Detected" in text
        or "[VAD]:" in text
        or "Listening (active session)" in text
        or "Using barge-in audio" in text
    ):
        return "listening", text

    if any(
        marker in text
        for marker in (
            "[You]:",
            "[Intent]:",
            "[LLM]:",
            "[Tools]:",
            "[Closing]:",
            "[Session]:",
            "[Memory]:",
            "[RAG]:",
            "[Locale]:",
        )
    ):
        return "processing", text

    if any(marker in text for marker in _SPEECH_MARKERS):
        return "responding", text

    if "[Running]:" in text:
        return "processing", text

    return None


def _char_burst(ch: str) -> float:
    """Approximate spoken intensity for one character (Spanish-ish)."""
    if ch.isspace():
        return 0.04
    if ch in ",.;:…—-":
        return 0.06
    if ch in "!?¡¿":
        return 0.95
    if ch in _VOWELS:
        return 0.72 + random.uniform(0.0, 0.22)
    if ch.isdigit():
        return 0.42
    # consonants / other
    return 0.28 + random.uniform(0.0, 0.32)


def _char_duration(ch: str) -> float:
    """Rough TTS pacing so streamed text is stretched toward spoken timing."""
    if ch.isspace():
        return 0.075
    if ch in ".!?¡¿":
        return 0.20
    if ch in ",.;:":
        return 0.11
    return 0.052


class EngineBridge(QObject):
    """Runs the voice engine as a subprocess and emits status for the HUD."""

    status_updated = Signal(str, float, str)  # state, amplitude, message
    _state_from_engine = Signal(str, str)  # state, message (queued from reader thread)
    _speech_text = Signal(str)  # spoken text delta (queued from reader thread)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state: PipelineState = "starting"
        self.message: str = "Starting engine..."
        self.amplitude: float = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._stopping = False
        self._ready = False
        self._t0 = time.monotonic()
        self._bus_mode = ui_bus_mode()
        self._mqtt_connected = False
        self._tts_active = False
        self._use_stdout_envelope = self._bus_mode in ("dual", "stdout")

        # Text→speech envelope (main thread only).
        self._burst_queue: Deque[Tuple[float, float]] = deque()
        self._next_burst_at = 0.0
        self._speech_until = 0.0
        self._speech_energy = 0.0
        self._speech_gate = 0.0  # smooth 0..1 open/close

        self._amp_timer = QTimer(self)
        self._amp_timer.setInterval(33)  # ~30 Hz envelope
        self._amp_timer.timeout.connect(self._tick_amplitude)
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(500)
        self._watch_timer.timeout.connect(self._watch_process)
        self._state_from_engine.connect(self._apply_state)
        self._speech_text.connect(self._ingest_speech)

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def set_mqtt_connected(self, connected: bool) -> None:
        self._mqtt_connected = bool(connected)

    def apply_mqtt_state(self, state: str, message: str = "") -> None:
        """HUD state from MQTT (preferred in mqtt/dual when connected)."""
        if self._bus_mode == "stdout":
            return
        if self._bus_mode == "dual" and not self._mqtt_connected:
            return
        self._apply_state(state, message)

    def apply_tts_active(self, active: bool) -> None:
        self._tts_active = bool(active)
        if active and self.state not in ("error", "starting") and self._ready:
            if self.state != "responding":
                self.state = "responding"
                self._emit()

    def start(self) -> None:
        if self._running and self.is_alive:
            return
        if not ENGINE_PYTHON.exists():
            self._fail(f"Engine Python not found: {ENGINE_PYTHON}")
            return
        if not ENGINE_SCRIPT.exists():
            self._fail(f"Engine script not found: {ENGINE_SCRIPT}")
            return

        self._stopping = False
        self._running = True
        self._ready = False
        self._t0 = time.monotonic()
        self._bus_mode = ui_bus_mode()
        self._use_stdout_envelope = self._bus_mode in ("dual", "stdout")
        self.state = "starting"
        self.message = "Starting engine..."
        self.amplitude = 0.0
        self._burst_queue.clear()
        self._next_burst_at = 0.0
        self._speech_until = 0.0
        self._speech_energy = 0.0
        self._speech_gate = 0.0
        self._tts_active = False

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self._proc = subprocess.Popen(
                [str(ENGINE_PYTHON), "-u", str(ENGINE_SCRIPT)],
                cwd=str(ENGINE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._fail(f"Failed to start engine: {exc}")
            return

        self._reader_thread = threading.Thread(
            target=self._read_stdout_thread,
            name="engine-stdout",
            daemon=True,
        )
        self._reader_thread.start()
        self._amp_timer.start()
        self._watch_timer.start()
        self._emit()

    def stop(self) -> None:
        self._stopping = True
        self._running = False
        self._amp_timer.stop()
        self._watch_timer.stop()

        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3.0)
            except OSError:
                pass
        self._proc = None

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None

    def fail(self, message: str) -> None:
        """Surface a fatal desktop/engine error on the HUD."""
        self._fail(message)

    def _fail(self, message: str) -> None:
        print(f"[desktop] {message}", flush=True)
        self._running = False
        self._watch_timer.stop()
        if not self._amp_timer.isActive():
            self._amp_timer.start()
        self._burst_queue.clear()
        self._speech_energy = 0.0
        self._speech_gate = 0.0
        self.state = "error"
        self.message = message
        self.amplitude = 0.0
        self._emit()

    def _emit(self) -> None:
        self.status_updated.emit(
            self.state,
            round(self.amplitude, 3),
            self.message,
        )

    def _stdout_drives_state(self) -> bool:
        if self._bus_mode == "stdout":
            return True
        if self._bus_mode == "mqtt":
            return False
        # dual: stdout fallback when MQTT is down
        return not self._mqtt_connected

    def _apply_state(self, state: PipelineState, message: str = "") -> None:
        if self.state == "error" and state != "error" and not self._stopping:
            if message:
                self.message = message
            return

        # Stay in starting until the engine is ready (ignore other chatter while loading).
        if not self._ready:
            ready_hint = READY_MARKER in (message or "") or (
                state == "idle" and "READY" in (message or "").upper()
            )
            if state == "idle" and ready_hint:
                self._ready = True
                state = "idle"
            elif state == "error":
                pass
            else:
                # Keep showing the boot loader; refresh message if useful.
                if message and message != self.message:
                    self.message = message
                    self._emit()
                return

        changed = state != self.state or (message and message != self.message)
        self.state = state
        if message:
            self.message = message
        if changed:
            self._emit()

    def _ingest_speech(self, text: str) -> None:
        """Queue character bursts paced like speech (main thread)."""
        if not text or not self._use_stdout_envelope:
            return
        now = time.monotonic()
        t = max(now, self._next_burst_at)
        for ch in text:
            if ch == "\r":
                continue
            if ch == "\n":
                t += 0.12
                continue
            self._burst_queue.append((t, _char_burst(ch)))
            t += _char_duration(ch)
        self._next_burst_at = t
        # Hold the gate a bit past the last scheduled burst (TTS lag after tokens).
        self._speech_until = max(self._speech_until, t + 0.45)
        if self.state != "error" and self._stdout_drives_state():
            self.state = "responding"

    def _watch_process(self) -> None:
        if self._stopping or not self._running:
            return
        proc = self._proc
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            return
        if code == 0:
            self._fail("Engine exited unexpectedly")
        else:
            self._fail(f"Engine crashed (exit code {code})")

    def _read_stdout_thread(self) -> None:
        """Read engine stdout in chunks so streaming prints (no newline) are live."""
        proc = self._proc
        if not proc or not proc.stdout:
            return

        pending = ""
        active_marker: Optional[str] = None
        ingested = 0
        drive_state = True  # recomputed; stdout always drained

        try:
            while self._running:
                chunk = proc.stdout.read(256)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                print(text, end="", flush=True)
                pending += text
                drive_state = self._stdout_drives_state()

                if self._use_stdout_envelope:
                    if active_marker is None:
                        for marker in _SPEECH_MARKERS:
                            if marker in pending:
                                active_marker = marker
                                start = pending.find(marker) + len(marker)
                                content = pending[start:]
                                ingested = 0
                                preview = pending.replace("\n", " ").strip()[-120:]
                                if drive_state:
                                    self._state_from_engine.emit("responding", preview)
                                if content:
                                    self._speech_text.emit(content)
                                    ingested = len(content)
                                break
                    else:
                        start = pending.find(active_marker) + len(active_marker)
                        if start >= len(active_marker):
                            content = pending[start:]
                            if len(content) > ingested:
                                self._speech_text.emit(content[ingested:])
                                ingested = len(content)

                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    active_marker = None
                    ingested = 0
                    if not drive_state:
                        continue
                    parsed = parse_engine_line(line)
                    if parsed:
                        state, message = parsed
                        self._state_from_engine.emit(state, message)
        except Exception as exc:
            print(f"\n[desktop] Engine stdout reader error: {exc}", flush=True)
            if self._running and not self._stopping:
                self._state_from_engine.emit("error", f"Engine stdout error: {exc}")
        finally:
            if pending.strip():
                print(flush=True)
                if self._stdout_drives_state():
                    parsed = parse_engine_line(pending)
                    if parsed and self._running and not self._stopping:
                        state, message = parsed
                        self._state_from_engine.emit(state, message)

            if self._running and not self._stopping:
                proc = self._proc
                if proc is not None and proc.poll() is not None and proc.returncode not in (
                    0,
                    None,
                ):
                    self._state_from_engine.emit(
                        "error",
                        f"Engine crashed (exit code {proc.returncode})",
                    )
                elif proc is not None and proc.poll() is not None:
                    self._state_from_engine.emit("error", "Engine process ended")

    def _tick_amplitude(self) -> None:
        if not self._running and self.state != "error":
            return

        now = time.monotonic()

        if self.state == "error":
            self.amplitude = 0.0
            self._emit()
            return

        # Apply due character bursts → speech energy contour.
        while self._burst_queue and self._burst_queue[0][0] <= now:
            _, burst = self._burst_queue.popleft()
            self._speech_energy = min(1.0, self._speech_energy * 0.35 + burst)

        # Soft decay between bursts (syllable falloff).
        self._speech_energy *= 0.88

        speaking = (
            bool(self._burst_queue)
            or now < self._speech_until
            or self._speech_energy > 0.05
        )

        # Synthetic pulse when MQTT drives UI and we lack stdout speech envelope.
        if not speaking and self.state in ("responding", "listening"):
            if self._tts_active or self.state == "responding":
                pulse = 0.35 + 0.35 * abs(math.sin(now * 6.0))
                self.amplitude = pulse
                self._emit()
                return
            if self.state == "listening":
                pulse = 0.12 + 0.10 * abs(math.sin(now * 2.5))
                self.amplitude = pulse
                self._emit()
                return

        target_gate = 1.0 if speaking else 0.0
        # Asymmetric smoothing: faster attack, slower release (no hard cuts).
        if target_gate > self._speech_gate:
            self._speech_gate += (target_gate - self._speech_gate) * 0.22
        else:
            self._speech_gate += (target_gate - self._speech_gate) * 0.07

        if self._speech_gate < 0.01:
            self._speech_gate = 0.0
            self._speech_energy = 0.0

        # Contour rides on the gate so flat↔active blends smoothly.
        contour = 0.12 + 0.88 * self._speech_energy
        self.amplitude = max(0.0, min(1.0, self._speech_gate * contour))
        self._emit()

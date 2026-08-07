from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

# idle | listening | processing | responding | error
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


def parse_engine_line(line: str) -> Optional[tuple[PipelineState, str]]:
    """Map engine stdout markers to pipeline state. Returns (state, message) or None."""
    text = line.strip()
    if not text:
        return None

    if any(marker in text for marker in _ERROR_MARKERS):
        return "error", text

    if "JARVIS EN ESPERA" in text or "Volviendo a estado de espera" in text:
        return "idle", text

    if "[Wake Word Detectado" in text or "[VAD]:" in text:
        return "listening", text

    if any(
        marker in text
        for marker in ("[Tú]:", "[Intent]:", "[LLM]:", "[Tools]:", "[Cierre]:")
    ):
        return "processing", text

    if "[Jarvis]:" in text or "[Confirmación]:" in text:
        return "responding", text

    if "Jarvis ejecutando" in text:
        return "processing", text

    return None


class EngineBridge(QObject):
    """Runs the voice engine as a subprocess and emits status for the HUD."""

    status_updated = Signal(str, float, str)  # state, amplitude, message
    _state_from_engine = Signal(str, str)  # state, message (queued from reader thread)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state: PipelineState = "idle"
        self.message: str = ""
        self.amplitude: float = 0.08
        self._proc: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._stopping = False
        self._t0 = time.monotonic()
        self._amp_timer = QTimer(self)
        self._amp_timer.setInterval(50)  # ~20 Hz
        self._amp_timer.timeout.connect(self._tick_amplitude)
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(500)
        self._watch_timer.timeout.connect(self._watch_process)
        self._state_from_engine.connect(self._apply_state)

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

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
        self._t0 = time.monotonic()
        self.state = "idle"
        self.message = "Starting engine..."
        self.amplitude = 0.08

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self._proc = subprocess.Popen(
                [str(ENGINE_PYTHON), str(ENGINE_SCRIPT)],
                cwd=str(ENGINE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
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

    def _fail(self, message: str) -> None:
        print(f"[desktop] {message}", flush=True)
        self._running = False
        self._watch_timer.stop()
        # Keep amplitude timer so the red wave still breathes.
        if not self._amp_timer.isActive():
            self._amp_timer.start()
        self.state = "error"
        self.message = message
        self.amplitude = 0.2
        self._emit()

    def _emit(self) -> None:
        self.status_updated.emit(
            self.state,
            round(self.amplitude, 3),
            self.message,
        )

    def _apply_state(self, state: PipelineState, message: str = "") -> None:
        # Once in error (unexpected), keep it until restart — except intentional stop.
        if self.state == "error" and state != "error" and not self._stopping:
            if message:
                self.message = message
            return
        changed = state != self.state or (message and message != self.message)
        self.state = state
        if message:
            self.message = message
        if changed:
            self._emit()

    def _watch_process(self) -> None:
        if self._stopping or not self._running:
            return
        proc = self._proc
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            return
        # Process exited while we still expected it to run.
        if code == 0:
            self._fail("Engine exited unexpectedly")
        else:
            self._fail(f"Engine crashed (exit code {code})")

    def _read_stdout_thread(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return

        try:
            for line in proc.stdout:
                if not self._running:
                    break
                print(f"[engine] {line.rstrip()}", flush=True)
                parsed = parse_engine_line(line)
                if parsed:
                    state, message = parsed
                    self._state_from_engine.emit(state, message)
        except Exception as exc:
            print(f"[desktop] Engine stdout reader error: {exc}", flush=True)
            if self._running and not self._stopping:
                self._state_from_engine.emit("error", f"Engine stdout error: {exc}")
        finally:
            if self._running and not self._stopping:
                # Prefer exit-code watch for the final message; this covers pipe EOF early.
                proc = self._proc
                if proc is not None and proc.poll() is not None and proc.returncode not in (0, None):
                    self._state_from_engine.emit(
                        "error",
                        f"Engine crashed (exit code {proc.returncode})",
                    )
                elif proc is not None and proc.poll() is not None:
                    self._state_from_engine.emit("error", "Engine process ended")

    def _tick_amplitude(self) -> None:
        if not self._running and self.state != "error":
            return

        t = time.monotonic() - self._t0
        state = self.state

        if state == "responding":
            target = 0.55 + 0.35 * abs(math.sin(t * 5.5)) + random.uniform(0, 0.1)
        elif state == "error":
            target = 0.05
        else:
            # Flat line unless the assistant is speaking.
            target = 0.0

        self.amplitude = self.amplitude * 0.55 + min(1.0, max(0.0, target)) * 0.45
        self._emit()

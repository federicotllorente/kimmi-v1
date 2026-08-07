from __future__ import annotations

import asyncio
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .ws_hub import WsHub

PipelineState = str  # idle | listening | processing | responding

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "engine"
ENGINE_SCRIPT = ENGINE_DIR / "assistant.py"

if sys.platform == "win32":
    ENGINE_PYTHON = ENGINE_DIR / "proj-env" / "Scripts" / "python.exe"
else:
    ENGINE_PYTHON = ENGINE_DIR / "proj-env" / "bin" / "python"


def parse_engine_line(line: str) -> Optional[tuple[PipelineState, str]]:
    """Map engine stdout markers to pipeline state. Returns (state, message) or None."""
    text = line.strip()
    if not text:
        return None

    if "KIMMI EN ESPERA" in text or "Volviendo a estado de espera" in text:
        return "idle", text

    if "[Wake Word Detectado" in text or "[VAD]:" in text:
        return "listening", text

    if any(
        marker in text
        for marker in ("[Tú]:", "[Intent]:", "[LLM]:", "[Tools]:", "[Cierre]:")
    ):
        return "processing", text

    if "[Kimmi]:" in text or "[Confirmación]:" in text:
        return "responding", text

    if "Kimmi ejecutando" in text:
        return "processing", text

    return None


class EngineBridge:
    """Runs the voice engine as a subprocess and broadcasts parsed status events.

    Uses ``subprocess.Popen`` + a reader thread because uvicorn on Windows often
    runs a SelectorEventLoop that cannot create asyncio subprocesses.
    """

    def __init__(self, hub: WsHub) -> None:
        self.hub = hub
        self.state: PipelineState = "idle"
        self.message: str = ""
        self.amplitude: float = 0.08
        self._proc: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._amp_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._t0 = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "amplitude": round(self.amplitude, 3),
            "message": self.message,
        }

    async def start(self) -> None:
        if self._running:
            return
        if not ENGINE_PYTHON.exists():
            raise FileNotFoundError(f"Engine Python not found: {ENGINE_PYTHON}")
        if not ENGINE_SCRIPT.exists():
            raise FileNotFoundError(f"Engine script not found: {ENGINE_SCRIPT}")

        self._running = True
        self._t0 = time.monotonic()
        self._loop = asyncio.get_running_loop()

        env = os.environ.copy()
        # Force UTF-8 so Spanish prints don't break the line parser on Windows.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        creationflags = 0
        if sys.platform == "win32":
            # Allow clean terminate of the child process tree on Windows.
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

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
        self._reader_thread = threading.Thread(
            target=self._read_stdout_thread,
            name="engine-stdout",
            daemon=True,
        )
        self._reader_thread.start()
        self._amp_task = asyncio.create_task(self._amplitude_loop())
        await self._emit()

    async def stop(self) -> None:
        self._running = False

        if self._amp_task and not self._amp_task.done():
            self._amp_task.cancel()
            try:
                await self._amp_task
            except asyncio.CancelledError:
                pass
        self._amp_task = None

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
        self._loop = None

    async def _emit(self) -> None:
        await self.hub.broadcast(self.snapshot())

    def _schedule(self, coro: Any) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(coro, loop)

    async def _set_state(self, state: PipelineState, message: str = "") -> None:
        changed = state != self.state or (message and message != self.message)
        self.state = state
        if message:
            self.message = message
        if changed:
            await self._emit()

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
                    self._schedule(self._set_state(state, message))
        except Exception as exc:
            print(f"[backend] Engine stdout reader error: {exc}", flush=True)
        finally:
            if self._running:
                self._schedule(self._set_state("idle", "Engine process ended"))

    async def _amplitude_loop(self) -> None:
        """Synthesize a 0–1 amplitude driven by current pipeline state."""
        while self._running:
            t = time.monotonic() - self._t0
            state = self.state

            if state == "idle":
                target = 0.06 + 0.04 * (0.5 + 0.5 * math.sin(t * 0.9))
            elif state == "listening":
                target = 0.35 + 0.45 * abs(math.sin(t * 7.0)) + random.uniform(0, 0.15)
            elif state == "processing":
                pulse = 0.5 + 0.5 * math.sin(t * 4.5)
                target = 0.25 + 0.35 * pulse
            else:  # responding
                target = 0.55 + 0.35 * abs(math.sin(t * 5.5)) + random.uniform(0, 0.1)

            # Smooth toward target so the wave doesn't jump harshly.
            self.amplitude = self.amplitude * 0.65 + min(1.0, max(0.0, target)) * 0.35
            await self._emit()
            await asyncio.sleep(1 / 20)  # ~20 Hz

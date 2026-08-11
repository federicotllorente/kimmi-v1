"""Rolling partial STT while the user is still speaking.

Runs Whisper on a growing PCM buffer on a background thread. The capture loop
only feeds bytes and never blocks on transcription. Overlapping ticks are
skipped so a slow GPU job does not queue up.

Enable with ``KIMMI_PARTIAL_STT=1``. Final medium STT remains authoritative.
"""
from __future__ import annotations

import io
import threading
import time
import wave
from typing import Any, Dict, Optional


class PartialSttWorker:
    """Background rolling transcription over captured PCM int16 mono."""

    def __init__(
        self,
        model: Any,
        model_lock: threading.Lock,
        language: str,
        rate: int,
        interval_ms: int,
        min_audio_sec: float,
    ) -> None:
        self._model = model
        self._model_lock = model_lock
        self._language = language
        self._rate = rate
        self._interval_sec = max(0.05, interval_ms / 1000.0)
        self._min_bytes = max(1, int(min_audio_sec * rate * 2))  # int16 mono

        self._buf_lock = threading.Lock()
        self._buffer = bytearray()
        self._stop = threading.Event()
        self._busy = False
        self._last_text = ""
        self._calls = 0
        self._total_ms = 0.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="partial-stt",
            daemon=True,
        )
        self._thread.start()

    def feed(self, pcm_chunk: bytes) -> None:
        if not pcm_chunk:
            return
        with self._buf_lock:
            self._buffer.extend(pcm_chunk)

    def stop(self) -> Dict[str, float]:
        """Stop the worker and wait for the current tick to finish."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        return {"calls": float(self._calls), "total_ms": float(self._total_ms)}

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_sec):
            if self._busy:
                continue
            with self._buf_lock:
                if len(self._buffer) < self._min_bytes:
                    continue
                snapshot = bytes(self._buffer)
            self._busy = True
            try:
                self._transcribe_snapshot(snapshot)
            finally:
                self._busy = False

    def _transcribe_snapshot(self, pcm: bytes) -> None:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._rate)
            wf.writeframes(pcm)
        wav_buffer.seek(0)

        started = time.monotonic()
        text = ""
        try:
            with self._model_lock:
                segments_iter, _info = self._model.transcribe(
                    wav_buffer,
                    language=self._language,
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                parts = []
                for seg in segments_iter:
                    part = (seg.text or "").strip()
                    if part:
                        parts.append(part)
                text = " ".join(parts).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  [partial_stt]: transcribe failed ({exc})")
            return
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._calls += 1
            self._total_ms += elapsed_ms

        if not text or text == self._last_text:
            return
        self._last_text = text
        print(f"[You~]: {text}", flush=True)

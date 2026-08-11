"""Per-turn latency instrumentation for the voice pipeline.

Emits one line per turn prefixed with ``[perf]``. None of the stdout parsers in
``backend/app/engine_bridge.py`` or ``desktop/bridge.py`` match that prefix, so
these lines are inert for the UI.

Example output::

    [perf] turn=3 stt=412 rag=189 llm_ttft=233 synth1=890 ttfa=1310 total=3480

All values are milliseconds since the turn started, except ``stt``/``rag``/
``synth1`` which are stage durations, and ``spoken`` which is seconds of audio.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_OFF_VALUES = {"0", "false", "no", "off"}
ENABLED = os.environ.get("KIMMI_PERF", "1").strip().lower() not in _OFF_VALUES

# Render order for the report line; unknown fields are appended after these.
_FIELD_ORDER = (
    "earcon",
    "stt",
    "partial_n",
    "partial_ms",
    "lang_detect",
    "lang_switch",
    "rag",
    "llm_ttft",
    "synth1",
    "filler",
    "ttfa",
    "spoken",
    "total",
)


class TurnTimer:
    """Collects stage timings for a single conversation turn.

    Thread safe: the audio callback and the synthesis thread both write to it.
    """

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self._t0 = time.monotonic()
        self._fields: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def elapsed_ms(self) -> int:
        """Milliseconds since the turn started."""
        return self._elapsed_ms()

    def mark(self, field: str, once: bool = True) -> None:
        """Record milliseconds elapsed from turn start until now."""
        with self._lock:
            if once and field in self._fields:
                return
            self._fields[field] = str(self._elapsed_ms())

    def duration(self, field: str, milliseconds: float) -> None:
        """Record an explicit stage duration."""
        with self._lock:
            self._fields[field] = str(int(milliseconds))

    def value(self, field: str, value: object) -> None:
        """Record an arbitrary value (rendered verbatim)."""
        with self._lock:
            self._fields[field] = str(value)

    @contextmanager
    def span(self, field: str) -> Iterator[None]:
        """Time the wrapped block and store it as a stage duration."""
        started = time.monotonic()
        try:
            yield
        finally:
            self.duration(field, (time.monotonic() - started) * 1000.0)

    def report(self) -> None:
        if not ENABLED:
            return
        with self._lock:
            self._fields.setdefault("total", str(self._elapsed_ms()))
            ordered = [
                f"{key}={self._fields[key]}"
                for key in _FIELD_ORDER
                if key in self._fields
            ]
            extra = [
                f"{key}={value}"
                for key, value in self._fields.items()
                if key not in _FIELD_ORDER
            ]
        print(f"[perf] turn={self.turn_id} " + " ".join(ordered + extra), flush=True)


_current: Optional[TurnTimer] = None
_current_lock = threading.Lock()
_turn_counter = 0


def start_turn() -> TurnTimer:
    """Begin a new turn and make it the current one."""
    global _current, _turn_counter
    with _current_lock:
        _turn_counter += 1
        _current = TurnTimer(_turn_counter)
        return _current


def current() -> Optional[TurnTimer]:
    with _current_lock:
        return _current


def finish_turn() -> None:
    """Report the current turn (if any) and clear it."""
    global _current
    with _current_lock:
        timer = _current
        _current = None
    if timer is not None:
        timer.report()


def mark(field: str, once: bool = True) -> None:
    """Mark a field on the current turn; no-op when no turn is active."""
    timer = current()
    if timer is not None:
        timer.mark(field, once=once)


def elapsed_ms() -> Optional[int]:
    """Milliseconds since the current turn started, or None if there is none."""
    timer = current()
    return timer.elapsed_ms() if timer is not None else None


def duration(field: str, milliseconds: float) -> None:
    timer = current()
    if timer is not None:
        timer.duration(field, milliseconds)


def value(field: str, val: object) -> None:
    timer = current()
    if timer is not None:
        timer.value(field, val)


@contextmanager
def span(field: str) -> Iterator[None]:
    """Time a block against the current turn, if there is one."""
    timer = current()
    if timer is None:
        yield
        return
    with timer.span(field):
        yield

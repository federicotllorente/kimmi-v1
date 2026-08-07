"""Low-latency audio output and speech pipeline.

Design notes (full analysis in engine/ROADMAP.md):

- ``AudioSink`` keeps a single callback-driven output stream open for the whole
  process. Kokoro's sample rate is a constant 24 kHz, so the stream never needs
  reopening; that removes the per-sentence device warmup and the lead-in
  silence the previous implementation used to hide it, and makes consecutive
  chunks gapless.
- ``SpeechPipeline`` synthesizes on its own thread so chunk N+1 is being
  synthesized while chunk N is still audible.
- ``BargeInMonitor`` owns its microphone stream and thread instead of
  interleaving blocking reads with blocking writes inside the playback loop,
  which used to make detection lag behind real time.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

import numpy as np
import sounddevice as sd

import perf

# Kokoro always produces 24 kHz mono; keeping this fixed is what lets the
# output stream stay open forever.
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_BLOCKSIZE = 512

# Upper bound on how far synthesis may run ahead of playback. Work beyond this
# is wasted whenever the user barges in.
MAX_LOOKAHEAD_SEC = 3.0


def linear_resample(waveform: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resample, only used for the non-default Piper engine."""
    if src_rate == dst_rate or waveform.size == 0:
        return waveform
    duration = waveform.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=waveform.size, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False, dtype=np.float64)
    return np.interp(dst_x, src_x, waveform).astype(np.float32)


# --- TEXT CHUNKING ---------------------------------------------------------

_HARD_TERMINALS = ".!?\n"
_SOFT_TERMINALS = ",;:"
_CLOSERS = "\"')]»”’"

# A hard terminal always yields a chunk as long as there is something to say.
_HARD_MIN_CHARS = 2
# Soft terminals only split once the chunk is long enough to sound natural.
_SOFT_MIN_WORDS = 4
_SOFT_MIN_CHARS = 20


def _is_decimal_point(text: str, index: int) -> bool:
    """True for the dot in ``3.5`` or ``192.168.1.50`` (must not split there)."""
    if text[index] != ".":
        return False
    if index == 0 or index + 1 >= len(text):
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()


def _is_speakable(text: str, min_words: int, min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    # Punctuation-only fragments would make the TTS emit noise.
    if not any(ch.isalnum() for ch in stripped):
        return False
    return len(stripped.split()) >= min_words


def split_for_tts(buffer: str, allow_soft_split: bool) -> Tuple[List[str], str]:
    """Split buffered text into speakable chunks.

    ``allow_soft_split`` lets the first chunk of a turn break on a comma so
    audio starts sooner; later chunks wait for sentence-ending punctuation so
    prosody stays natural. Returns ``(chunks, remainder)``.
    """
    if not buffer:
        return [], ""

    chunks: List[str] = []
    soft_allowed = allow_soft_split
    start = 0
    i = 0
    length = len(buffer)

    while i < length:
        char = buffer[i]
        hard = char in _HARD_TERMINALS and not _is_decimal_point(buffer, i)
        soft = soft_allowed and char in _SOFT_TERMINALS
        if not hard and not soft:
            i += 1
            continue

        # Swallow closing quotes/brackets so they travel with the chunk.
        cut = i + 1
        while cut < length and buffer[cut] in _CLOSERS:
            cut += 1

        candidate = buffer[start:cut].strip()
        min_words = 1 if hard else _SOFT_MIN_WORDS
        min_chars = _HARD_MIN_CHARS if hard else _SOFT_MIN_CHARS
        if _is_speakable(candidate, min_words, min_chars):
            chunks.append(candidate)
            start = cut
            # Only the very first chunk of a turn is allowed to split softly.
            soft_allowed = False
        i = cut

    return chunks, buffer[start:]


# --- AUDIO OUTPUT ----------------------------------------------------------


class AudioSink:
    """A single always-open output stream fed from a lock-protected queue.

    The stream callback never blocks: it copies whatever is queued and pads the
    rest of the block with silence.

    Each queued block carries a ``notify`` flag. Only blocks that opt in fire
    ``on_first_audio``, so non-speech audio (the earcon, a filler clip) does not
    get counted as the start of the reply.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        on_first_audio: Optional[Callable[[], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._on_first_audio = on_first_audio
        self._lock = threading.Lock()
        self._blocks: Deque[Tuple[np.ndarray, bool]] = deque()
        self._pending_frames = 0
        self._emitted_audio = False
        self._drained = threading.Event()
        self._drained.set()
        self._stream: Optional[sd.OutputStream] = None
        self._failed = False

    # -- lifecycle --

    def start(self) -> bool:
        """Open the output stream. Returns False if no device is usable."""
        if self._stream is not None or self._failed:
            return self._stream is not None
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                latency="low",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            print(f"  [Audio]: no se pudo abrir la salida ({exc}); TTS mudo.")
            self._stream = None
            self._failed = True
            return False
        return True

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    # -- audio thread --

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        try:
            written = 0
            notified = False
            with self._lock:
                while written < frames and self._blocks:
                    block, notify = self._blocks[0]
                    take = min(frames - written, block.shape[0])
                    outdata[written : written + take, 0] = block[:take]
                    written += take
                    self._pending_frames -= take
                    notified = notified or notify
                    if take == block.shape[0]:
                        self._blocks.popleft()
                    else:
                        self._blocks[0] = (block[take:], notify)
                if not self._blocks:
                    self._pending_frames = 0
                    self._drained.set()
            if written < frames:
                outdata[written:, 0] = 0.0
            if notified and not self._emitted_audio:
                self._emitted_audio = True
                callback = self._on_first_audio
                if callback is not None:
                    callback()
        except Exception:  # noqa: BLE001
            # Never raise out of the audio thread: it would kill the stream.
            outdata[:] = 0.0

    # -- producer side --

    def reset_first_audio(self) -> None:
        """Re-arm the first-audio notification for a new turn."""
        self._emitted_audio = False

    def has_emitted_audio(self) -> bool:
        """True once a notifying block has actually reached the device."""
        return self._emitted_audio

    def enqueue(self, waveform: np.ndarray, notify: bool = True) -> None:
        """Queue audio. ``notify=False`` keeps it out of the first-audio metric."""
        if self._failed or waveform is None or waveform.size == 0:
            return
        block = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).reshape(-1))
        with self._lock:
            self._blocks.append((block, notify))
            self._pending_frames += block.shape[0]
            self._drained.clear()

    def pending_seconds(self) -> float:
        with self._lock:
            return self._pending_frames / float(self.sample_rate)

    def flush_with_fade(self, fade_ms: int) -> None:
        """Drop queued audio, ramping the next ``fade_ms`` down to silence.

        The ramp is built here rather than in the callback because it is a few
        thousand multiplications, far below one block period.
        """
        with self._lock:
            fade_frames = int(self.sample_rate * (max(0, fade_ms) / 1000.0))
            fade_frames = min(fade_frames, self._pending_frames)
            if fade_frames <= 1:
                self._blocks.clear()
                self._pending_frames = 0
                self._drained.set()
                return
            head, notify = self._take_frames_locked(fade_frames)
            head *= np.linspace(1.0, 0.0, num=head.shape[0], dtype=np.float32)
            self._blocks.clear()
            self._blocks.append((head, notify))
            self._pending_frames = head.shape[0]

    def _take_frames_locked(self, count: int) -> Tuple[np.ndarray, bool]:
        parts: List[np.ndarray] = []
        notify = False
        remaining = count
        while remaining > 0 and self._blocks:
            block, block_notify = self._blocks[0]
            take = min(remaining, block.shape[0])
            parts.append(block[:take])
            notify = notify or block_notify
            remaining -= take
            if take == block.shape[0]:
                self._blocks.popleft()
            else:
                self._blocks[0] = (block[take:], block_notify)
        head = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return head, notify

    def wait_drain(self, timeout: Optional[float] = None) -> bool:
        """Block until the queue is empty and the device buffer has played out."""
        if self._failed:
            return True
        drained = self._drained.wait(timeout=timeout)
        # The callback empties the queue slightly before the device is silent.
        time.sleep(3.0 * self.blocksize / float(self.sample_rate))
        return drained


# --- BARGE-IN --------------------------------------------------------------


class BargeInMonitor:
    """Watches the microphone on its own thread while Kimmi is speaking.

    The stream is opened once and kept open; chunks are discarded while
    disarmed so the driver buffer never accumulates. Speech detection is
    injected so this module stays free of VAD/PyAudio specifics.
    """

    def __init__(
        self,
        open_stream: Callable[[], Any],
        is_user_speech: Callable[[bytes, float, float], bool],
        reset_vad: Callable[[], None],
        chunk_frames: int,
        chunk_ms: float,
        grace_ms: float,
        min_speech_ms: float,
        capture_silence_sec: float,
        max_capture_sec: float,
        speech_threshold: float,
        speech_min_rms: float,
        capture_threshold: float,
        capture_min_rms: float,
    ) -> None:
        self._open_stream = open_stream
        self._is_user_speech = is_user_speech
        self._reset_vad = reset_vad
        self._chunk_frames = chunk_frames
        self._chunk_ms = chunk_ms
        self._grace_ms = grace_ms
        self._min_speech_ms = min_speech_ms
        self._capture_silence_sec = capture_silence_sec
        self._max_capture_sec = max_capture_sec
        self._speech_threshold = speech_threshold
        self._speech_min_rms = speech_min_rms
        self._capture_threshold = capture_threshold
        self._capture_min_rms = capture_min_rms

        self._stream: Any = None
        self._thread: Optional[threading.Thread] = None
        self._closing = threading.Event()
        self._armed = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._detected = threading.Event()
        self._capture_done = threading.Event()
        self._capture_done.set()
        self._on_detect: Optional[Callable[[], None]] = None
        self._armed_at = 0.0
        self._audio_lock = threading.Lock()
        self._audio: Optional[bytes] = None

    # -- lifecycle --

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="barge-in-monitor",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closing.set()
        self._armed.clear()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        self._drop_stream()

    # -- arming --

    def arm(self, on_detect: Callable[[], None]) -> None:
        if self._thread is None:
            return
        self._on_detect = on_detect
        self._detected.clear()
        self._capture_done.set()
        with self._audio_lock:
            self._audio = None
        self._reset_vad()
        self._armed_at = time.monotonic()
        self._idle.clear()
        self._armed.set()

    def disarm(self, timeout: float = 1.0) -> None:
        """Stop evaluating speech and wait for the loop to acknowledge."""
        self._armed.clear()
        if self._thread is None:
            return
        self._idle.wait(timeout=timeout)

    def detected(self) -> bool:
        return self._detected.is_set()

    def wait_capture(self, timeout: float = 6.0) -> None:
        self._capture_done.wait(timeout=timeout)

    def take_audio(self) -> Optional[bytes]:
        with self._audio_lock:
            data = self._audio
            self._audio = None
        return data or None

    # -- monitor thread --

    def _ensure_stream(self) -> Any:
        if self._stream is None:
            try:
                self._stream = self._open_stream()
            except Exception as exc:  # noqa: BLE001
                print(f"  [Barge-in]: no se pudo abrir el mic monitor ({exc})")
                self._stream = None
        return self._stream

    def _drop_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        speech_run_ms = 0.0
        seed: List[bytes] = []

        while not self._closing.is_set():
            stream = self._ensure_stream()
            if stream is None:
                time.sleep(0.25)
                continue

            try:
                data = stream.read(self._chunk_frames, exception_on_overflow=False)
            except Exception:  # noqa: BLE001
                self._drop_stream()
                time.sleep(0.05)
                continue

            if not self._armed.is_set():
                # Disarmed: drop the audio and never touch the VAD, so the
                # capture path on the main thread owns it exclusively.
                speech_run_ms = 0.0
                seed.clear()
                self._idle.set()
                continue

            if (time.monotonic() - self._armed_at) * 1000.0 < self._grace_ms:
                continue

            if self._is_user_speech(data, self._speech_threshold, self._speech_min_rms):
                seed.append(data)
                speech_run_ms += self._chunk_ms
                if speech_run_ms >= self._min_speech_ms:
                    self._trigger(stream, list(seed))
                    speech_run_ms = 0.0
                    seed.clear()
            else:
                speech_run_ms = 0.0
                seed.clear()

        self._drop_stream()

    def _trigger(self, stream: Any, seed: List[bytes]) -> None:
        self._detected.set()
        self._capture_done.clear()
        print("  [Barge-in]: te escuché; corto y te atiendo.")

        callback = self._on_detect
        if callback is not None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                print(f"  [Barge-in]: fallo al cortar el TTS ({exc})")

        try:
            captured = self._capture_utterance(stream, seed)
            with self._audio_lock:
                self._audio = captured
            print(f"  [Barge-in]: capturados {len(captured)} bytes de audio")
        except Exception as exc:  # noqa: BLE001
            print(f"  [Barge-in]: no se pudo capturar audio ({exc})")
        finally:
            # One detection per turn; the caller re-arms on the next turn.
            self._armed.clear()
            self._idle.set()
            self._capture_done.set()

    def _capture_utterance(self, stream: Any, seed: List[bytes]) -> bytes:
        """Keep recording until silence so the user need not repeat themselves."""
        frames: List[bytes] = list(seed)
        chunk_sec = self._chunk_ms / 1000.0
        max_silent = max(1, int(self._capture_silence_sec / chunk_sec))
        max_total = max(1, int(self._max_capture_sec / chunk_sec))
        silent = 0

        for _ in range(max_total):
            try:
                data = stream.read(self._chunk_frames, exception_on_overflow=False)
            except Exception:  # noqa: BLE001
                break
            frames.append(data)
            # Thresholds relax once the speaker is no longer competing.
            if self._is_user_speech(data, self._capture_threshold, self._capture_min_rms):
                silent = 0
            else:
                silent += 1
                if silent >= max_silent:
                    break

        return b"".join(frames)


# --- SPEECH PIPELINE -------------------------------------------------------

_TURN_END = object()


class SpeechPipeline:
    """Two-stage text-to-speech: chunking + synthesis thread + audio sink.

    ``synthesize(text, is_first_chunk)`` returns a float32 waveform already at
    the sink's sample rate, or None to skip the chunk.
    """

    def __init__(
        self,
        synthesize: Callable[[str, bool], Optional[np.ndarray]],
        sink: AudioSink,
        monitor: Optional[BargeInMonitor] = None,
        fade_out_ms: int = 180,
    ) -> None:
        self._synthesize = synthesize
        self._sink = sink
        self._monitor = monitor
        self._fade_out_ms = fade_out_ms

        self._text_q: "queue.Queue[Any]" = queue.Queue()
        self._turn_end_ack = threading.Event()
        self._abort = threading.Event()
        self._barge_in = threading.Event()
        self._closing = threading.Event()
        self._buffer = ""
        self._first_chunk = True
        self._spoken_seconds = 0.0
        self._chunk_index = 0

        self._thread = threading.Thread(
            target=self._synth_loop,
            name="tts-synth",
            daemon=True,
        )
        self._thread.start()

    # -- turn lifecycle --

    def begin_turn(self, allow_barge_in: bool = True) -> None:
        self._abort.clear()
        self._barge_in.clear()
        self._turn_end_ack.clear()
        self._buffer = ""
        self._first_chunk = True
        self._spoken_seconds = 0.0
        self._chunk_index = 0
        self._sink.reset_first_audio()
        if allow_barge_in and self._monitor is not None:
            self._monitor.arm(self._on_barge_in)

    def play_clip(self, waveform: Optional[np.ndarray], notify: bool = False) -> bool:
        """Queue an already-rendered clip. Returns False if the turn was cut.

        Deliberately does not touch ``_chunk_index``: the next synthesized chunk
        is still the first one, so ``synth1`` stays measurable and it keeps its
        fade-in, which is the attack of a new phrase rather than a dip mid-word.
        """
        if waveform is None or waveform.size == 0 or self._abort.is_set():
            return False
        self._spoken_seconds += waveform.size / float(self._sink.sample_rate)
        self._sink.enqueue(waveform, notify=notify)
        return True

    def feed(self, text: str) -> None:
        """Accumulate text and queue whatever is already speakable."""
        if not text or self._abort.is_set():
            return
        self._buffer += text
        chunks, self._buffer = split_for_tts(self._buffer, self._first_chunk)
        for chunk in chunks:
            self._text_q.put(chunk)
            self._first_chunk = False

    def end_turn(self) -> bool:
        """Flush, wait for playback to finish, and report whether it was cut."""
        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder and not self._abort.is_set():
            self._text_q.put(remainder)

        self._text_q.put(_TURN_END)
        self._turn_end_ack.wait(timeout=30.0)

        if not self._abort.is_set():
            self._sink.wait_drain(timeout=60.0)

        if self._monitor is not None:
            self._monitor.disarm()
            if self._barge_in.is_set():
                self._monitor.wait_capture()

        if self._spoken_seconds > 0:
            perf.value("spoken", f"{self._spoken_seconds:.1f}s")
        return self._barge_in.is_set()

    def barge_in_happened(self) -> bool:
        return self._barge_in.is_set()

    def close(self) -> None:
        self._closing.set()
        self._abort.set()
        self._text_q.put(_TURN_END)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    # -- barge-in --

    def _on_barge_in(self) -> None:
        """Called from the monitor thread the moment the user interrupts."""
        self._barge_in.set()
        self._abort.set()
        self._drain_text_queue()
        self._sink.flush_with_fade(self._fade_out_ms)

    def _drain_text_queue(self) -> None:
        """Drop pending chunks but keep any turn sentinel so end_turn returns."""
        pending_end = False
        while True:
            try:
                item = self._text_q.get_nowait()
            except queue.Empty:
                break
            if item is _TURN_END:
                pending_end = True
        if pending_end:
            self._text_q.put(_TURN_END)

    # -- synthesis thread --

    def _synth_loop(self) -> None:
        while not self._closing.is_set():
            item = self._text_q.get()
            if item is _TURN_END:
                self._turn_end_ack.set()
                continue
            if self._abort.is_set():
                continue

            first = self._chunk_index == 0
            self._chunk_index += 1
            try:
                waveform = self._synthesize(item, first)
            except Exception as exc:  # noqa: BLE001
                print(f"  [TTS]: fallo sintetizando ({exc})")
                continue

            if waveform is None or waveform.size == 0 or self._abort.is_set():
                continue

            self._await_playback_room()
            if self._abort.is_set():
                continue
            self._spoken_seconds += waveform.size / float(self._sink.sample_rate)
            self._sink.enqueue(waveform)

    def _await_playback_room(self) -> None:
        """Keep synthesis a bounded distance ahead of playback."""
        while not self._abort.is_set() and not self._closing.is_set():
            if self._sink.pending_seconds() < MAX_LOOKAHEAD_SEC:
                return
            time.sleep(0.05)

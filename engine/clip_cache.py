"""On-disk cache of pre-rendered TTS clips.

The fixed phrases Kimmi says (wake word acks, closings, short notices, fillers)
are a small closed set. Synthesizing them live costs a couple hundred
milliseconds every time for audio that never changes, which is most of the
perceived delay of a "Sí, Fede?".

Clips are stored as raw float32 waveforms at the sink's sample rate, so playing
one is a memcpy into the audio queue. A missing or unreadable clip simply
returns None and the caller falls back to live synthesis, so a cold or corrupt
cache can never break speech.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

MANIFEST_NAME = "manifest.json"


def make_signature(*parts: object) -> str:
    """Short hash of the voice configuration a cached clip was rendered with.

    Clips live under a directory named after this, so changing voice, speed,
    language or engine starts a fresh namespace instead of playing back audio
    that no longer matches how Kimmi sounds.
    """
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class ClipCache:
    """Text -> waveform, backed by RAM and then by ``.npy`` files on disk."""

    def __init__(
        self,
        root: str,
        synthesize: Callable[[str], Optional[np.ndarray]],
        signature: str,
    ) -> None:
        self.directory = os.path.join(root, signature)
        self._synthesize = synthesize
        self._memory: Dict[str, np.ndarray] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.npy")

    def get(self, text: str) -> Optional[np.ndarray]:
        """Return the cached waveform for ``text``, or None if it is not cached."""
        if not text or not text.strip():
            return None
        key = self._key(text)

        cached = self._memory.get(key)
        if cached is not None:
            return cached

        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            waveform = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [Clips]: no se pudo leer {key[:8]} ({exc})")
            return None

        waveform = np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
        self._memory[key] = waveform
        return waveform

    def prime(self, texts: Iterable[str]) -> Tuple[int, int]:
        """Make sure every phrase is cached. Returns (reused, synthesized)."""
        wanted: Dict[str, str] = {}
        for text in texts:
            cleaned = (text or "").strip()
            if cleaned:
                wanted[self._key(cleaned)] = cleaned

        os.makedirs(self.directory, exist_ok=True)
        reused = 0
        synthesized = 0

        for key, text in wanted.items():
            if self.get(text) is not None:
                reused += 1
                continue
            try:
                waveform = self._synthesize(text)
            except Exception as exc:  # noqa: BLE001
                print(f"  [Clips]: fallo sintetizando {text!r} ({exc})")
                continue
            if waveform is None or waveform.size == 0:
                continue
            waveform = np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
            self._memory[key] = waveform
            try:
                np.save(self._path(key), waveform, allow_pickle=False)
                synthesized += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [Clips]: no se pudo guardar {text!r} ({exc})")

        self._write_manifest(wanted)
        self._prune(set(wanted))
        return reused, synthesized

    def _write_manifest(self, wanted: Dict[str, str]) -> None:
        """Human-readable index, only so the cache dir can be inspected."""
        try:
            with open(
                os.path.join(self.directory, MANIFEST_NAME), "w", encoding="utf-8"
            ) as handle:
                json.dump(wanted, handle, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [Clips]: no se pudo escribir el manifest ({exc})")

    def _prune(self, keep: set) -> None:
        """Drop clips for phrases that no longer exist in the code."""
        try:
            entries: List[str] = os.listdir(self.directory)
        except OSError:
            return
        for entry in entries:
            if not entry.endswith(".npy"):
                continue
            if entry[: -len(".npy")] in keep:
                continue
            try:
                os.remove(os.path.join(self.directory, entry))
            except OSError:
                pass

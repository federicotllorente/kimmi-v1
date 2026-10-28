"""Eval of speech-to-text: accuracy (WER) and latency per Whisper model size.

Run from engine/:
    proj-env\\Scripts\\python.exe tests\\eval_stt.py
    proj-env\\Scripts\\python.exe tests\\eval_stt.py --sizes medium small base

Audio comes from two sources:
  * Synthetic: the phrases below, spoken by Kokoro (Kimmi's own TTS voices).
    Clean studio-like audio, so real-world WER will be higher.
  * Your recordings (recommended): put pairs like tests/audio/es_taza.wav +
    tests/audio/es_taza.txt (the exact words) in tests/audio/. 16 kHz mono WAV.
    The language is taken from the file name prefix: es_ or en_.

Settings match Kimmi's transcribe_audio (fixed language, beam 1, temperature
0) except the initial_prompt, which lives in assistant.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import time
import unicodedata
import wave
from typing import List, Tuple

import numpy as np

import _common  # noqa: F401  (sets sys.path)
from _common import ENGINE_DIR

from locale_config import LOCALES

SAMPLE_RATE = 16000

PHRASES: List[Tuple[str, str]] = [
    ("es", "Hola Kimmi, ¿me escuchas?"),
    ("es", "¿Qué ves en la mesa?"),
    ("es", "¿Dónde está el lápiz rojo?"),
    ("es", "Abre Visual Studio Code en el proyecto del brazo robótico"),
    ("es", "¿Qué tiempo hace mañana en Sevilla?"),
    ("es", "Recuerda que el servo del codo va conectado al pin catorce"),
    ("es", "Reinicia el contenedor de mosquitto"),
    ("es", "¿Cuánto es el quince por ciento de doscientos treinta?"),
    ("es", "Busca las tijeras y dime si están a la izquierda"),
    ("es", "Investiga qué microcontrolador es mejor para controlar servos"),
    ("en", "Hey Kimmi, can you hear me?"),
    ("en", "What do you see on the table?"),
    ("en", "Where is the blue cup?"),
    ("en", "Open the browser and search for ESP32 pinout"),
    ("en", "Remember that the camera is mounted above the arm"),
    ("en", "Restart the postgres container please"),
]


def normalize(text: str) -> List[str]:
    """Lowercase, no accents, no punctuation: compare words, not formatting."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9ñ ]+", " ", text).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance / number of reference words."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    dist = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        prev, dist[0] = dist[0], i
        for j, hyp_word in enumerate(hyp, start=1):
            prev, dist[j] = dist[j], min(
                dist[j] + 1,  # deletion
                dist[j - 1] + 1,  # insertion
                prev + (ref_word != hyp_word),  # substitution
            )
    return dist[len(hyp)] / max(len(ref), 1)


def synthesize(phrases: List[Tuple[str, str]]) -> List[Tuple[str, str, np.ndarray]]:
    from kokoro_onnx import Kokoro

    # CPU on purpose: keeps the GPU free for Whisper so the VRAM numbers are fair
    kokoro = Kokoro(
        os.path.join(ENGINE_DIR, "kokoro", "kokoro-v1_0.onnx"),
        os.path.join(ENGINE_DIR, "kokoro", "voices-v1_0.bin"),
    )
    clips = []
    for lang, text in phrases:
        loc = LOCALES[lang]
        samples, rate = kokoro.create(text, voice=loc.tts_voice, speed=1.0, lang=loc.tts_lang)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        # Kokoro speaks at 24 kHz; Whisper expects 16 kHz
        positions = np.linspace(0, len(samples) - 1, int(len(samples) * SAMPLE_RATE / rate))
        clips.append((lang, text, np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)))
    return clips


def load_recordings() -> List[Tuple[str, str, np.ndarray]]:
    clips = []
    for wav_path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "audio", "*.wav"))):
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            continue
        lang = os.path.basename(wav_path).split("_", 1)[0]
        with wave.open(wav_path, "rb") as wf:
            if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
                print(f"  skip {wav_path}: needs 16 kHz mono")
                continue
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        with open(txt_path, encoding="utf-8") as f:
            clips.append((lang, f.read().strip(), pcm.astype(np.float32) / 32768.0))
    return clips


def gpu_memory_used_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def evaluate(size: str, clips) -> Tuple[float, float, int]:
    from faster_whisper import WhisperModel

    before = gpu_memory_used_mb()
    model = WhisperModel(size, device="cuda", compute_type="float16")
    # Warm-up: the first call pays CUDA initialization
    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")[0])
    vram = gpu_memory_used_mb() - before

    print(f"\n== whisper {size}  (+{vram} MB VRAM)")
    total_wer, total_time = 0.0, 0.0
    for lang, reference, audio in clips:
        start = time.time()
        segments, _info = model.transcribe(
            audio,
            language=LOCALES[lang].whisper_language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        hypothesis = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.time() - start
        wer = word_error_rate(reference, hypothesis)
        total_wer += wer
        total_time += elapsed
        mark = "OK " if wer == 0 else "ERR"
        print(f"  {mark} {elapsed * 1000:5.0f} ms  WER {wer:4.0%}  {hypothesis!r}")
        if wer:
            print(f"                          expected {reference!r}")

    del model
    return total_wer / len(clips), total_time / len(clips), vram


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["medium", "small"])
    args = parser.parse_args()

    recordings = load_recordings()
    print(f"Synthesizing {len(PHRASES)} phrases with Kokoro; {len(recordings)} recordings found.")
    clips = synthesize(PHRASES) + recordings

    results = [(size, *evaluate(size, clips)) for size in args.sizes]

    print("\n== Summary (lower is better)")
    for size, wer, latency, vram in results:
        print(f"  {size:<8} WER {wer:5.1%}   avg {latency * 1000:5.0f} ms   VRAM +{vram} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

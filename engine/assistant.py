import os
import io
import json
import random
import re
import subprocess
import sys
import time
import wave
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyaudio
import torch
import ollama
from openwakeword.model import Model
from faster_whisper import WhisperModel

import perf
from audio_out import (
    DEFAULT_SAMPLE_RATE,
    AudioSink,
    BargeInMonitor,
    SpeechPipeline,
    linear_resample,
)
from clip_cache import ClipCache, make_signature
from locale_config import Locale, current_locale
from tools.app_allowlist import list_allowed_app_ids
from tools.intent import (
    INTENT_CONFIDENCE_THRESHOLD,
    classify_intent,
    confirmation_prompt,
    parse_yes_no,
    previous_user_fact,
    should_force_memory_save,
    tool_needs_confirmation,
)
from tools.registry import build_system_prompt, build_tools, format_memory_block, tools_for_intent
from tools.memory import memory_db

# The reply is streamed to stdout token by token. On a Windows console that is
# cp1252 by default, a single curly quote or dash from the model raises
# UnicodeEncodeError and kills the turn. The bridges already set
# PYTHONIOENCODING, so this only covers running the engine straight from a shell.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

# --- CONFIGURACIÓN DE PARÁMETROS ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# Silero VAD requiere tamaños de chunk específicos para 16kHz (512, 1024 o 1536 muestras)
VAD_CHUNK = 512  # ~32 ms por chunk
OWW_CHUNK = 1280  # ~80 ms por chunk para openWakeWord

WAKE_WORD_THRESHOLD = 0.5
# Tras TTS / fin de turno, ignorar wake word (eco del altavoz en el mic).
# Solo aplica al detector de wake word, no a la escucha dentro de una sesión.
WAKE_COOLDOWN_SEC = 1.25
# Pausa corta antes de volver a grabar en una sesión activa (eco residual).
# El barge-in ya cubre el caso de hablar encima, así que no hace falta más.
MIC_SETTLE_SEC = 0.25
# Pre-roll mientras esperamos wake word (comando pegado: "Hey Kimmi, …").
WAKE_RING_BUFFER_SEC = 2.0
# Ventana post-wake para decidir si el usuario sigue hablando (sin acuse TTS).
WAKE_FOLLOW_SPEECH_SEC = 0.5
# Últimos N segundos del ring a inspeccionar con VAD.
WAKE_TAIL_SPEECH_SEC = 0.7
# Tras responder, seguir escuchando sin wake word hasta cierre o silencio.
CONVERSATION_MODE = True
# Segundos de silencio (sin voz) para cerrar la sesión en turnos de seguimiento.
CONVERSATION_FOLLOWUP_TIMEOUT_SEC = 8.0
SILENCE_LIMIT_SEC = 1.0  # Un poco más de margen para no cortar a mitad de frase
MAX_RECORD_SEC = 15.0  # Tiempo máximo de grabación de seguridad
# Conservar un poco de audio antes de detectar voz (evita cortar la primera sílaba)
VAD_PRE_ROLL_SEC = 0.3

HISTORY_MAX_TURNS = 6  # pares user/assistant ≈ 12 mensajes
MIN_TRANSCRIPT_CHARS = 3
MAX_NO_SPEECH_PROB = 0.6
# Rechazar segmentos muy inciertos (avg_logprob más negativo = peor)
MIN_AVG_LOGPROB = -1.0
TOOL_RESULT_MAX_CHARS = 500

# Sandbox: el asistente solo puede leer/escribir archivos dentro de este directorio
ALLOWED_DATA_ROOT = os.path.realpath("./")

# Cursor SDK (local one-shot). Requiere: pip install cursor-sdk y CURSOR_API_KEY en el entorno.
CURSOR_WORKSPACE = os.path.realpath("./")
CURSOR_MODEL = "composer-2.5"

PIPER_EXE = "proj-env/Scripts/piper"

# TTS: "kokoro" o "piper" — cambia esta variable para probar uno u otro.
# La voz, la velocidad y el idioma salen del locale activo.
TTS_ENGINE = "kokoro"
KOKORO_MODEL = "kokoro/kokoro-v1_0.onnx"
KOKORO_VOICES = "kokoro/voices-v1_0.bin"
# Fade-in solo en el primer chunk del turno. El stream de salida nunca se
# cierra, así que ya no hacen falta ni lead-in ni warmup del dispositivo.
KOKORO_FADE_IN_MS = 50

# Earcon: tono corto en cuanto el VAD corta la frase, para confirmar "te
# escuché" a los ~0 ms. No pasa por TTS, así que no tiene idioma ni coste.
EARCON_ENABLED = False
EARCON_FREQ_HZ = 880.0
EARCON_MS = 90
EARCON_GAIN = 0.12

# Barge-in: escuchar al usuario mientras Kimmi habla.
# Si interrumpe: corta el TTS con fade-out corto y captura su frase.
BARGE_IN_ENABLED = True
BARGE_IN_GRACE_MS = 400  # no detectar al inicio (eco / arranque del altavoz)
BARGE_IN_MIN_SPEECH_MS = 400  # voz sostenida para confirmar interrupción
BARGE_IN_VAD_THRESHOLD = 0.72  # más alto que el VAD de captura (eco del TTS)
BARGE_IN_MIN_RMS = 0.025
BARGE_IN_FADE_OUT_MS = 180  # corta el audio con fade, no de golpe
BARGE_IN_CAPTURE_SILENCE_SEC = 0.85  # fin de la frase del usuario tras cortar TTS

# Nombre del usuario para el acuse del wake word.
USER_NAME = "Fede"

# Feature flags
ENABLE_INTENT_ROUTER = False  # Clasificar intent antes de responder (añade latencia)
ENABLE_TOOLS = False  # Function calling / acciones locales

# Relleno hablado cuando el turno se va de tiempo. El umbral se mide desde el
# corte del VAD, no desde el LLM: hoy un turno normal empieza a sonar entre 830
# y 1100 ms, así que a 1200 ms el relleno solo aparece cuando algo va lento.
# Las frases están en el locale.
FILLER_ENABLED = True
FILLER_AFTER_MS = 1200

# Modelo ligero: clasificar intención + charla sin tools.
FAST_MODEL = "qwen2.5:3b"
# Modelo pesado: solo function calling y resumen tras tools.
LLM_MODEL = "qwen2.5:14b"

# num_ctx explícito evita que Ollama redecida el contexto en cada request.
# num_predict acota la respuesta: es voz, no un ensayo.
LLM_CHAT_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 96,
    "temperature": 0.6,
    "top_k": 20,
}

# STT: medium es más preciso que small (cuesta algo más de latencia en GPU).
# WHISPER_MODEL_SIZE = "tiny"
# WHISPER_MODEL_SIZE = "small"
WHISPER_MODEL_SIZE = "medium"
# beam 1: con temperature 0.0 el beam search apenas cambia el resultado y
# cuesta bastante latencia. best_of solo aplica con muestreo, así que va a 1.
WHISPER_BEAM_SIZE = 1

_stt_prompt_cache: Dict[str, str] = {}


def stt_initial_prompt(loc: Locale) -> str:
    """Bias prompt for Whisper, with the user name and app list filled in.

    Cached per language: the app list is a join over the allowlist and this runs
    on every transcription.
    """
    cached = _stt_prompt_cache.get(loc.code)
    if cached is None:
        cached = loc.stt_initial_prompt.format(
            name=USER_NAME,
            apps=", ".join(list_allowed_app_ids()),
        )
        _stt_prompt_cache[loc.code] = cached
    return cached


# --- TTS (carga solo el motor elegido) ---
TTS_ENGINE = TTS_ENGINE.strip().lower()
if TTS_ENGINE not in ("kokoro", "piper"):
    raise ValueError(f"Invalid TTS_ENGINE: {TTS_ENGINE!r}. Use 'kokoro' or 'piper'.")

def _load_kokoro():
    """Load Kokoro preferring CUDA.

    kokoro-onnx hardcodes CPUExecutionProvider and its GPU branch checks
    find_spec("onnxruntime-gpu"), which can never match because the module is
    named onnxruntime. Building the session here is the only way to get CUDA
    while keeping CPU as a fallback inside the same provider list.
    torch is imported above, so its CUDA/cuDNN DLLs are already in the process.
    """
    from kokoro_onnx import Kokoro

    print(f"Loading Kokoro TTS ({current_locale().tts_voice})...")
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # Errors only: the CUDA EP is chatty at load and the bridges parse stdout.
        options.log_severity_level = 3
        session = ort.InferenceSession(
            KOKORO_MODEL,
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        engine = Kokoro.from_session(session, KOKORO_VOICES)
        providers = session.get_providers()
        print(f"  Kokoro providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            print("  Kokoro running on CPU: synthesis will be ~4x slower.")
        return engine
    except Exception as exc:  # noqa: BLE001
        print(f"  Kokoro GPU unavailable ({exc}); falling back to CPU.")
        return Kokoro(KOKORO_MODEL, KOKORO_VOICES)


kokoro = None
if TTS_ENGINE == "kokoro":
    kokoro = _load_kokoro()
else:
    print(f"Using Piper TTS ({current_locale().piper_model})...")

# --- TOOLS (FUNCTION CALLING) ---
tools_schema, available_tools = build_tools(
    allowed_root=ALLOWED_DATA_ROOT,
    llm_model=LLM_MODEL,
    cursor_workspace=CURSOR_WORKSPACE,
    cursor_model=CURSOR_MODEL,
    cursor_api_key=os.environ.get("CURSOR_API_KEY"),
)

# --- INICIALIZACIÓN DE COMPONENTES ---
print("Loading wake word model...")
# English phonetics, so these work in both languages; nothing to retrain.
oww_model = Model(
    wakeword_models=[
        "oww/kee_me.onnx",       # "Kimmi"
        "oww/ghee_me.onnx",       # "Kimmi" (closer to the Spanish pronunciation)
        "oww/hey_kee_me.onnx",   # "Hey Kimmi"
        "oww/okay_kee_me.onnx",  # "OK Kimmi"
        "oww/eigh_ghee_me.onnx", # "Ey Kimmi" (closer to the Spanish pronunciation)
    ],
    inference_framework="onnx",
)

print("Loading Silero VAD...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
# Second instance for the barge-in thread. Silero is stateful, so sharing one
# model between the capture loop and the monitor thread would corrupt it.
barge_vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)

print("Loading Faster-Whisper (STT)...")
print(f"  Whisper model: {WHISPER_MODEL_SIZE} (beam={WHISPER_BEAM_SIZE})")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cuda",
    compute_type="float16",
)

audio = pyaudio.PyAudio()

# Timestamp monotonic hasta el cual se ignoran wake words (anti-eco TTS).
_wake_ignore_until = 0.0

_system_prompt_cache: Dict[str, str] = {}


def system_prompt(loc: Locale) -> str:
    """System prompt for the active language.

    Cached per language so the LLM prefix stays byte-identical between turns and
    Ollama can reuse its KV cache instead of re-prefilling the whole prompt.
    """
    cached = _system_prompt_cache.get(loc.code)
    if cached is None:
        cached = build_system_prompt(ALLOWED_DATA_ROOT, loc.language_name)
        _system_prompt_cache[loc.code] = cached
    return cached


def suppress_wake_word(seconds: float = WAKE_COOLDOWN_SEC) -> None:
    """Bloquea detección de wake word durante N segundos."""
    global _wake_ignore_until
    _wake_ignore_until = max(_wake_ignore_until, time.monotonic() + seconds)


def wake_word_suppressed() -> bool:
    return time.monotonic() < _wake_ignore_until


def flush_mic_stream(stream, max_chunks: int = 25) -> None:
    """Vacía audio encolado en el mic (eco residual del TTS)."""
    for _ in range(max_chunks):
        try:
            available = stream.get_read_available()
        except Exception:
            break
        if available < OWW_CHUNK:
            break
        stream.read(OWW_CHUNK, exception_on_overflow=False)


def reset_wake_word_model() -> None:
    """Limpia buffers internos de openWakeWord tras un turno/TTS."""
    try:
        buf = getattr(oww_model, "prediction_buffer", None)
        if isinstance(buf, dict):
            for key in list(buf.keys()):
                buf[key] = []
    except Exception:
        pass


def _iter_vad_chunks(audio_bytes: bytes):
    """Parte PCM int16 mono en chunks del tamaño que espera Silero VAD."""
    step = VAD_CHUNK * 2  # int16 → 2 bytes/muestra
    for i in range(0, len(audio_bytes) - step + 1, step):
        yield audio_bytes[i : i + step]


def _vad_has_speech(audio_bytes: bytes, threshold: float = 0.5) -> bool:
    """True si algún chunk del audio supera el umbral VAD."""
    if not audio_bytes or len(audio_bytes) < VAD_CHUNK * 2:
        return False
    vad_model.reset_states()
    for chunk in _iter_vad_chunks(audio_bytes):
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        speech_prob = float(vad_model(torch.from_numpy(audio_float32), RATE).item())
        if speech_prob > threshold:
            return True
    return False


def _audio_tail(audio_bytes: bytes, seconds: float) -> bytes:
    """Últimos N segundos de PCM int16 mono."""
    max_bytes = int(seconds * RATE) * 2
    if max_bytes <= 0 or len(audio_bytes) <= max_bytes:
        return audio_bytes
    # Alinear a muestra (2 bytes).
    max_bytes -= max_bytes % 2
    return audio_bytes[-max_bytes:]


def speech_continues_after_wake(
    seed_bytes: bytes,
    stream,
) -> Tuple[bool, bytes]:
    """Decide si hay comando pegado tras la wake word.

    1) VAD sobre los últimos WAKE_TAIL_SPEECH_SEC del ring.
    2) Siempre lee WAKE_FOLLOW_SPEECH_SEC del mic (bytes acumulados).
    3) continues si hay habla en el peek (fiable) o, si el peek es silencio,
       no usamos el tail solo (la wake word vive ahí y daría falsos positivos).

    Devuelve (continues, peeked_bytes).
    """
    tail = _audio_tail(seed_bytes, WAKE_TAIL_SPEECH_SEC)
    tail_speech = _vad_has_speech(tail, threshold=0.5)

    peek_chunks = max(1, int(WAKE_FOLLOW_SPEECH_SEC / (VAD_CHUNK / RATE)))
    peeked_parts: List[bytes] = []
    vad_model.reset_states()
    peek_speech = False
    for _ in range(peek_chunks):
        data = stream.read(VAD_CHUNK, exception_on_overflow=False)
        peeked_parts.append(data)
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        speech_prob = float(vad_model(torch.from_numpy(audio_float32), RATE).item())
        if speech_prob > 0.5:
            peek_speech = True

    peeked = b"".join(peeked_parts)
    # Peek decide el acuse. El tail se registra para diagnóstico.
    if peek_speech:
        continues = True
    elif tail_speech:
        # Habla en el buffer pero silencio justo después del trigger → solo wake.
        print("  [Wake]: speech in buffer, silence after trigger -> ack")
        continues = False
    else:
        continues = False
    return continues, peeked


def strip_wake_phrase(text: str) -> str:
    """Quita 'Hey Kimmi' (y variantes) del inicio del transcript."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    stripped = current_locale().wake_phrase_re.sub("", cleaned, count=1).strip()
    stripped = re.sub(r"^[,.\s]+", "", stripped).strip()
    return stripped


def _apply_seed_audio_to_recording(
    seed_audio: bytes,
    frames: List[bytes],
    pre_roll: deque,
) -> Tuple[bool, int]:
    """Incorpora audio previo al estado de grabación VAD.

    Devuelve (has_spoken, silent_chunks al final del seed).
    """
    has_spoken = False
    silent_chunks = 0
    if not seed_audio:
        return has_spoken, silent_chunks

    vad_model.reset_states()
    for data in _iter_vad_chunks(seed_audio):
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        speech_prob = float(vad_model(torch.from_numpy(audio_float32), RATE).item())
        if speech_prob > 0.5:
            if not has_spoken:
                has_spoken = True
                frames.extend(pre_roll)
                pre_roll.clear()
            frames.append(data)
            silent_chunks = 0
        else:
            if has_spoken:
                frames.append(data)
                silent_chunks += 1
            else:
                pre_roll.append(data)
    return has_spoken, silent_chunks


# --- REPRODUCCIÓN Y SÍNTESIS DE AUDIO ---
def _apply_fade_in(waveform: np.ndarray, sample_rate: int, fade_ms: int) -> np.ndarray:
    """Fade-in lineal al inicio del primer chunk del turno."""
    if fade_ms <= 0 or waveform.size == 0:
        return waveform
    n = min(len(waveform), int(sample_rate * (fade_ms / 1000.0)))
    if n <= 1:
        return waveform
    out = waveform.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def _mic_chunk_is_user_speech(
    audio_bytes: bytes,
    vad_threshold: float = BARGE_IN_VAD_THRESHOLD,
    min_rms: float = BARGE_IN_MIN_RMS,
) -> bool:
    """Heurística anti-eco: VAD alto + energía suficiente.

    Runs on the barge-in thread, so it uses its own Silero instance.
    """
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(audio_float32)))) if audio_float32.size else 0.0
    if rms < min_rms:
        return False
    tensor_chunk = torch.from_numpy(audio_float32)
    try:
        speech_prob = float(barge_vad_model(tensor_chunk, RATE).item())
    except Exception:
        return False
    return speech_prob >= vad_threshold


def _open_barge_in_stream():
    """Dedicated input stream for the barge-in monitor thread."""
    return audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=VAD_CHUNK,
    )


audio_sink = AudioSink(
    sample_rate=DEFAULT_SAMPLE_RATE,
    on_first_audio=lambda: perf.mark("ttfa"),
)


def _build_earcon(sample_rate: int) -> np.ndarray:
    """Sine under a raised-cosine envelope, so it fades in and out clickless."""
    n = int(sample_rate * (EARCON_MS / 1000.0))
    if n <= 1:
        return np.zeros(0, dtype=np.float32)
    ramp = np.arange(n, dtype=np.float32)
    tone = np.sin(2.0 * np.pi * EARCON_FREQ_HZ * ramp / sample_rate)
    envelope = 0.5 - 0.5 * np.cos(2.0 * np.pi * ramp / (n - 1))
    return (tone * envelope * EARCON_GAIN).astype(np.float32)


EARCON = _build_earcon(DEFAULT_SAMPLE_RATE)


def play_earcon() -> None:
    """Acuse instantáneo al cortar el VAD, antes incluso de transcribir.

    notify=False: no es la respuesta, así que no debe contar como ttfa.
    """
    if not EARCON_ENABLED or EARCON.size == 0:
        return
    audio_sink.enqueue(EARCON, notify=False)
    perf.mark("earcon")

barge_in_monitor = BargeInMonitor(
    open_stream=_open_barge_in_stream,
    is_user_speech=_mic_chunk_is_user_speech,
    reset_vad=barge_vad_model.reset_states,
    chunk_frames=VAD_CHUNK,
    chunk_ms=1000.0 * VAD_CHUNK / RATE,
    grace_ms=BARGE_IN_GRACE_MS,
    min_speech_ms=BARGE_IN_MIN_SPEECH_MS,
    capture_silence_sec=BARGE_IN_CAPTURE_SILENCE_SEC,
    max_capture_sec=MAX_RECORD_SEC,
    speech_threshold=BARGE_IN_VAD_THRESHOLD,
    speech_min_rms=BARGE_IN_MIN_RMS,
    # Once the TTS is cut the speaker no longer competes, so relax a bit.
    capture_threshold=0.55,
    capture_min_rms=0.015,
)


def take_barge_in_audio() -> Optional[bytes]:
    """Consume el audio capturado en el último barge-in (o None)."""
    return barge_in_monitor.take_audio()


def _synthesize_chunk(text: str, is_first: bool) -> Optional[np.ndarray]:
    """Synthesize one chunk at the sink's sample rate. Runs on the TTS thread."""
    if not text or not text.strip():
        return None

    loc = current_locale()
    started = time.monotonic()
    if TTS_ENGINE == "piper":
        waveform, sample_rate = _synthesize_piper(text, loc)
    else:
        samples, sample_rate = kokoro.create(
            text,
            voice=loc.tts_voice,
            speed=loc.tts_speed,
            lang=loc.tts_lang,
        )
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)

    if waveform is None or waveform.size == 0:
        return None
    if sample_rate != audio_sink.sample_rate:
        waveform = linear_resample(waveform, sample_rate, audio_sink.sample_rate)

    if is_first:
        waveform = _apply_fade_in(waveform, audio_sink.sample_rate, KOKORO_FADE_IN_MS)
        perf.duration("synth1", (time.monotonic() - started) * 1000.0)
    return waveform


def _synthesize_piper(text: str, loc: Locale) -> Tuple[Optional[np.ndarray], int]:
    """Piper fallback engine; kept working but no longer the default path."""
    if loc.piper_model is None:
        print(f"  [TTS Piper]: no Piper voice for {loc.code!r}; use TTS_ENGINE='kokoro'")
        return None, audio_sink.sample_rate

    wav_output = "temp_sentence.wav"
    cmd_piper = [PIPER_EXE, "--model", loc.piper_model, "--output_file", wav_output]
    process = subprocess.Popen(
        cmd_piper,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    process.communicate(input=text)
    if process.returncode != 0 or not os.path.exists(wav_output):
        print(f"  [TTS Piper]: synthesis failed (code={process.returncode})")
        return None, audio_sink.sample_rate

    with wave.open(wav_output, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

    if sampwidth != 2:
        print(f"  [TTS Piper]: unsupported format (sampwidth={sampwidth})")
        return None, framerate

    waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return waveform, framerate


speech = SpeechPipeline(
    synthesize=_synthesize_chunk,
    sink=audio_sink,
    monitor=barge_in_monitor,
    fade_out_ms=BARGE_IN_FADE_OUT_MS,
)

CLIP_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".kimmi_cache", "tts"
)
_clip_caches: Dict[str, ClipCache] = {}


def clip_cache_for(loc: Locale) -> ClipCache:
    """Clip cache for one language, memoized.

    The signature covers voice, speed and language, so each locale lands in its
    own directory and switching back and forth never resynthesizes anything.
    """
    cache = _clip_caches.get(loc.code)
    if cache is None:
        cache = ClipCache(
            root=CLIP_CACHE_ROOT,
            synthesize=lambda text: _synthesize_chunk(text, True),
            signature=make_signature(
                TTS_ENGINE,
                loc.tts_voice if TTS_ENGINE == "kokoro" else loc.piper_model,
                loc.tts_speed,
                loc.tts_lang,
                audio_sink.sample_rate,
            ),
        )
        _clip_caches[loc.code] = cache
    return cache


def fixed_phrases(loc: Locale) -> List[str]:
    """Todo lo que Kimmi dice con texto fijo, o sea, lo cacheable."""
    phrases = [t.format(name=USER_NAME) for t in loc.wake_ack]
    phrases += list(loc.closing_ack)
    phrases += list(loc.filler)
    phrases.append(loc.filler_tools)
    # Skip templates with placeholders: they are never spoken as-is.
    phrases += [t for t in loc.spoken.values() if "{" not in t]
    return phrases


# Un solo relleno por turno: si el camino de tools ya avisó, el chat no repite.
_filler_played = threading.Event()


def _enqueue_filler(phrase: str) -> bool:
    """Encola un relleno si de verdad hace falta. No imprime nada.

    Silencio deliberado: desktop/bridge.py ingiere caracteres tras [Kimmi]:
    hasta el primer salto de línea, y stream_ollama_to_tts ya dejó esa línea
    abierta. Un print desde aquí congelaría la onda el resto de la respuesta.
    """
    if not FILLER_ENABLED or _filler_played.is_set():
        return False
    # Si ya suena o hay algo encolado, el relleno caería por detrás de la voz.
    if audio_sink.has_emitted_audio() or audio_sink.pending_seconds() > 0:
        return False
    clip = clip_cache_for(current_locale()).get(phrase)
    if clip is None:
        return False
    _filler_played.set()
    audio_sink.enqueue(clip, notify=False)
    perf.mark("filler")
    return True


def arm_filler() -> Optional[threading.Timer]:
    """Programa el relleno para que suene solo si el turno tarda de más."""
    if not FILLER_ENABLED or _filler_played.is_set():
        return None
    elapsed = perf.elapsed_ms() or 0
    delay = max(0.0, (FILLER_AFTER_MS - elapsed) / 1000.0)
    timer = threading.Timer(
        delay,
        _enqueue_filler,
        args=(random.choice(current_locale().filler),),
    )
    timer.daemon = True
    timer.start()
    return timer


def speak_text(text: str, allow_barge_in: bool = True) -> bool:
    """Reproduce texto por TTS (bloqueante hasta que termina el audio).

    Devuelve True si hubo barge-in. El audio del usuario queda en
    take_barge_in_audio() para no pedirle que repita.
    """
    if not text or not text.strip():
        return False

    print(f"[Kimmi]: {text}")
    speech.begin_turn(allow_barge_in=allow_barge_in and BARGE_IN_ENABLED)
    # Las frases fijas salen del caché: es un memcpy en vez de ~250 ms de Kokoro.
    clip = clip_cache_for(current_locale()).get(text)
    if clip is not None:
        speech.play_clip(clip, notify=True)
    else:
        speech.feed(text.strip())
    barged = speech.end_turn()
    # Tras hablar, el mic capta el eco → no aceptar wake word un momento.
    suppress_wake_word(WAKE_COOLDOWN_SEC)
    return barged


def speak_wake_ack():
    """Acuse inmediato tras la wake word; frase aleatoria."""
    template = random.choice(current_locale().wake_ack)
    ack = template.format(name=USER_NAME)
    # Sin barge-in en el acuse corto (muy breve y fácil de confundir con eco).
    speak_text(ack, allow_barge_in=False)


def is_conversation_closing(text: str) -> bool:
    """True si el usuario está cerrando el turno (gracias, listo, etc.).

    Si es True, NO se llama al LLM: solo un acuse breve, sin preguntas nuevas.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    loc = current_locale()

    # Quitar signos de apertura/puntuación final para matching más robusto.
    normalized = cleaned.lower().strip()
    normalized = re.sub(r"^[¿¡\s]+", "", normalized)
    normalized = re.sub(r"[\s\.\!\?\,;:]+$", "", normalized)

    if loc.closing_re.match(cleaned) or loc.closing_re.match(normalized):
        return True

    # Frases cortas que terminan en "gracias" o contienen despedida explícita.
    # Límite de longitud: evita cerrar pedidos largos que solo mencionan "gracias".
    if len(normalized) <= 90 and loc.closing_soft_re.search(normalized):
        return True

    return any(
        normalized == s
        or normalized.startswith(s + " ")
        or normalized.startswith(s + ",")
        or normalized.startswith(s + ".")
        for s in loc.closing_starters
    )


def speak_closing_ack() -> str:
    """Cierre breve; no pregunta si necesita algo más."""
    ack = random.choice(current_locale().closing_ack)
    speak_text(ack, allow_barge_in=False)
    return ack


# --- CAPTURA Y TRANCRIPCIÓN ---
def record_user_audio_vad(
    stream,
    prompt: str = "",
    max_wait_for_speech_sec: Optional[float] = None,
    seed_audio: bytes = b"",
) -> bytes:
    """Graba audio dinámicamente cortando al detectar silencio post-habla.

    Descarta el silencio inicial y guarda un pre-roll corto para no cortar
    el inicio de la primera palabra.

    Si seed_audio está presente, se trata como audio ya capturado (p.ej. ring
    buffer de la wake word) y se continúa hasta silencio.

    Si max_wait_for_speech_sec está definido y no hay voz en ese tiempo,
    devuelve bytes vacíos (útil para cerrar sesión conversacional).
    """
    if prompt:
        print(f"\n[Kimmi]: {prompt}")
    frames: List[bytes] = []
    has_spoken = False
    silent_chunks = 0
    wait_chunks = 0

    chunk_duration_sec = VAD_CHUNK / RATE
    max_silent_chunks = int(SILENCE_LIMIT_SEC / chunk_duration_sec)
    max_total_chunks = int(MAX_RECORD_SEC / chunk_duration_sec)
    pre_roll_len = max(1, int(VAD_PRE_ROLL_SEC / chunk_duration_sec))
    pre_roll: deque = deque(maxlen=pre_roll_len)
    max_wait_chunks = (
        int(max_wait_for_speech_sec / chunk_duration_sec)
        if max_wait_for_speech_sec is not None and max_wait_for_speech_sec > 0
        else None
    )

    if seed_audio:
        has_spoken, silent_chunks = _apply_seed_audio_to_recording(
            seed_audio, frames, pre_roll
        )
        if has_spoken:
            print("  [VAD]: Voz detectada (pre-roll wake)...")
            if silent_chunks >= max_silent_chunks:
                print("  [VAD]: end of turn on silence (in seed).")
                return b"".join(frames)

    total_chunks = 0
    vad_model.reset_states()

    while total_chunks < max_total_chunks:
        data = stream.read(VAD_CHUNK, exception_on_overflow=False)
        total_chunks += 1

        audio_int16 = np.frombuffer(data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor_chunk = torch.from_numpy(audio_float32)

        speech_prob = vad_model(tensor_chunk, RATE).item()

        if speech_prob > 0.5:
            if not has_spoken:
                print("  [VAD]: Voz detectada...")
                has_spoken = True
                frames.extend(pre_roll)
            frames.append(data)
            silent_chunks = 0
        else:
            if has_spoken:
                frames.append(data)
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    print("  [VAD]: end of turn on silence.")
                    break
            else:
                pre_roll.append(data)
                wait_chunks += 1
                if max_wait_chunks is not None and wait_chunks >= max_wait_chunks:
                    print("  [VAD]: Sin voz (timeout de escucha).")
                    return b""

    return b"".join(frames)


def transcribe_audio(audio_bytes: bytes) -> Tuple[str, bool]:
    """STT con filtro de calidad. Devuelve (texto, ok)."""
    if not audio_bytes or len(audio_bytes) < RATE * 2 // 10:
        # Menos de ~0.1 s de audio 16-bit mono
        return "", False

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)

    wav_buffer.seek(0)
    loc = current_locale()
    # No vad_filter: record_user_audio_vad ya recortó el audio con Silero,
    # así que volver a correr VAD aquí sería trabajo duplicado.
    segments_iter, _info = whisper_model.transcribe(
        wav_buffer,
        language=loc.whisper_language,
        task="transcribe",
        beam_size=WHISPER_BEAM_SIZE,
        best_of=WHISPER_BEAM_SIZE,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        initial_prompt=stt_initial_prompt(loc),
    )
    segments = list(segments_iter)
    if not segments:
        return "", False

    good = []
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        avg_lp = getattr(seg, "avg_logprob", 0.0)
        if no_speech > MAX_NO_SPEECH_PROB:
            continue
        if avg_lp < MIN_AVG_LOGPROB:
            continue
        text_part = (seg.text or "").strip()
        if text_part:
            good.append(text_part)

    if not good:
        return "", False

    text = " ".join(good).strip()
    text = re.sub(r"^[\s\-–—:]+", "", text)
    if len(text) < MIN_TRANSCRIPT_CHARS:
        return text, False
    return text, True


def stream_ollama_to_tts(
    messages,
    model: str = FAST_MODEL,
    tools=None,
) -> str:
    """Stream Ollama → TTS por chunks. Devuelve el texto completo.

    El print incremental del marker [Kimmi]: es contrato con desktop/bridge.py,
    que lo consume carácter a carácter para animar la onda.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "keep_alive": "24h",
        "stream": True,
        "options": dict(LLM_CHAT_OPTIONS),
    }
    if tools is not None:
        kwargs["tools"] = tools

    stream = ollama.chat(**kwargs)

    print("[Kimmi]: ", end="", flush=True)
    full_text = ""

    for chunk in stream:
        content = chunk["message"].get("content") or ""
        if not content:
            continue
        perf.mark("llm_ttft")
        print(content, end="", flush=True)
        full_text += content
        speech.feed(content)
        if speech.barge_in_happened():
            break

    print()
    return full_text.strip()


def _truncate(text: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_messages(
    history: List[Dict[str, Any]],
    user_text: str,
    memory_block: str = "",
) -> List[Dict[str, Any]]:
    """Arma los mensajes dejando el prefijo estable entre turnos.

    El bloque de memoria va pegado al mensaje del usuario, no al system, para
    que [system] + history sea byte-idéntico y el KV cache se reutilice.
    """
    max_msgs = HISTORY_MAX_TURNS * 2
    content = f"{user_text}\n\n{memory_block}" if memory_block else user_text
    return (
        [{"role": "system", "content": system_prompt(current_locale())}]
        + history[-max_msgs:]
        + [{"role": "user", "content": content}]
    )


def _confirm_action(stream, fn_name: str, fn_args: dict) -> bool:
    """Pide confirmación por voz; True si el usuario acepta."""
    prompt = confirmation_prompt(fn_name, fn_args)
    speak_text(prompt)
    audio_bytes = record_user_audio_vad(stream, prompt="Waiting for confirmation...")
    transcript, ok = transcribe_audio(audio_bytes)
    print(f"[Confirm]: {transcript!r} (ok={ok})")
    if not ok:
        speak_text(current_locale().spoken["not_heard_cancelled"])
        return False
    decision = parse_yes_no(transcript)
    if decision is True:
        return True
    if decision is False:
        speak_text(current_locale().spoken["cancelled"])
        return False
    speak_text(current_locale().spoken["confirm_unclear"])
    return False


def _chat_reply(messages: List[Dict[str, Any]], model: str) -> str:
    speech.begin_turn(allow_barge_in=BARGE_IN_ENABLED)
    filler_timer = arm_filler()
    try:
        final = stream_ollama_to_tts(messages, model=model)
    finally:
        if filler_timer is not None:
            filler_timer.cancel()
        speech.end_turn()
        suppress_wake_word(WAKE_COOLDOWN_SEC)
    return final or current_locale().spoken["done"]


def _run_tools_turn(
    messages: List[Dict[str, Any]],
    tools: List[dict],
    mic_stream,
) -> str:
    """Una llamada con tools (modelo pesado) + confirmación + resumen."""
    print(f"  [LLM]: using {LLM_MODEL} with tools...")
    # El 14B va sin stream y solo habla al terminar: aquí ya sabemos que va a
    # tardar, así que el aviso es inmediato y no espera al timer.
    _enqueue_filler(current_locale().filler_tools)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        keep_alive="24h",
    )

    tool_calls = response["message"].get("tool_calls") or []
    if not tool_calls:
        direct = (response["message"].get("content") or "").strip()
        if direct:
            speak_text(direct)
            return direct
        return _chat_reply(messages, FAST_MODEL)

    if len(tool_calls) > 1:
        print(
            f"  [Tools]: model asked for {len(tool_calls)} tools; "
            "only the first one runs."
        )
    tool_call = tool_calls[0]
    fn_name = tool_call["function"]["name"]
    fn_args = tool_call["function"].get("arguments") or {}
    if isinstance(fn_args, str):
        try:
            fn_args = json.loads(fn_args)
        except json.JSONDecodeError:
            fn_args = {}

    if fn_name not in available_tools:
        final = current_locale().spoken["tool_unavailable"]
        speak_text(final)
        return final

    if tool_needs_confirmation(fn_name, fn_args):
        if not _confirm_action(mic_stream, fn_name, fn_args):
            return current_locale().spoken["cancelled"]

    print(f"\n[Running]: {fn_name}")
    try:
        tool_result = available_tools[fn_name](**fn_args)
    except TypeError as exc:
        tool_result = f"Error de argumentos en {fn_name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        tool_result = f"Error al ejecutar {fn_name}: {exc}"

    messages.append(response["message"])
    messages.append({
        "role": "tool",
        "content": _truncate(str(tool_result)),
    })
    return _chat_reply(messages, LLM_MODEL)


def handle_user_turn(
    user_text: str,
    history: List[Dict[str, Any]],
    mic_stream,
) -> str:
    """
    Pipeline: cierre rápido | (opcional intent) | chat o tools.
    Devuelve el texto final del asistente para el historial.
    """
    _filler_played.clear()

    # Cierre conversacional: no LLM, no "¿necesitas algo más?"
    if is_conversation_closing(user_text):
        print("  [Closing]: goodbye/thanks detected")
        return speak_closing_ack()

    # RAG: recuperar recuerdos relevantes antes de llamar al LLM.
    retrieved_context = ""
    with perf.span("rag"):
        try:
            retrieved_context = memory_db.search_memory(user_text, n_results=3)
        except Exception as exc:  # noqa: BLE001
            print(f"  [Memory]: RAG failed: {exc}")
    if retrieved_context:
        print(f"\n[RAG]:\n{retrieved_context}")

    memory_block = format_memory_block(retrieved_context)

    # Si el asistente ofreció recordar y el usuario confirma, forzar remember_fact.
    force_memory = ENABLE_TOOLS and should_force_memory_save(user_text, history)
    effective_user_text = user_text
    if force_memory:
        fact = previous_user_fact(history)
        if fact:
            effective_user_text = current_locale().spoken["force_memory_save"].format(
                fact=fact
            )
            print("  [Memory]: proactive confirmation -> remember_fact")
        else:
            force_memory = False

    messages = _build_messages(history, effective_user_text, memory_block)

    # --- Modo rápido (sin router de intents) ---
    if not ENABLE_INTENT_ROUTER:
        if ENABLE_TOOLS:
            if force_memory:
                subset = tools_for_intent("memory", tools_schema)
                return _run_tools_turn(messages, subset or tools_schema, mic_stream)
            return _run_tools_turn(messages, tools_schema, mic_stream)
        return _chat_reply(messages, FAST_MODEL)

    # --- Modo con router de intents ---
    if force_memory:
        subset = tools_for_intent("memory", tools_schema)
        if subset:
            return _run_tools_turn(messages, subset, mic_stream)

    print(f"  [Intent]: classifying with {FAST_MODEL}...")
    intent_result = classify_intent(
        effective_user_text,
        model=FAST_MODEL,
        recent_context=history,
    )
    print(
        f"  [Intent]: {intent_result.intent} "
        f"(conf={intent_result.confidence:.2f}) — {intent_result.brief_reason}"
    )

    low_confidence = intent_result.confidence < INTENT_CONFIDENCE_THRESHOLD
    if intent_result.intent == "clarify" or low_confidence:
        final = current_locale().spoken["unclear_request"]
        speak_text(final)
        return final

    use_tools = ENABLE_TOOLS and intent_result.should_use_tools
    if not use_tools:
        return _chat_reply(messages, FAST_MODEL)

    subset = tools_for_intent(intent_result.intent, tools_schema)
    if not subset:
        return _chat_reply(messages, FAST_MODEL)
    return _run_tools_turn(messages, subset, mic_stream)


def _prepare_listen(mic_stream) -> None:
    """Anti-eco antes de volver a grabar (flush + pausa corta).

    En sesión activa el wake word no se usa, así que aquí solo interesa el eco
    residual del altavoz: esperar el cooldown completo del wake word metía más
    de un segundo de tiempo muerto antes de cada turno de seguimiento.
    """
    flush_mic_stream(mic_stream)
    reset_wake_word_model()
    settle_until = time.monotonic() + MIC_SETTLE_SEC
    while time.monotonic() < settle_until:
        try:
            mic_stream.read(OWW_CHUNK, exception_on_overflow=False)
        except Exception:
            break


def run_conversation_session(
    mic_stream,
    conversation_history: List[Dict[str, Any]],
    initial_audio: Optional[bytes] = None,
    strip_wake_on_first_turn: bool = False,
) -> None:
    """Tras la wake word: varios turnos seguidos hasta cierre o silencio.

    initial_audio: primer turno ya capturado (comando pegado a la wake word).
    strip_wake_on_first_turn: quitar 'Hey Kimmi' del transcript del 1er turno.
    """
    first_turn = True
    pending_audio: Optional[bytes] = initial_audio

    while True:
        wait_sec = None if first_turn else CONVERSATION_FOLLOWUP_TIMEOUT_SEC

        if pending_audio is not None:
            if first_turn:
                print("\n=== Using post-wake audio (command ran on) ===")
            else:
                print("\n=== Using barge-in audio ===")
            user_audio_bytes = pending_audio
            pending_audio = None
        else:
            if not first_turn:
                print("\n=== Listening (active session) ===")
                _prepare_listen(mic_stream)

            user_audio_bytes = record_user_audio_vad(
                mic_stream,
                max_wait_for_speech_sec=wait_sec,
            )

        if not user_audio_bytes:
            if first_turn:
                speak_text(current_locale().spoken["not_heard_session_end"])
            else:
                print("  [Session]: silence -> conversation over")
            break

        # El turno arranca cuando el usuario deja de hablar: todo lo que venga
        # después es latencia que él percibe.
        perf.start_turn()
        play_earcon()
        with perf.span("stt"):
            user_text, stt_ok = transcribe_audio(user_audio_bytes)
        if first_turn and stt_ok:
            stripped = strip_wake_phrase(user_text)
            if stripped != user_text:
                print(f"  [Wake]: cleaned transcript {user_text!r} -> {stripped!r}")
            user_text = stripped
            if strip_wake_on_first_turn and not user_text.strip():
                # Solo wake word en el audio sembrado → acuse y escuchar de verdad.
                print("  [Wake]: only the wake phrase after strip -> ack + listen")
                speak_wake_ack()
                strip_wake_on_first_turn = False
                user_audio_bytes = record_user_audio_vad(mic_stream)
                if not user_audio_bytes:
                    speak_text(current_locale().spoken["not_heard_session_end"])
                    break
                perf.start_turn()
                play_earcon()
                with perf.span("stt"):
                    user_text, stt_ok = transcribe_audio(user_audio_bytes)
                if stt_ok:
                    user_text = strip_wake_phrase(user_text)

        print(f"[You]: {user_text!r} (ok={stt_ok})")
        first_turn = False
        strip_wake_on_first_turn = False

        if not stt_ok or not (user_text or "").strip():
            speak_text(current_locale().spoken["not_heard_repeat"])
            perf.finish_turn()
            continue

        closing = is_conversation_closing(user_text)
        # Limpia captura previa por si acaso; speak_text/_chat_reply la rellenan.
        take_barge_in_audio()
        assistant_text = handle_user_turn(
            user_text,
            conversation_history,
            mic_stream,
        )
        perf.finish_turn()

        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({
            "role": "assistant",
            "content": assistant_text,
        })
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(conversation_history) > max_msgs:
            conversation_history[:] = conversation_history[-max_msgs:]

        if closing:
            print("  [Session]: closing detected")
            break

        # Si el usuario interrumpió, procesar lo que dijo sin pedirle que repita.
        barge_audio = take_barge_in_audio()
        if barge_audio:
            pending_audio = barge_audio
            print("  [Session]: barge-in -> next turn uses the captured audio")
            continue

        if not CONVERSATION_MODE:
            break


def capture_after_wake_word(mic_stream, ring_chunks: deque) -> Tuple[Optional[bytes], bool]:
    """Tras detectar wake word: ack inteligente + posible captura inmediata.

    Devuelve (initial_audio | None, strip_wake_on_first_turn).
    """
    seed = b"".join(ring_chunks)
    continues, peeked = speech_continues_after_wake(seed, mic_stream)
    seed_with_peek = seed + peeked

    if continues:
        print("[Wake]: command ran on -> no ack")
        first_audio = record_user_audio_vad(
            mic_stream,
            seed_audio=seed_with_peek,
        )
        if first_audio:
            return first_audio, True
        print("  [Wake]: empty capture after a ran-on command -> ack")

    speak_wake_ack()
    return None, False


def warmup() -> None:
    """Pay the one-off costs at boot instead of on the user's first turn.

    Covers the Ollama model load, the embedding model, Kokoro's first ONNX run
    (which also initializes espeak) and Whisper's CUDA allocation.
    """
    print("Warming up models...")
    loc = current_locale()

    try:
        ollama.chat(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive="24h",
            options={"num_predict": 1, "num_ctx": LLM_CHAT_OPTIONS["num_ctx"]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [Warmup]: Ollama {FAST_MODEL} did not answer ({exc})")

    if ENABLE_TOOLS:
        try:
            ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                keep_alive="24h",
                options={"num_predict": 1},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [Warmup]: Ollama {LLM_MODEL} did not answer ({exc})")

    try:
        memory_db.warmup()
    except Exception as exc:  # noqa: BLE001
        print(f"  [Warmup]: embeddings unavailable ({exc})")

    if kokoro is not None:
        try:
            kokoro.create(
                loc.spoken["done"],
                voice=loc.tts_voice,
                speed=loc.tts_speed,
                lang=loc.tts_lang,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [Warmup]: Kokoro failed ({exc})")

    # Después de Kokoro: la primera síntesis es la que inicializa espeak.
    # Solo el locale activo; la tanda 4 cebará el segundo si hace falta.
    try:
        started = time.monotonic()
        reused, synthesized = clip_cache_for(loc).prime(fixed_phrases(loc))
        print(
            f"  [Clips]: {reused + synthesized} cached for {loc.code} "
            f"({synthesized} synthesized in {time.monotonic() - started:.1f}s)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [Clips]: could not prepare the cache ({exc})")

    try:
        transcribe_audio(b"\x00\x00" * (RATE // 2))
    except Exception as exc:  # noqa: BLE001
        print(f"  [Warmup]: Whisper failed ({exc})")


def main():
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=OWW_CHUNK,
    )

    audio_sink.start()
    barge_in_monitor.start()
    warmup()

    conversation_history: List[Dict[str, Any]] = []
    ring_maxlen = max(1, int(WAKE_RING_BUFFER_SEC * RATE / OWW_CHUNK))
    wake_ring: deque = deque(maxlen=ring_maxlen)

    print("\n=== KIMMI READY ===")
    print(
        f"Fast: {FAST_MODEL} | Heavy: {LLM_MODEL} | User: {USER_NAME}"
    )
    print(
        f"Flags: INTENT_ROUTER={ENABLE_INTENT_ROUTER} | TOOLS={ENABLE_TOOLS} | "
        f"TTS={TTS_ENGINE} | CONVERSATION={CONVERSATION_MODE} | BARGE_IN={BARGE_IN_ENABLED}"
    )

    try:
        while True:
            data = mic_stream.read(OWW_CHUNK, exception_on_overflow=False)
            wake_ring.append(bytes(data))
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Durante/tras TTS no evaluamos wake word (evita falsos positivos por eco).
            if wake_word_suppressed():
                continue

            prediction = oww_model.predict(audio_data)
            scores = list(prediction.values())

            if any(score >= WAKE_WORD_THRESHOLD for score in scores):
                print("\n[Wake Detected!]")
                initial_audio, strip_wake = capture_after_wake_word(
                    mic_stream, wake_ring
                )
                wake_ring.clear()

                run_conversation_session(
                    mic_stream,
                    conversation_history,
                    initial_audio=initial_audio,
                    strip_wake_on_first_turn=strip_wake,
                )

                # Anti-eco antes de volver a escuchar wake word.
                flush_mic_stream(mic_stream)
                reset_wake_word_model()
                suppress_wake_word(WAKE_COOLDOWN_SEC)
                wake_ring.clear()
                print("\n=== KIMMI IDLE ===")

    except KeyboardInterrupt:
        print("\nShutting down Kimmi...")
    finally:
        speech.close()
        barge_in_monitor.close()
        audio_sink.close()
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()

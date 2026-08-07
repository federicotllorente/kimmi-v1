import os
import io
import json
import random
import re
import subprocess
import time
import wave
import queue
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyaudio
import torch
import ollama
from openwakeword.model import Model
from faster_whisper import WhisperModel
import sounddevice as sd

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
from tools.registry import build_system_prompt, build_tools, tools_for_intent
from tools.memory import KimmiMemory

# --- CONFIGURACIÓN DE PARÁMETROS ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# Silero VAD requiere tamaños de chunk específicos para 16kHz (512, 1024 o 1536 muestras)
VAD_CHUNK = 512  # ~32 ms por chunk
OWW_CHUNK = 1280  # ~80 ms por chunk para openWakeWord

WAKE_WORD_THRESHOLD = 0.5
# Tras TTS / fin de turno, ignorar wake word (eco del altavoz en el mic).
WAKE_COOLDOWN_SEC = 1.25
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
PIPER_MODEL = "piper/es_MX-claude-high.onnx"

# TTS: "kokoro" o "piper" — cambia esta variable para probar uno u otro.
TTS_ENGINE = "kokoro"
KOKORO_VOICE = "ef_dora"
KOKORO_SPEED = 1.2
KOKORO_MODEL = "kokoro/kokoro-v1_0.onnx"
KOKORO_VOICES = "kokoro/voices-v1_0.bin"
# Silencio al inicio/final (ms) para que el altavoz no corte la primera sílaba.
KOKORO_LEAD_IN_MS = 280
KOKORO_TRAIL_MS = 80
KOKORO_FADE_IN_MS = 50
# Calentar el dispositivo con silencio antes de la primera frase del turno.
TTS_DEVICE_WARMUP_MS = 80

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

# Variantes del acuse; usa {name} donde quieras el nombre.
WAKE_ACK_PHRASES = [
    "Sí, {name}?",
    "Dime, {name}.",
    "Te escucho.",
    "Ajá.",
    "Dime.",
    "Aquí estoy.",
    "¿Qué necesitas, {name}?",
]

# Cierres cuando el usuario termina (gracias, listo, etc.) — sin repreguntar.
CLOSING_ACK_PHRASES = [
    "De nada.",
    "Cuando quieras.",
    "Aquí estaré.",
    "Perfecto.",
    "Hasta luego.",
    "Avísame si me necesitas.",
    "Listo.",
    "Un placer.",
]

# Heurística: despedida / agradecimiento → cierre corto, sin LLM.
# Frases "exactas" (casi todo el utterance).
_CLOSING_RE = re.compile(
    r"(?i)^\s*(?:bueno[,.]?\s+|vale[,.]?\s+|ok(?:ay)?[,.]?\s+|perfecto[,.]?\s+)?"
    r"("
    r"gracias(?:\s+(?:mucho|mil|de\s+nuevo))?"
    r"|(?:muy\s+)?bien(?:\s*,?\s*gracias)?"
    r"|perfecto(?:\s*,?\s*gracias)?"
    r"|vale(?:\s*,?\s*gracias)?"
    r"|ok(?:ay)?(?:\s*,?\s*gracias)?"
    r"|listo(?:\s*,?\s*gracias)?"
    r"|eso\s+es\s+todo"
    r"|nada\s+m[aá]s(?:\s+por\s+ahora)?"
    r"|ya\s+est[aá]"
    r"|ya\s+bastante"
    r"|no\s+(?:necesito|preciso)\s+(?:nada\s+)?m[aá]s(?:\s+por\s+(?:ahora|el\s+momento))?"
    r"|por\s+ahora\s+(?:nada|estoy\s+bien|no(?:\s+necesito)?\s+nada)?"
    r"|(?:ahora\s+)?estoy\s+(?:bien|ok|okay)(?:\s*,?\s*gracias)?"
    r"|chau"
    r"|adi[oó]s"
    r"|hasta\s+luego"
    r"|nos\s+vemos"
    r"|bye"
    r"|cuando\s+quiera(?:s)?"
    r"|no\s+por\s+ahora"
    r")[\s\.\!\,]*$",
)

# Señales de cierre dentro de frases cortas (no hace falta que sea TODO el texto).
_CLOSING_SOFT_RE = re.compile(
    r"(?i)("
    r"\bgracias(?:\s+(?:mucho|mil))?\s*$"  # termina en gracias
    r"|\bno\s+(?:necesito|preciso)\s+(?:nada\s+)?m[aá]s\b"
    r"|\bnada\s+m[aá]s\s+por\s+ahora\b"
    r"|\bpor\s+el\s+momento\s+nada\b"
    r"|\bno\s+necesito\s+nada\s+m[aá]s\s+por\s+(?:ahora|el\s+momento)\b"
    r"|\bestoy\s+bien(?:\s*,?\s*gracias)?\b"
    r"|\beso\s+es\s+todo\b"
    r"|\bya\s+est[aá]\b"
    r"|\bhasta\s+luego\b"
    r"|\badi[oó]s\b"
    r")",
)

# Modelo ligero: clasificar intención + charla sin tools.
FAST_MODEL = "qwen2.5:3b"
# Modelo pesado: solo function calling y resumen tras tools.
LLM_MODEL = "qwen2.5:14b"

# STT: medium es más preciso que small (cuesta algo más de latencia en GPU).
# WHISPER_MODEL_SIZE = "tiny"
# WHISPER_MODEL_SIZE = "small"
WHISPER_MODEL_SIZE = "medium"
WHISPER_BEAM_SIZE = 5

STT_INITIAL_PROMPT = (
    "El usuario habla en español claro con Kimmi. "
    f"Se llama {USER_NAME}. "
    "Ejemplos: abre el navegador, abre Cursor, abre VS Code, abre la terminal, "
    "lista los archivos, lee la carpeta tools, escribe un archivo, "
    "busca en el código, Docker, logs del contenedor, portapapeles, "
    "recuerda que la IP es, qué IP tiene la Raspberry, "
    "cuánto es diecisiete por veintitrés, documentación de FastAPI, "
    "hola qué tal, cancela, sí confirma. "
    "Apps: " + ", ".join(list_allowed_app_ids()) + "."
)

# --- TTS (carga solo el motor elegido) ---
TTS_ENGINE = TTS_ENGINE.strip().lower()
if TTS_ENGINE not in ("kokoro", "piper"):
    raise ValueError(f"TTS_ENGINE inválido: {TTS_ENGINE!r}. Usa 'kokoro' o 'piper'.")

kokoro = None
if TTS_ENGINE == "kokoro":
    from kokoro_onnx import Kokoro

    print(f"Cargando Kokoro TTS ({KOKORO_VOICE})...")
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
else:
    print(f"Usando Piper TTS ({PIPER_MODEL})...")

# --- TOOLS (FUNCTION CALLING) ---
tools_schema, available_tools = build_tools(
    allowed_root=ALLOWED_DATA_ROOT,
    llm_model=LLM_MODEL,
    cursor_workspace=CURSOR_WORKSPACE,
    cursor_model=CURSOR_MODEL,
    cursor_api_key=os.environ.get("CURSOR_API_KEY"),
)

# --- INICIALIZACIÓN DE COMPONENTES ---
print("Cargando modelo Wake Word...")
oww_model = Model(
    # wakeword_models=["hey_jarvis"],
    wakeword_models=["oww/hey_kee_me.onnx"],
    inference_framework="onnx",
)

print("Cargando Silero VAD...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)

print("Cargando Faster-Whisper (STT)...")
print(f"  Modelo Whisper: {WHISPER_MODEL_SIZE} (beam={WHISPER_BEAM_SIZE})")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cuda",
    compute_type="float16",
)

audio = pyaudio.PyAudio()

# Timestamp monotonic hasta el cual se ignoran wake words (anti-eco TTS).
_wake_ignore_until = 0.0

# Memoria
memory_db = KimmiMemory()

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


# --- REPRODUCCIÓN Y SÍNTESIS DE AUDIO ---
# Flags de la reproducción en curso (barge-in).
_tts_barge_in = threading.Event()
_tts_skip_remaining = threading.Event()
# Audio capturado del usuario durante barge-in (para no pedirle que repita).
_barge_in_audio_bytes: Optional[bytes] = None
_barge_in_audio_lock = threading.Lock()


def take_barge_in_audio() -> Optional[bytes]:
    """Consume el audio capturado en el último barge-in (o None)."""
    global _barge_in_audio_bytes
    with _barge_in_audio_lock:
        data = _barge_in_audio_bytes
        _barge_in_audio_bytes = None
        return data


def _store_barge_in_audio(data: bytes) -> None:
    global _barge_in_audio_bytes
    with _barge_in_audio_lock:
        _barge_in_audio_bytes = data if data else None


def _apply_fade_in(waveform: np.ndarray, sample_rate: int, fade_ms: int) -> np.ndarray:
    """Fade-in lineal al inicio del audio con habla (tras el lead-in)."""
    if fade_ms <= 0 or waveform.size == 0:
        return waveform
    n = min(len(waveform), int(sample_rate * (fade_ms / 1000.0)))
    if n <= 1:
        return waveform
    out = waveform.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def _prepare_kokoro_waveform(samples, sample_rate: int) -> np.ndarray:
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.reshape(-1)

    lead = int(sample_rate * (KOKORO_LEAD_IN_MS / 1000.0))
    trail = int(sample_rate * (KOKORO_TRAIL_MS / 1000.0))
    speech = _apply_fade_in(waveform, sample_rate, KOKORO_FADE_IN_MS)
    parts = []
    if lead > 0:
        parts.append(np.zeros(lead, dtype=np.float32))
    parts.append(speech)
    if trail > 0:
        parts.append(np.zeros(trail, dtype=np.float32))
    return np.concatenate(parts) if parts else speech


def _mic_chunk_is_user_speech(
    audio_bytes: bytes,
    vad_threshold: float = BARGE_IN_VAD_THRESHOLD,
    min_rms: float = BARGE_IN_MIN_RMS,
) -> bool:
    """Heurística anti-eco: VAD alto + energía suficiente."""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(audio_float32)))) if audio_float32.size else 0.0
    if rms < min_rms:
        return False
    tensor_chunk = torch.from_numpy(audio_float32)
    try:
        speech_prob = float(vad_model(tensor_chunk, RATE).item())
    except Exception:
        return False
    return speech_prob >= vad_threshold


def _capture_barge_in_utterance(
    monitor,
    seed_frames: List[bytes],
) -> bytes:
    """Tras cortar el TTS, sigue grabando hasta silencio para completar la frase."""
    frames: List[bytes] = list(seed_frames)
    chunk_duration_sec = VAD_CHUNK / RATE
    max_silent_chunks = max(1, int(BARGE_IN_CAPTURE_SILENCE_SEC / chunk_duration_sec))
    max_total_chunks = int(MAX_RECORD_SEC / chunk_duration_sec)
    silent_chunks = 0
    total = 0

    # Tras cortar el TTS, umbral algo más bajo (ya no compite tanto el altavoz).
    while total < max_total_chunks:
        try:
            mic_data = monitor.read(VAD_CHUNK, exception_on_overflow=False)
        except Exception:
            break
        total += 1
        frames.append(mic_data)
        if _mic_chunk_is_user_speech(mic_data, vad_threshold=0.55, min_rms=0.015):
            silent_chunks = 0
        else:
            silent_chunks += 1
            if silent_chunks >= max_silent_chunks:
                break

    return b"".join(frames)


def _play_waveform_kokoro(
    waveform: np.ndarray,
    sample_rate: int,
    allow_barge_in: bool,
) -> None:
    """Reproduce waveform. Si hay barge-in: fade-out, corta y captura la frase del usuario."""
    if waveform.size == 0:
        return

    block = max(256, int(sample_rate * 0.032))  # ~32 ms
    chunk_ms = 1000.0 * VAD_CHUNK / RATE
    elapsed_ms = 0.0
    speech_run_ms = 0.0
    barge_latched = False
    pos = 0
    speech_seed: List[bytes] = []

    monitor = None
    if allow_barge_in and BARGE_IN_ENABLED:
        try:
            monitor = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=VAD_CHUNK,
            )
            vad_model.reset_states()
        except Exception as exc:  # noqa: BLE001
            print(f"  [Barge-in]: no se pudo abrir mic monitor ({exc})")
            monitor = None

    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
        ) as stream:
            # Calentar dispositivo (evita cortar el ataque de la primera sílaba).
            warm = int(sample_rate * (TTS_DEVICE_WARMUP_MS / 1000.0))
            if warm > 0:
                stream.write(np.zeros((warm, 1), dtype=np.float32))

            while pos < len(waveform):
                end = min(pos + block, len(waveform))
                chunk = waveform[pos:end]
                stream.write(chunk.reshape(-1, 1))
                pos = end
                elapsed_ms += 1000.0 * len(chunk) / sample_rate

                if (
                    monitor is not None
                    and not barge_latched
                    and elapsed_ms >= BARGE_IN_GRACE_MS
                ):
                    try:
                        mic_data = monitor.read(VAD_CHUNK, exception_on_overflow=False)
                    except Exception:
                        continue
                    if _mic_chunk_is_user_speech(mic_data):
                        speech_seed.append(mic_data)
                        speech_run_ms += chunk_ms
                        if speech_run_ms >= BARGE_IN_MIN_SPEECH_MS:
                            barge_latched = True
                            _tts_barge_in.set()
                            _tts_skip_remaining.set()
                            print(
                                "  [Barge-in]: te escuché; corto y te atiendo."
                            )
                            # Fade-out corto y salir (no terminar listas largas).
                            fade_n = int(sample_rate * (BARGE_IN_FADE_OUT_MS / 1000.0))
                            if fade_n > 0 and pos < len(waveform):
                                tail = waveform[pos : pos + fade_n].astype(np.float32)
                                if tail.size:
                                    tail = tail * np.linspace(
                                        1.0, 0.0, num=len(tail), dtype=np.float32
                                    )
                                    stream.write(tail.reshape(-1, 1))
                            break
                    else:
                        speech_run_ms = 0.0
                        speech_seed.clear()

            # Capturar el resto de la frase del usuario con el mic ya abierto.
            if barge_latched and monitor is not None:
                try:
                    captured = _capture_barge_in_utterance(monitor, speech_seed)
                    _store_barge_in_audio(captured)
                    print(
                        f"  [Barge-in]: capturados {len(captured)} bytes de audio"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [Barge-in]: no se pudo capturar audio ({exc})")
    finally:
        if monitor is not None:
            try:
                monitor.stop_stream()
                monitor.close()
            except Exception:
                pass


def play_tts_piper(text: str, allow_barge_in: bool = False):
    """Sintetiza texto a voz con Piper y reproduce el WAV."""
    if not text.strip():
        return

    wav_output = "temp_sentence.wav"
    cmd_piper = [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", wav_output]
    process = subprocess.Popen(
        cmd_piper,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    process.communicate(input=text)
    if process.returncode != 0 or not os.path.exists(wav_output):
        print(f"  [TTS Piper]: error sintetizando (code={process.returncode})")
        return

    wf = wave.open(wav_output, "rb")
    frames = wf.readframes(wf.getnframes())
    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    wf.close()

    if sampwidth == 2:
        audio_i16 = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        # Fallback simple: reproducción bloqueante clásica
        wf = wave.open(wav_output, "rb")
        p_out = pyaudio.PyAudio()
        stream_out = p_out.open(
            format=p_out.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(1024)
        while data:
            stream_out.write(data)
            data = wf.readframes(1024)
        stream_out.stop_stream()
        stream_out.close()
        p_out.terminate()
        wf.close()
        return

    if channels > 1:
        audio_i16 = audio_i16.reshape(-1, channels).mean(axis=1)
    # Reutilizar el mismo path interruptible (resample no; Piper suele ser 22050).
    _play_waveform_kokoro(audio_i16, framerate, allow_barge_in=allow_barge_in)


def play_tts_kokoro(text: str, allow_barge_in: bool = True):
    """Sintetiza texto a voz con Kokoro y reproduce (con barge-in opcional)."""
    if not text.strip() or kokoro is None:
        return
    samples, sample_rate = kokoro.create(
        text,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
        lang="es",
    )
    waveform = _prepare_kokoro_waveform(samples, sample_rate)
    _play_waveform_kokoro(waveform, sample_rate, allow_barge_in=allow_barge_in)


def play_tts_sentence(text: str, allow_barge_in: bool = True):
    """Despacha al motor TTS configurado en TTS_ENGINE."""
    if TTS_ENGINE == "piper":
        play_tts_piper(text, allow_barge_in=allow_barge_in)
    else:
        play_tts_kokoro(text, allow_barge_in=allow_barge_in)
    # Tras hablar, el mic suele captar el eco → no aceptar wake word un momento.
    suppress_wake_word(WAKE_COOLDOWN_SEC)


def _drain_sentence_queue(sentence_queue: queue.Queue) -> None:
    """Vacía oraciones pendientes tras barge-in."""
    while True:
        try:
            item = sentence_queue.get_nowait()
        except queue.Empty:
            break
        sentence_queue.task_done()
        if item is None:
            break


def tts_worker(sentence_queue: queue.Queue, allow_barge_in: bool = True):
    """Worker: reproduce oraciones; si hay barge-in, no lanza las siguientes."""
    while True:
        if _tts_skip_remaining.is_set():
            _drain_sentence_queue(sentence_queue)
            break

        sentence = sentence_queue.get()
        if sentence is None:
            sentence_queue.task_done()
            break

        if _tts_skip_remaining.is_set():
            sentence_queue.task_done()
            _drain_sentence_queue(sentence_queue)
            break

        play_tts_sentence(sentence, allow_barge_in=allow_barge_in)
        sentence_queue.task_done()

        if _tts_barge_in.is_set():
            _tts_skip_remaining.set()
            _drain_sentence_queue(sentence_queue)
            break


def speak_text(text: str, allow_barge_in: bool = True) -> bool:
    """Reproduce una o más oraciones por TTS (bloqueante).

    Devuelve True si hubo barge-in. El audio del usuario queda en
    take_barge_in_audio() para no pedirle que repita.
    """
    if not text or not text.strip():
        return False

    _tts_barge_in.clear()
    _tts_skip_remaining.clear()
    _store_barge_in_audio(b"")  # limpia captura previa

    print(f"[Kimmi]: {text}")
    sentence_queue: queue.Queue = queue.Queue()
    tts_thread = threading.Thread(
        target=tts_worker,
        args=(sentence_queue, allow_barge_in and BARGE_IN_ENABLED),
    )
    tts_thread.start()
    sentence_regex = re.compile(r"([^.!?\n]+[.!?\n]+)")
    buffer = text.strip()
    for match in sentence_regex.findall(buffer):
        if _tts_skip_remaining.is_set():
            break
        sentence_queue.put(match.strip())
        buffer = buffer[len(match) :]
    if buffer.strip() and not _tts_skip_remaining.is_set():
        sentence_queue.put(buffer.strip())
    sentence_queue.put(None)
    tts_thread.join()
    return _tts_barge_in.is_set()


def speak_wake_ack():
    """Acuse inmediato tras la wake word; frase aleatoria."""
    template = random.choice(WAKE_ACK_PHRASES)
    ack = template.format(name=USER_NAME)
    print(f"[Kimmi]: {ack}")
    # Sin barge-in en el acuse corto (muy breve y fácil de confundir con eco).
    play_tts_sentence(ack, allow_barge_in=False)


def is_conversation_closing(text: str) -> bool:
    """True si el usuario está cerrando el turno (gracias, listo, etc.).

    Si es True, NO se llama al LLM: solo un acuse breve, sin preguntas nuevas.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    # Quitar signos de apertura/puntuación final para matching más robusto.
    normalized = cleaned.lower().strip()
    normalized = re.sub(r"^[¿¡\s]+", "", normalized)
    normalized = re.sub(r"[\s\.\!\?\,;:]+$", "", normalized)

    if _CLOSING_RE.match(cleaned) or _CLOSING_RE.match(normalized):
        return True

    # Frases cortas que terminan en "gracias" o contienen despedida explícita.
    # Límite de longitud: evita cerrar pedidos largos que solo mencionan "gracias".
    if len(normalized) <= 90 and _CLOSING_SOFT_RE.search(normalized):
        return True

    starters = (
        "gracias",
        "perfecto, gracias",
        "perfecto gracias",
        "vale, gracias",
        "ok, gracias",
        "okay, gracias",
        "listo, gracias",
        "eso es todo",
        "nada más",
        "nada mas",
        "ya está",
        "ya esta",
        "hasta luego",
        "nos vemos",
        "no necesito nada más",
        "no necesito nada mas",
        "estoy bien",
        "ahora estoy bien",
    )
    return any(
        normalized == s
        or normalized.startswith(s + " ")
        or normalized.startswith(s + ",")
        or normalized.startswith(s + ".")
        for s in starters
    )


def speak_closing_ack() -> str:
    """Cierre breve; no pregunta si necesita algo más."""
    ack = random.choice(CLOSING_ACK_PHRASES)
    speak_text(ack, allow_barge_in=False)
    return ack


# --- CAPTURA Y TRANCRIPCIÓN ---
def record_user_audio_vad(
    stream,
    prompt: str = "",
    max_wait_for_speech_sec: Optional[float] = None,
) -> bytes:
    """Graba audio dinámicamente cortando al detectar silencio post-habla.

    Descarta el silencio inicial y guarda un pre-roll corto para no cortar
    el inicio de la primera palabra.

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
                    print("  [VAD]: Fin de intervención por silencio.")
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
    segments_iter, _info = whisper_model.transcribe(
        wav_buffer,
        language="es",
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=400,
            speech_pad_ms=200,
        ),
        beam_size=WHISPER_BEAM_SIZE,
        best_of=WHISPER_BEAM_SIZE,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        initial_prompt=STT_INITIAL_PROMPT,
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
    sentence_queue,
    model: str = FAST_MODEL,
    tools=None,
) -> str:
    """Stream Ollama → TTS por oraciones. Devuelve el texto completo."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "keep_alive": "24h",
        "stream": True,
    }
    if tools is not None:
        kwargs["tools"] = tools

    stream = ollama.chat(**kwargs)

    print("[Kimmi]: ", end="", flush=True)
    text_buffer = ""
    full_text = ""
    sentence_regex = re.compile(r"([^.!?\n]+[.!?\n]+)")

    for chunk in stream:
        content = chunk["message"].get("content") or ""
        if not content:
            continue
        print(content, end="", flush=True)
        text_buffer += content
        full_text += content

        matches = sentence_regex.findall(text_buffer)
        if matches:
            for match in matches:
                if _tts_skip_remaining.is_set():
                    text_buffer = ""
                    break
                sentence_queue.put(match)
                text_buffer = text_buffer[len(match) :]

    if text_buffer.strip() and not _tts_skip_remaining.is_set():
        sentence_queue.put(text_buffer.strip())

    print()
    return full_text.strip()


def _truncate(text: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_messages(
    system_prompt: str,
    history: List[Dict[str, Any]],
    user_text: str,
) -> List[Dict[str, Any]]:
    max_msgs = HISTORY_MAX_TURNS * 2
    return (
        [{"role": "system", "content": system_prompt}]
        + history[-max_msgs:]
        + [{"role": "user", "content": user_text}]
    )


def _confirm_action(stream, fn_name: str, fn_args: dict) -> bool:
    """Pide confirmación por voz; True si el usuario acepta."""
    prompt = confirmation_prompt(fn_name, fn_args)
    speak_text(prompt)
    audio_bytes = record_user_audio_vad(stream, prompt="Esperando confirmación...")
    transcript, ok = transcribe_audio(audio_bytes)
    print(f"[Confirmación]: {transcript!r} (ok={ok})")
    if not ok:
        speak_text("No te escuché bien. Cancelado.")
        return False
    decision = parse_yes_no(transcript)
    if decision is True:
        return True
    if decision is False:
        speak_text("Cancelado.")
        return False
    speak_text("No entendí la confirmación. Cancelado.")
    return False


def _chat_reply(messages: List[Dict[str, Any]], model: str) -> str:
    _tts_barge_in.clear()
    _tts_skip_remaining.clear()
    _store_barge_in_audio(b"")
    sentence_queue: queue.Queue = queue.Queue()
    tts_thread = threading.Thread(
        target=tts_worker,
        args=(sentence_queue, BARGE_IN_ENABLED),
    )
    tts_thread.start()
    final = stream_ollama_to_tts(messages, sentence_queue, model=model)
    sentence_queue.put(None)
    tts_thread.join()
    return final or "Listo."


def _run_tools_turn(
    messages: List[Dict[str, Any]],
    tools: List[dict],
    mic_stream,
) -> str:
    """Una llamada con tools (modelo pesado) + confirmación + resumen."""
    print(f"  [LLM]: usando {LLM_MODEL} con tools...")
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
            f"  [Tools]: el modelo pidió {len(tool_calls)} tools; "
            "solo se ejecuta la primera."
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
        final = "No puedo usar esa herramienta."
        speak_text(final)
        return final

    if tool_needs_confirmation(fn_name, fn_args):
        if not _confirm_action(mic_stream, fn_name, fn_args):
            return "Cancelado."

    print(f"\n[Kimmi ejecutando {fn_name}...]")
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
    # Cierre conversacional: no LLM, no "¿necesitas algo más?"
    if is_conversation_closing(user_text):
        print("  [Cierre]: despedida/agradecimiento detectado")
        return speak_closing_ack()

    # RAG: recuperar recuerdos relevantes antes de llamar al LLM.
    retrieved_context = ""
    try:
        retrieved_context = memory_db.search_memory(user_text, n_results=3)
    except Exception as exc:  # noqa: BLE001
        print(f"  [Memoria]: fallo RAG: {exc}")
    if retrieved_context:
        print(f"\n[RAG Memoria]:\n{retrieved_context}")

    system_prompt = build_system_prompt(
        ALLOWED_DATA_ROOT,
        memory_context=retrieved_context,
    )

    # Si el asistente ofreció recordar y el usuario confirma, forzar remember_fact.
    force_memory = ENABLE_TOOLS and should_force_memory_save(user_text, history)
    effective_user_text = user_text
    if force_memory:
        fact = previous_user_fact(history)
        if fact:
            effective_user_text = (
                f"Sí, usa remember_fact para guardar este dato exactamente: {fact}"
            )
            print("  [Memoria]: confirmación proactiva → remember_fact")
        else:
            force_memory = False

    messages = _build_messages(system_prompt, history, effective_user_text)

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

    print(f"  [Intent]: clasificando con {FAST_MODEL}...")
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
        final = (
            "No estoy seguro de lo que pediste. "
            "¿Puedes reformularlo en una frase más concreta?"
        )
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
    """Anti-eco antes de volver a grabar (flush + cooldown corto)."""
    flush_mic_stream(mic_stream)
    reset_wake_word_model()
    # Esperar a que termine el cooldown de eco del TTS sin evaluar wake word.
    while wake_word_suppressed():
        try:
            mic_stream.read(OWW_CHUNK, exception_on_overflow=False)
        except Exception:
            break


def run_conversation_session(
    mic_stream,
    conversation_history: List[Dict[str, Any]],
) -> None:
    """Tras la wake word: varios turnos seguidos hasta cierre o silencio."""
    first_turn = True
    pending_audio: Optional[bytes] = None

    while True:
        wait_sec = None if first_turn else CONVERSATION_FOLLOWUP_TIMEOUT_SEC

        if pending_audio is not None:
            print("\n=== Usando audio del barge-in ===")
            user_audio_bytes = pending_audio
            pending_audio = None
        else:
            if not first_turn:
                print("\n=== Escuchando (sesión activa) ===")
                _prepare_listen(mic_stream)

            user_audio_bytes = record_user_audio_vad(
                mic_stream,
                max_wait_for_speech_sec=wait_sec,
            )

        if not user_audio_bytes:
            if first_turn:
                speak_text("No te escuché. Cuando quieras, vuelve a llamarme.")
            else:
                print("  [Sesión]: silencio → fin de conversación")
            break

        user_text, stt_ok = transcribe_audio(user_audio_bytes)
        print(f"[Tú]: {user_text!r} (ok={stt_ok})")
        first_turn = False

        if not stt_ok:
            speak_text("No te escuché bien, ¿puedes repetir?")
            continue

        closing = is_conversation_closing(user_text)
        # Limpia captura previa por si acaso; speak_text/_chat_reply la rellenan.
        take_barge_in_audio()
        assistant_text = handle_user_turn(
            user_text,
            conversation_history,
            mic_stream,
        )

        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({
            "role": "assistant",
            "content": assistant_text,
        })
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(conversation_history) > max_msgs:
            conversation_history[:] = conversation_history[-max_msgs:]

        if closing:
            print("  [Sesión]: cierre detectado")
            break

        # Si el usuario interrumpió, procesar lo que dijo sin pedirle que repita.
        barge_audio = take_barge_in_audio()
        if barge_audio:
            pending_audio = barge_audio
            print("  [Sesión]: barge-in → siguiente turno con audio capturado")
            continue

        if not CONVERSATION_MODE:
            break


def main():
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=OWW_CHUNK,
    )

    conversation_history: List[Dict[str, Any]] = []

    print("\n=== KIMMI EN ESPERA ===")
    print(
        f"Rápido: {FAST_MODEL} | Pesado: {LLM_MODEL} | Usuario: {USER_NAME}"
    )
    print(
        f"Flags: INTENT_ROUTER={ENABLE_INTENT_ROUTER} | TOOLS={ENABLE_TOOLS} | "
        f"TTS={TTS_ENGINE} | CONVERSATION={CONVERSATION_MODE} | BARGE_IN={BARGE_IN_ENABLED}"
    )

    try:
        while True:
            data = mic_stream.read(OWW_CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Durante/tras TTS no evaluamos wake word (evita falsos positivos por eco).
            if wake_word_suppressed():
                continue

            prediction = oww_model.predict(audio_data)
            scores = list(prediction.values())

            if any(score >= WAKE_WORD_THRESHOLD for score in scores):
                print("\n[Wake Word Detectado!]")
                speak_wake_ack()

                run_conversation_session(mic_stream, conversation_history)

                # Anti-eco antes de volver a escuchar wake word.
                flush_mic_stream(mic_stream)
                reset_wake_word_model()
                suppress_wake_word(WAKE_COOLDOWN_SEC)
                print("\n=== Volviendo a estado de espera ===")

    except KeyboardInterrupt:
        print("\nCerrando Kimmi...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()

import os
import io
import json
import random
import re
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
from kokoro_onnx import Kokoro
import sounddevice as sd

from tools.app_allowlist import list_allowed_app_ids
from tools.intent import (
    INTENT_CONFIDENCE_THRESHOLD,
    classify_intent,
    confirmation_prompt,
    parse_yes_no,
    tool_needs_confirmation,
)
from tools.registry import build_system_prompt, build_tools, tools_for_intent

# --- CONFIGURACIÓN DE PARÁMETROS ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# Silero VAD requiere tamaños de chunk específicos para 16kHz (512, 1024 o 1536 muestras)
VAD_CHUNK = 512  # ~32 ms por chunk
OWW_CHUNK = 1280  # ~80 ms por chunk para openWakeWord

WAKE_WORD_THRESHOLD = 0.5
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
PIPER_MODEL = "es_MX-claude-high.onnx"

# Nombre del usuario para el acuse del wake word.
USER_NAME = "Fede"

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

# Modelo ligero: clasificar intención + charla sin tools.
FAST_MODEL = "qwen2.5:3b"
# Modelo pesado: solo function calling y resumen tras tools.
LLM_MODEL = "qwen2.5:14b"

# STT: medium es más preciso que small (cuesta algo más de latencia en GPU).
WHISPER_MODEL_SIZE = "medium"
WHISPER_BEAM_SIZE = 5

STT_INITIAL_PROMPT = (
    "El usuario habla en español claro con Jarvis. "
    f"Se llama {USER_NAME}. "
    "Ejemplos: abre el navegador, abre Cursor, abre VS Code, abre la terminal, "
    "lista los archivos, lee la carpeta tools, escribe un archivo, "
    "busca en el código, Docker, logs del contenedor, portapapeles, "
    "cuánto es diecisiete por veintitrés, documentación de FastAPI, "
    "hola qué tal, cancela, sí confirma. "
    "Apps: " + ", ".join(list_allowed_app_ids()) + "."
)

kokoro = Kokoro("kokoro/kokoro-v1_0.onnx", "kokoro/voices-v1_0.bin")

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
    wakeword_models=["hey_jarvis"],
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

# NO BORRAR ESTE COMENTARIO
# --- REPRODUCCIÓN Y SÍNTESIS DE AUDIO (PIPER TTS) ---
# def play_tts_sentence(text: str):
#     """Sintetiza texto a voz usando Piper y reproduce el audio."""
#     if not text.strip():
#         return
    
#     # Redirigimos la salida de Piper directamente a PyAudio mediante aplay / ffplay / subprocess
#     # Para simplicidad multiplataforma, guardamos un WAV temporal si no usas aplay/paplay:
#     wav_output = "temp_sentence.wav"
#     cmd_piper = [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", wav_output]
    
#     # Genera el archivo de audio para la oración actual
#     process = subprocess.Popen(cmd_piper, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
#     process.communicate(input=text)
    
#     if not os.path.exists(wav_output):
#         return

#     # Reproducir WAV generado
#     wf = wave.open(wav_output, 'rb')
#     p_out = pyaudio.PyAudio()
#     stream_out = p_out.open(
#         format=p_out.get_format_from_width(wf.getsampwidth()),
#         channels=wf.getnchannels(),
#         rate=wf.getframerate(),
#         output=True
#     )
    
#     data = wf.readframes(1024)
#     while len(data) > 0:
#         stream_out.write(data)
#         data = wf.readframes(1024)
        
#     stream_out.stop_stream()
#     stream_out.close()
#     p_out.terminate()
#     wf.close()

# --- REPRODUCCIÓN Y SÍNTESIS DE AUDIO (KOKORO TTS) ---
def play_tts_sentence(text: str):
    if not text.strip():
        return
    samples, sample_rate = kokoro.create(text, voice="ef_dora", speed=1.0, lang="es")
    sd.play(samples, sample_rate)
    sd.wait()


def tts_worker(sentence_queue: queue.Queue):
    """Worker corriendo en hilo secundario procesando oraciones de la cola."""
    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            sentence_queue.task_done()
            break
        play_tts_sentence(sentence)
        sentence_queue.task_done()


def speak_text(text: str):
    """Reproduce una o más oraciones por TTS (bloqueante)."""
    if not text or not text.strip():
        return
    print(f"[Jarvis]: {text}")
    sentence_queue: queue.Queue = queue.Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,))
    tts_thread.start()
    sentence_regex = re.compile(r"([^.!?\n]+[.!?\n]+)")
    buffer = text.strip()
    for match in sentence_regex.findall(buffer):
        sentence_queue.put(match.strip())
        buffer = buffer[len(match) :]
    if buffer.strip():
        sentence_queue.put(buffer.strip())
    sentence_queue.put(None)
    tts_thread.join()


def speak_wake_ack():
    """Acuse inmediato tras la wake word; frase aleatoria."""
    template = random.choice(WAKE_ACK_PHRASES)
    ack = template.format(name=USER_NAME)
    print(f"[Jarvis]: {ack}")
    play_tts_sentence(ack)


# --- CAPTURA Y TRANCRIPCIÓN ---
def record_user_audio_vad(stream, prompt: str = "") -> bytes:
    """Graba audio dinámicamente cortando al detectar silencio post-habla.

    Descarta el silencio inicial y guarda un pre-roll corto para no cortar
    el inicio de la primera palabra.
    """
    if prompt:
        print(f"\n[Jarvis]: {prompt}")
    frames: List[bytes] = []
    has_spoken = False
    silent_chunks = 0

    chunk_duration_sec = VAD_CHUNK / RATE
    max_silent_chunks = int(SILENCE_LIMIT_SEC / chunk_duration_sec)
    max_total_chunks = int(MAX_RECORD_SEC / chunk_duration_sec)
    pre_roll_len = max(1, int(VAD_PRE_ROLL_SEC / chunk_duration_sec))
    pre_roll: deque = deque(maxlen=pre_roll_len)

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

    print("[Jarvis]: ", end="", flush=True)
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
                sentence_queue.put(match)
                text_buffer = text_buffer[len(match) :]

    if text_buffer.strip():
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


def handle_user_turn(
    user_text: str,
    history: List[Dict[str, Any]],
    mic_stream,
) -> str:
    """
    Pipeline: intent → chat o tools (subset) → confirmación → respuesta.
    Devuelve el texto final del asistente para el historial.
    """
    system_prompt = build_system_prompt(ALLOWED_DATA_ROOT)

    print(f"  [Intent]: clasificando con {FAST_MODEL}...")
    intent_result = classify_intent(
        user_text,
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

    if not intent_result.should_use_tools:
        # Charla: modelo ligero (no hace falta el 14B)
        messages = _build_messages(system_prompt, history, user_text)
        sentence_queue: queue.Queue = queue.Queue()
        tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,))
        tts_thread.start()
        final = stream_ollama_to_tts(messages, sentence_queue, model=FAST_MODEL)
        sentence_queue.put(None)
        tts_thread.join()
        return final or "Listo."

    # Acciones: modelo pesado solo aquí
    print(f"  [LLM]: usando {LLM_MODEL} para tools...")
    subset = tools_for_intent(intent_result.intent, tools_schema)
    messages = _build_messages(system_prompt, history, user_text)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        tools=subset,
        keep_alive="24h",
    )

    tool_calls = response["message"].get("tool_calls") or []
    if not tool_calls:
        direct = (response["message"].get("content") or "").strip()
        if direct:
            speak_text(direct)
            return direct
        # Fallback chat con modelo ligero
        sentence_queue = queue.Queue()
        tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,))
        tts_thread.start()
        final = stream_ollama_to_tts(messages, sentence_queue, model=FAST_MODEL)
        sentence_queue.put(None)
        tts_thread.join()
        return final or "Listo."

    # Una sola tool por turno
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

    print(f"\n[Jarvis ejecutando {fn_name}...]")
    try:
        tool_result = available_tools[fn_name](**fn_args)
    except TypeError as exc:
        tool_result = f"Error de argumentos en {fn_name}: {exc}"
    except Exception as exc:  # noqa: BLE001 — voz: informar fallo
        tool_result = f"Error al ejecutar {fn_name}: {exc}"

    messages.append(response["message"])
    messages.append({
        "role": "tool",
        "content": _truncate(str(tool_result)),
    })

    sentence_queue = queue.Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,))
    tts_thread.start()
    # Resumen post-tool: 14B (ya cargado en este path)
    final = stream_ollama_to_tts(messages, sentence_queue, model=LLM_MODEL)
    sentence_queue.put(None)
    tts_thread.join()
    return final or "Hecho."


def main():
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=OWW_CHUNK,
    )

    conversation_history: List[Dict[str, Any]] = []

    print("\n=== JARVIS EN ESPERA (intent router + tools + streaming) ===")
    print(f"Rápido: {FAST_MODEL} | Pesado (tools): {LLM_MODEL} | Usuario: {USER_NAME}")

    try:
        while True:
            data = mic_stream.read(OWW_CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            prediction = oww_model.predict(audio_data)
            scores = list(prediction.values())

            if any(score >= WAKE_WORD_THRESHOLD for score in scores):
                print("\n[Wake Word Detectado!]")
                speak_wake_ack()

                user_audio_bytes = record_user_audio_vad(mic_stream)
                user_text, stt_ok = transcribe_audio(user_audio_bytes)
                print(f"[Tú]: {user_text!r} (ok={stt_ok})")

                if not stt_ok:
                    speak_text("No te escuché bien, ¿puedes repetir?")
                    print("\n=== Volviendo a estado de espera ===")
                    continue

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
                # Mantener solo los últimos N turnos
                max_msgs = HISTORY_MAX_TURNS * 2
                if len(conversation_history) > max_msgs:
                    conversation_history[:] = conversation_history[-max_msgs:]

                print("\n=== Volviendo a estado de espera ===")

    except KeyboardInterrupt:
        print("\nCerrando Jarvis...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()

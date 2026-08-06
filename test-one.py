import os
import io
import re
import wave
import queue
import threading
import subprocess
import numpy as np
import pyaudio
import torch
import ollama
from openwakeword.model import Model
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
import sounddevice as sd

from tools.read_files import read_directory_files
from tools.list_files import list_directory
from tools.search_files import search_files
from tools.write_file import write_file
from tools.cursor_agent import run_cursor_agent

# --- CONFIGURACIÓN DE PARÁMETROS ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# Silero VAD requiere tamaños de chunk específicos para 16kHz (512, 1024 o 1536 muestras)
VAD_CHUNK = 512  # ~32 ms por chunk
OWW_CHUNK = 1280 # ~80 ms por chunk para openWakeWord

WAKE_WORD_THRESHOLD = 0.5
SILENCE_LIMIT_SEC = 0.8  # Segundos de silencio continuo para cortar grabación
MAX_RECORD_SEC = 15.0    # Tiempo máximo de grabación de seguridad

# Sandbox: el asistente solo puede leer/escribir archivos dentro de este directorio
# ALLOWED_DATA_ROOT = os.path.realpath("./data")
ALLOWED_DATA_ROOT = os.path.realpath("./")

# Cursor SDK (local one-shot). Requiere: pip install cursor-sdk y CURSOR_API_KEY en el entorno.
CURSOR_WORKSPACE = os.path.realpath("./")
CURSOR_MODEL = "composer-2.5"

# Rutas de modelos
# PIPER_EXE = "piper"  # O la ruta absoluta a tu ejecutable de piper: "C:/piper/piper.exe"
PIPER_EXE = "proj-env/Scripts/piper"
PIPER_MODEL = "es_MX-claude-high.onnx" # Modelo de voz en español para Piper
LLM_MODEL = "llama3.2:3b"

kokoro = Kokoro("kokoro/kokoro-v1_0.onnx", "kokoro/voices-v1_0.bin")

# --- TOOLS (FUNCTION CALLING) ---
tools_schema = [
    {
        'type': 'function',
        'function': {
            'name': 'read_directory_files',
            'description': (
                'Lee el contenido de los archivos de texto dentro del directorio de datos permitido '
                f'({ALLOWED_DATA_ROOT}) o de una subcarpeta suya. No puede acceder fuera de ese directorio.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'directory_path': {
                        'type': 'string',
                        'description': (
                            'Ruta relativa al directorio permitido (ej: ".", "notas", "documentos/2024") '
                            'o una ruta absoluta que esté dentro de ese directorio.'
                        ),
                    },
                },
                'required': ['directory_path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_directory',
            'description': (
                'Lista archivos y carpetas dentro del directorio permitido '
                f'({ALLOWED_DATA_ROOT}). No puede listar fuera de ese directorio.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': 'Ruta relativa al directorio permitido (ej: ".", "notas").',
                    },
                    'recursive': {
                        'type': 'boolean',
                        'description': 'Si es true, lista el árbol completo. Por defecto false.',
                    },
                },
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_files',
            'description': (
                'Busca un texto en archivos .txt, .md, .json y .py dentro del directorio permitido '
                f'({ALLOWED_DATA_ROOT}).'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Texto a buscar (sin distinguir mayúsculas).',
                    },
                    'path': {
                        'type': 'string',
                        'description': 'Subcarpeta relativa donde buscar (por defecto ".").',
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_file',
            'description': (
                'Escribe o crea un archivo de texto (.txt, .md, .json, .py) solo dentro del '
                f'directorio permitido ({ALLOWED_DATA_ROOT}).'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'file_path': {
                        'type': 'string',
                        'description': 'Ruta relativa del archivo a escribir (ej: "notas/idea.txt").',
                    },
                    'content': {
                        'type': 'string',
                        'description': 'Contenido completo a guardar en el archivo.',
                    },
                },
                'required': ['file_path', 'content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'run_cursor_agent',
            'description': (
                'Delega una tarea de código al agente de Cursor (SDK local) sobre el workspace '
                f'{CURSOR_WORKSPACE}. Úsala para editar código, refactorizar, explicar bugs o '
                'implementar cambios en el proyecto.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'prompt': {
                        'type': 'string',
                        'description': 'Instrucción clara para el agente de Cursor.',
                    },
                },
                'required': ['prompt'],
            },
        },
    },
]


def _read_directory_files_sandboxed(directory_path: str) -> str:
    """Wrapper que fuerza el sandbox ALLOWED_DATA_ROOT en la herramienta de lectura."""
    return read_directory_files(directory_path, allowed_root=ALLOWED_DATA_ROOT)


def _list_directory_sandboxed(path: str, recursive: bool = False) -> str:
    return list_directory(path, allowed_root=ALLOWED_DATA_ROOT, recursive=recursive)


def _search_files_sandboxed(query: str, path: str = ".") -> str:
    return search_files(query, allowed_root=ALLOWED_DATA_ROOT, path=path)


def _write_file_sandboxed(file_path: str, content: str) -> str:
    return write_file(file_path, content, allowed_root=ALLOWED_DATA_ROOT)


def _run_cursor_agent_wrapped(prompt: str) -> str:
    return run_cursor_agent(
        prompt,
        workspace=CURSOR_WORKSPACE,
        model=CURSOR_MODEL,
        api_key=os.environ.get("CURSOR_API_KEY"),
    )


available_tools = {
    'read_directory_files': _read_directory_files_sandboxed,
    'list_directory': _list_directory_sandboxed,
    'search_files': _search_files_sandboxed,
    'write_file': _write_file_sandboxed,
    'run_cursor_agent': _run_cursor_agent_wrapped,
}

# --- INICIALIZACIÓN DE COMPONENTES ---
print("Cargando modelo Wake Word...")
oww_model = Model(
  wakeword_models=["hey_jarvis"],
  inference_framework="onnx"
)
# oww_model = Model(inference_framework="onnx")

print("Cargando Silero VAD...")
vad_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)

print("Cargando Faster-Whisper (STT)...")
# Usa "cuda" y "float16" si tienes GPU NVIDIA, o "cpu" e "int8" si ejecutas en procesador
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
# whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
# whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

audio = pyaudio.PyAudio()

# --- REPRODUCCIÓN Y SÍNTESIS DE AUDIO ---
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
    if not text.strip(): return
    # Voces ES en Kokoro v1.0: ef_dora, em_alex, em_santa
    samples, sample_rate = kokoro.create(text, voice="ef_dora", speed=1.0, lang="es")
    sd.play(samples, sample_rate)
    sd.wait()

def tts_worker(sentence_queue: queue.Queue):
    """Worker corriendo en hilo secundario procesando oraciones de la cola."""
    while True:
        sentence = sentence_queue.get()
        if sentence is None:  # Señal de fin de respuesta
            sentence_queue.task_done()
            break
        
        play_tts_sentence(sentence)
        sentence_queue.task_done()

# --- CAPTURA Y TRANCRIPCIÓN ---
def record_user_audio_vad(stream) -> bytes:
    """Graba audio dinámicamente cortando al detectar silencio post-habla."""
    print("\n[Jarvis]: Te escucho...")
    frames = []
    has_spoken = False
    silent_chunks = 0
    
    chunk_duration_sec = VAD_CHUNK / RATE
    max_silent_chunks = int(SILENCE_LIMIT_SEC / chunk_duration_sec)
    max_total_chunks = int(MAX_RECORD_SEC / chunk_duration_sec)
    
    total_chunks = 0
    vad_model.reset_states() # Reinicia la memoria recurrente del modelo VAD
    
    while total_chunks < max_total_chunks:
        data = stream.read(VAD_CHUNK, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1
        
        # Convertir bytes int16 a Tensor Float32 normalizado [-1.0, 1.0]
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor_chunk = torch.from_numpy(audio_float32)
        
        # Obtener probabilidad de presencia de voz (0.0 a 1.0)
        speech_prob = vad_model(tensor_chunk, RATE).item()
        
        if speech_prob > 0.5:
            if not has_spoken:
                print("  [VAD]: Voz detectada...")
                has_spoken = True
            silent_chunks = 0
        else:
            if has_spoken:
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    print("  [VAD]: Fin de intervención por silencio.")
                    break
                    
    return b''.join(frames)

def transcribe_audio(audio_bytes: bytes) -> str:
    """Convierte los bytes de audio a texto usando faster-whisper."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    
    wav_buffer.seek(0)
    # segments, _ = whisper_model.transcribe(wav_buffer, language="es")
    segments, _ = whisper_model.transcribe(
        wav_buffer, 
        language="es",
        initial_prompt="Transcripción de un comando de voz claro para el asistente local Jarvis."
    )
    return " ".join([segment.text for segment in segments]).strip()

def stream_ollama_to_tts(messages, sentence_queue):
    """Envía la respuesta streaming de Ollama directamente a la cola de TTS por oraciones."""
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        keep_alive="24h",
        stream=True
    )
    
    print("[Jarvis]: ", end="", flush=True)
    text_buffer = ""
    sentence_regex = re.compile(r'([^.!?\n]+[.!?\n]+)')
    
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end="", flush=True)
        text_buffer += content
        
        matches = sentence_regex.findall(text_buffer)
        if matches:
            for match in matches:
                sentence_queue.put(match)
                text_buffer = text_buffer[len(match):]
            
    if text_buffer.strip():
        sentence_queue.put(text_buffer.strip())
        
    print()

# --- BUCLE PRINCIPAL ---
def main():
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=OWW_CHUNK
    )
    
    print("\n=== JARVIS EN ESPERA (Function Calling + Streaming activo) ===")
    
    try:
        while True:
            # 1. Escucha continua en busca de Wake Word
            data = mic_stream.read(OWW_CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Predicción con openWakeWord
            prediction = oww_model.predict(audio_data)
            
            # Comprobación de activación (ej: modelo por defecto como "alexa" o "hey_jarvis")
            scores = list(prediction.values())
            
            if any(score >= WAKE_WORD_THRESHOLD for score in scores):
                print("\n[Wake Word Detectado!]")
                
                # 2. Grabación dinámica mediante Silero VAD
                user_audio_bytes = record_user_audio_vad(mic_stream)
                
                # 3. Transcripción (STT)
                user_text = transcribe_audio(user_audio_bytes)
                print(f"[Tú]: {user_text}")
                
                if user_text:
                    # 4. Procesamiento LLM (Ollama)
                    system_prompt = (
                        "Eres Jarvis, un asistente local conciso, directo y útil. "
                        "Responde siempre en español. Usa frases cortas y conversacionales (1 o 2 oraciones máximo). "
                        "Evita completamente el uso de caracteres Markdown (sin asteriscos, guiones o numerales) o "
                        "listas, ya que tu respuesta será leída por un sintetizador de voz. "
                        "Si no entiendes la pregunta que te hace el usuario, o lo que dice el usuario no tiene mucho sentido, "
                        "no dudes en preguntarle qué quiere decir o pedirle que reformule la pregunta o petición. "
                        "Si te dice que no necesita nada más por ahora, simplemente con decirle \"Vale\" o "
                        "algo similar, es suficiente. Recuerda, sé conciso y pragmático. "
                        "Tienes herramientas locales para listar, buscar, leer y escribir archivos solo dentro del "
                        f"directorio permitido ({ALLOWED_DATA_ROOT}) y sus subcarpetas. "
                        "Si el usuario pide algo fuera de ahí, dile que no tienes permiso para acceder a esa ruta. "
                        "También puedes delegar tareas de código al agente de Cursor con run_cursor_agent."
                    )
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ]
                    
                    # 1. Primera consulta a Ollama con evaluador de Herramientas (sin streaming inicial para inspeccionar la intención)
                    response = ollama.chat(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=tools_schema,
                        keep_alive="24h"
                    )
                    
                    # Inicializamos la cola y el hilo de TTS en paralelo
                    sentence_queue = queue.Queue()
                    tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,))
                    tts_thread.start()
                    
                    # 2. Verificar si Ollama solicita ejecutar alguna función
                    if response['message'].get('tool_calls'):
                        print("\n[Jarvis ejecutando herramienta local...]")
                        messages.append(response['message'])
                        
                        for tool_call in response['message']['tool_calls']:
                            fn_name = tool_call['function']['name']
                            fn_args = tool_call['function']['arguments']
                            
                            if fn_name in available_tools:
                                tool_result = available_tools[fn_name](**fn_args)
                                # Añadimos el resultado de la función al historial de la conversación
                                messages.append({
                                    'role': 'tool',
                                    'content': tool_result
                                })
                        
                        # Transmitimos la respuesta final en streaming basándonos en los datos leídos
                        stream_ollama_to_tts(messages, sentence_queue)
                    
                    else:
                        # Si no requirió herramientas, procesamos el texto recibido directamente
                        direct_text = response['message']['content']
                        print(f"[Jarvis]: {direct_text}")
                        sentence_queue.put(direct_text)
                    
                    # Señalizamos el cierre al hilo TTS y esperamos que termine de hablar
                    sentence_queue.put(None)
                    tts_thread.join()
                    
                print("\n=== Volviendo a estado de espera ===")
                
    except KeyboardInterrupt:
        print("\nCerrando Jarvis...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()

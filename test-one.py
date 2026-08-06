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

# Rutas de modelos
# PIPER_EXE = "piper"  # O la ruta absoluta a tu ejecutable de piper: "C:/piper/piper.exe"
PIPER_EXE = "proj-env/Scripts/piper"
PIPER_MODEL = "es_MX-claude-high.onnx" # Modelo de voz en español para Piper
LLM_MODEL = "llama3.2:3b"

kokoro = Kokoro("kokoro/kokoro-v1_0.onnx", "kokoro/voices-v1_0.bin")

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

# --- BUCLE PRINCIPAL ---
def main():
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=OWW_CHUNK
    )
    
    print("\n=== JARVIS EN ESPERA (Streaming TTS activo) ===")
    
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
                        "Si no entiendes la pregunta o lo que digo no tiene mucho sentido, no tengas miedo a "
                        "preguntarme qué quiero decir o pedirme que reformule mi pregunta o petición. "
                        "Si te digo que no necesito nada más por ahora, simplemente con decirme \"Vale\" o "
                        "algo así, es suficiente. Recuerda, sé conciso y pragmático."
                    )
                    
                    # 1. Creamos la cola y lanzamos el hilo del sintetizador TTS
                    sentence_queue = queue.Queue()
                    tts_thread = threading.Thread(target=tts_worker, args=(sentence_queue,))
                    tts_thread.start()
                    
                    # 2. Solicitamos la respuesta en STREAMING a Ollama
                    stream = ollama.chat(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}
                        ],
                        stream=True
                    )
                    
                    print("[Jarvis]: ", end="", flush=True)
                    text_buffer = ""
                    # Regex para capturar oraciones terminadas en . ! ? o salto de línea
                    sentence_regex = re.compile(r'([^.!?\n]+[.!?\n]+)')
                    
                    for chunk in stream:
                        content = chunk['message']['content']
                        print(content, end="", flush=True)
                        text_buffer += content
                        
                        # Extraemos frases completas del buffer actual
                        matches = sentence_regex.findall(text_buffer)
                        if matches:
                            for match in matches:
                                sentence_queue.put(match)
                                text_buffer = text_buffer[len(match):]
                    
                    # Si quedó texto residual en el buffer sin signo de puntuación final
                    if text_buffer.strip():
                        sentence_queue.put(text_buffer.strip())
                    
                    print()  # Nueva línea tras completar el print en consola
                    
                    # 3. Enviamos la señal 'None' para avisar al hilo TTS que no habrá más frases
                    sentence_queue.put(None)
                    # Esperamos a que el sintetizador termine de hablar antes de volver a escuchar
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

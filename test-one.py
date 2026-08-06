import os
import io
import wave
import subprocess
import numpy as np
import pyaudio
import ollama
from openwakeword.model import Model
from faster_whisper import WhisperModel

# --- CONFIGURACIÓN DE PARÁMETROS ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # Tamaño recomendado para openWakeWord (80ms a 16kHz)

WAKE_WORD_THRESHOLD = 0.5
SILENCE_DURATION_SEC = 2.0  # Segundos de silencio para cortar la grabación
RECORD_TIMEOUT = 10         # Tiempo máximo de escucha tras el wake word

# Rutas de modelos
# PIPER_EXE = "piper"  # O la ruta absoluta a tu ejecutable de piper: "C:/piper/piper.exe"
PIPER_EXE = "proj-env/Scripts/piper"
PIPER_MODEL = "es_MX-claude-high.onnx" # Modelo de voz en español para Piper
LLM_MODEL = "llama3.2:3b"

# --- INICIALIZACIÓN DE COMPONENTES ---
print("Cargando modelo Wake Word...")
oww_model = Model(
  wakeword_models=["hey_jarvis"],
  inference_framework="onnx"
)

print("Cargando Faster-Whisper (STT)...")
# Usa "cuda" y "float16" si tienes GPU NVIDIA, o "cpu" e "int8" si ejecutas en procesador
whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")

audio = pyaudio.PyAudio()

def play_tts(text: str):
    """Sintetiza texto a voz usando Piper y reproduce el audio."""
    if not text.strip():
        return
    
    cmd_piper = [
        PIPER_EXE,
        "--model", PIPER_MODEL,
        "--output-raw"
    ]
    
    # Redirigimos la salida de Piper directamente a PyAudio mediante aplay / ffplay / subprocess
    # Para simplicidad multiplataforma, guardamos un WAV temporal si no usas aplay/paplay:
    wav_output = "temp_response.wav"
    cmd_piper_file = [PIPER_EXE, "--model", PIPER_MODEL, "--output_file", wav_output]
    
    process = subprocess.Popen(cmd_piper_file, stdin=subprocess.PIPE, text=True)
    process.communicate(input=text)
    
    # Reproducir WAV generado
    wf = wave.open(wav_output, 'rb')
    p_out = pyaudio.PyAudio()
    stream_out = p_out.open(
        format=p_out.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    
    data = wf.readframes(1024)
    while len(data) > 0:
        stream_out.write(data)
        data = wf.readframes(1024)
        
    stream_out.stop_stream()
    stream_out.close()
    p_out.terminate()
    wf.close()

def record_user_audio(stream) -> bytes:
    """Graba el audio del usuario tras detectar la palabra de activación."""
    print("\n[Jarvis]: Te escucho...")
    frames = []
    
    # Grabación simple por tiempo fijado (100 frames aprox ~8 seg)
    # En producción se sustituye por Silero VAD para detectar fin de habla
    for _ in range(0, int(RATE / CHUNK * 6)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
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
    segments, _ = whisper_model.transcribe(wav_buffer, language="es")
    text = " ".join([segment.text for segment in segments]).strip()
    return text

# --- BUCLE PRINCIPAL ---
def main():
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("\n=== JARVIS EN ESPERA (Di la palabra clave o activa openWakeWord) ===")
    
    try:
        while True:
            # 1. Escucha continua en busca de Wake Word
            data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Predicción con openWakeWord
            prediction = oww_model.predict(audio_data)
            
            # Comprobación de activación (ej: modelo por defecto como "alexa" o "hey_jarvis")
            scores = list(prediction.values())
            if any(score >= WAKE_WORD_THRESHOLD for score in scores):
                print("\n[Wake Word Detectado!]")
                
                # 2. Captura de la consulta de voz
                user_audio_bytes = record_user_audio(mic_stream)
                
                # 3. Transcripción (STT)
                user_text = transcribe_audio(user_audio_bytes)
                print(f"[Tú]: {user_text}")
                
                if user_text:
                    # 4. Inferencia con Ollama (LLM)
                    system_prompt = (
                        "Eres Jarvis, un asistente local conciso, directo y útil. "
                        "Responde siempre en español. Sé muy breve y conversacional (1 o 2 oraciones máximo) "
                        "evitando formateo Markdown o listas, ya que tu respuesta será leída por un sintetizador de voz."
                    )
                    
                    response = ollama.chat(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}
                        ]
                    )
                    
                    llm_text = response['message']['content']
                    print(f"[Jarvis]: {llm_text}")
                    
                    # 5. Síntesis de Voz (TTS)
                    play_tts(llm_text)
                    
                print("\n=== Volviendo a estado de espera ===")
                
    except KeyboardInterrupt:
        print("\nCerrando Jarvis...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()

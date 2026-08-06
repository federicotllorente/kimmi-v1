# Jarvis — asistente de voz local

Jarvis es un asistente de IA por voz que corre **en tu máquina**. Escucha una wake word, transcribe lo que dices, clasifica la **intención**, razona con un LLM local (Ollama) y solo entonces puede usar **herramientas** acotadas: abrir apps, buscar en el código, mirar logs de Docker, etc.

Todo está pensado para respuestas cortas y habladas (TTS), en español. Las acciones con efecto lateral piden **confirmación** por voz (estilo Siri).

---

## Qué incluye

| Pieza | Rol |
|-------|-----|
| **openWakeWord** | Detecta la wake word (`hey_jarvis`) |
| **Silero VAD** | Detecta cuándo dejas de hablar |
| **faster-whisper** (`medium`) | Speech-to-text (más preciso; GPU recomendada) |
| **Ollama** (`qwen2.5:3b` + `qwen2.5:14b`) | Intent/chat rápido; 14B solo para tools |
| **Kokoro ONNX** | Text-to-speech en español |
| **`tools/`** | Acciones locales sandboxed / con allowlist |

---

## Requisitos

- Windows (el foco de ventanas y parte del portapapeles están orientados a Win32; el resto es portable)
- Python 3.12+ con el entorno del proyecto (`proj-env`)
- [Ollama](https://ollama.com) en marcha, con el modelo:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:14b
```

Opcional pero recomendado:

- **ripgrep** (`rg`) — búsquedas regex rápidas en el código
- **Docker CLI** — listar, logs, stats y compose
- Modelo de embeddings para búsqueda semántica:

```bash
ollama pull nomic-embed-text
```

- `CURSOR_API_KEY` + `pip install cursor-sdk` — si quieres delegar ediciones de código al agente de Cursor
- Micrófono y altavoces/auriculares

Modelos de voz Kokoro esperados en:

- `kokoro/kokoro-v1_0.onnx`
- `kokoro/voices-v1_0.bin`

---

## Cómo ejecutar Jarvis

Desde la raíz del proyecto:

```powershell
# Activar el entorno (PowerShell)
.\proj-env\Scripts\Activate.ps1

# Arrancar
python assistant.py
```

Verás mensajes de carga (wake word, VAD, Whisper). Cuando termine:

```text
=== JARVIS EN ESPERA (intent router + tools + streaming) ===
```

1. Di **“hey Jarvis”** (o la frase del modelo `hey_jarvis`).
2. Jarvis responde al momento **“Sí, Fede?”** (el nombre se configura con `USER_NAME` en `assistant.py`).
3. Formula el pedido; al detectar silencio, transcribe y clasifica intención con el modelo **rápido**.
4. Charla → modelo rápido. Acciones/tools → modelo **14B**, confirmación si hace falta, y respuesta por voz.
5. Vuelve al modo espera (historial corto de conversación).

Para salir: `Ctrl+C`.

### Variables de entorno útiles

| Variable | Uso |
|----------|-----|
| `CURSOR_API_KEY` | Agente Cursor (`run_cursor_agent`) |
| `JARVIS_VSCODE_PATH` / `JARVIS_CURSOR_PATH` / `JARVIS_BROWSER_PATH` / `JARVIS_TERMINAL_PATH` | Rutas a ejecutables si no están en el PATH |
| `JARVIS_BROWSER_PROFILE_WORK` | Directorio de perfil del navegador para `profile=work` |
| `JARVIS_WORKSPACE_ASSISTANT` | Sobrescribe el alias de workspace `assistant` |

En [`assistant.py`](assistant.py) puedes ajustar:

- `ALLOWED_DATA_ROOT` — sandbox de archivos (por defecto el directorio del proyecto)
- `USER_NAME` — nombre en el acuse del wake word
- `ENABLE_INTENT_ROUTER` — clasificar intent antes de responder (`False` por defecto; más rápido)
- `ENABLE_TOOLS` — function calling / acciones locales (`True` por defecto)
- `FAST_MODEL` — chat rápido (`qwen2.5:3b`)
- `LLM_MODEL` — tools / resumen (`qwen2.5:14b`)
- `CURSOR_WORKSPACE` / `CURSOR_MODEL`
- `HISTORY_MAX_TURNS` — memoria corta de conversación
- `WHISPER_MODEL_SIZE` / `WHISPER_BEAM_SIZE` — precisión vs latencia del STT

---

## Cómo funciona (flujo)

```text
Micrófono
   │
   ▼
Wake word ──► acuse TTS ──► VAD + grabación
   │
   ▼
Whisper (texto) ──► filtro STT
   │
   ▼
¿Cierre? (gracias / listo / adiós…) ──sí──► cierre corto TTS ──► espera wake word
   │ no
   ▼
ENABLE_INTENT_ROUTER?
   ├─ False (default) ──► ENABLE_TOOLS? ──sí──► 14B + tools ──► TTS
   │                              └─ no ──► 3B chat ──► TTS
   └─ True ──► clasificar intent ──► chat 3B o subset tools 14B ──► TTS
```

El router vive en [`tools/intent.py`](tools/intent.py). Las tools se registran en [`tools/registry.py`](tools/registry.py); `tools_for_intent` limita qué schemas ve el modelo en cada turno.

**Límites de seguridad:**

- Archivos: solo dentro de `ALLOWED_DATA_ROOT`
- Apps: allowlist (`vscode`, `cursor`, `browser`, `terminal`)
- Docker: sin `exec` / `rm` / `run` arbitrario; compose sin `down -v`
- Docs web: solo hosts allowlisted
- REPL Python: imports restringidos, timeout, sin disco ni red
- Confirmación obligatoria en launch/write/cursor/scaffold y Docker/clipboard con efecto

Cachés locales (embeddings, docs) van a `.jarvis_cache/` (ignorado por git).

---

## Qué puedes pedirle (guía de uso)

Habla en lenguaje natural. Jarvis clasifica la intención y elige la tool; no hace falta decir el nombre exacto, pero estos ejemplos ayudan.

### Archivos y código

- *“Lista lo que hay en la carpeta tools”* → `list_directory`
- *“Lee los archivos de notas”* → `read_directory_files`
- *“Escribe un README en borrador.md con …”* → `write_file` (pide confirmación)
- *“Busca la función launch_app en el código”* → `search_codebase` (`text` / `regex`)
- *“¿En qué archivos se menciona Docker?”* → `search_codebase` (`files_only`)
- *“Busca código relacionado con el portapapeles”* → `search_codebase` (`semantic`, necesita embeddings)

### Entorno de trabajo

- *“Abre Cursor en este proyecto”* / *“Abre VS Code”* → `launch_app` (pide confirmación)
- *“Enfoca el navegador”* → `focus_window`
- *“Abre el navegador con el perfil de trabajo”* → `launch_app` + `profile=work`
- Alias de workspace: por defecto `assistant` apunta al proyecto (configurable por env)

### Portapapeles y cálculos

- *“¿Qué hay en el portapapeles?”* / *“Copia este texto …”* → `manage_clipboard`
- *“Añade al portapapeles …”* / *“Limpia el portapapeles”* / *“Historial del portapapeles”*
- *“Pasa 25 grados Celsius a Fahrenheit”* → `exec_python_repl`  
  (la sesión recuerda variables unos minutos)

### Proyectos nuevos

- *“Crea un proyecto FastAPI llamado api-demo”* → `scaffold_project` (pide confirmación)  
  Plantillas: `python`, `nodejs`, `nextjs`, `micropython`, `fastapi`, `esp32`  
  Opcional: abrirlo en Cursor/VS Code tras crearlo (`open_in`)

### Docker (debug)

- *“Lista los contenedores”* → `list`
- *“Muéstrame los últimos logs de mi-contenedor”* → `logs`
- *“¿Cuánta CPU/memoria usan?”* → `stats`
- *“Inspecciona / reinicia X”* → `inspect` / `restart` / `start` / `top`  
  (`start` / `restart` / compose up-down piden confirmación)
- Compose (fichero dentro del sandbox):  
  *“Levanta el compose”*, *“logs del servicio web”*, *“baja el stack”*

### Documentación

- *“Resúmeme la docs de FastAPI sobre dependency injection”* → `fetch_web_docs`  
- Librerías con alias: `python`, `nextjs`, `react`, `docker`, `micropython`, `tailwind`, `kicad`, …

### Agente Cursor

- *“Refactoriza X con el agente de Cursor”* → `run_cursor_agent` (pide confirmación)  
  Requiere `CURSOR_API_KEY` y el SDK instalado. Por defecto la respuesta va **breve** (mejor para TTS).

---

## Estructura del repo (resumen)

```text
assistant.py          # Bucle: audio + intent + Ollama + confirmación + TTS
tools/
  intent.py           # Clasificador JSON + mapeo intent→tools + confirmación
  registry.py         # Schemas, handlers, system prompt, tools_for_intent
  sandbox.py          # Rutas seguras
  common.py           # Truncado, skips, walk de archivos
  launch_app.py       # Abrir apps allowlisted
  focus_window.py     # Enfocar ventanas (Windows)
  manage_clipboard.py
  exec_python_repl.py
  scaffold_project.py
  search_codebase.py
  manage_docker_container.py
  fetch_web_docs.py
  cursor_agent.py
  …                   # Lectura/escritura/listado de archivos
kokoro/               # Modelos TTS
```

---

## Consejos de uso

1. **Frases claras y una intención por turno** — el VAD corta al silencio; evita monólogos largos.
2. **Nombres concretos** — contenedores Docker, carpetas y apps (“Cursor”, no “el editor”) mejoran el function calling.
3. **Sandbox** — si pides algo fuera de `ALLOWED_DATA_ROOT`, Jarvis te dirá que no puede.
4. **GPU** — Whisper usa `medium` + CUDA/`float16` en `assistant.py` (más preciso que `small`). Si te falta VRAM, baja `WHISPER_MODEL_SIZE` a `small` o usa CPU/`int8`.
5. **Voz** — las respuestas evitan markdown a propósito, para que el TTS suene natural.
6. **Confirmación** — responde “sí” / “dale” o “no” / “cancelar” cuando Jarvis pregunte antes de actuar.
7. **Micrófono** — habla tras el acuse, frases claras y sin ruido de fondo; el STT mejora mucho con buena entrada de audio.

---

## Licencia / estado

Proyecto experimental de asistente local. Las tools y allowlists están pensadas para productividad personal con límites explícitos de seguridad; amplíalas con cuidado.

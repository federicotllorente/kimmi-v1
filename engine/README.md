# Kimmi — asistente de voz local

Kimmi es un asistente de IA por voz que corre **en tu máquina**. Escucha una wake word, transcribe lo que dices, clasifica la **intención**, razona con un LLM local (Ollama) y solo entonces puede usar **herramientas** acotadas: abrir apps, buscar en el código, mirar logs de Docker, etc.

Todo está pensado para respuestas cortas y habladas (TTS), en español. Las acciones con efecto lateral piden **confirmación** por voz (estilo Siri).

## Instalación de dependencias

Instala primero los builds CUDA de PyTorch (no vienen de PyPI): `python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

Nota: Si no tienes GPU NVIDIA, instala la versión para CPU desde PyPI directamente (o adapta el comando según tu GPU): `python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1`

Después: `python -m pip install -r requirements.txt`

---

## Qué incluye

| Pieza | Rol |
|-------|-----|
| **openWakeWord** | Detecta la wake word (`hey_kee_me`) |
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

## Cómo ejecutar Kimmi

Desde la raíz del proyecto:

```powershell
# Activar el entorno (PowerShell)
.\proj-env\Scripts\Activate.ps1

# Arrancar
python assistant.py
```

Verás mensajes de carga (wake word, VAD, Whisper). Cuando termine:

```text
=== KIMMI EN ESPERA (intent router + tools + streaming) ===
```

1. Di **“Hey Kimmi”** (o la frase del modelo `hey_kee_me`), o el pedido en la misma frase (**“Hey Kimmi, qué tal estás?”**): si hay habla justo después de la wake word, Kimmi no dice el acuse y transcribe el comando pegado (pre-roll + continuación).
2. Si solo llamas a Kimmi (silencio después), responde al momento **“Sí, Fede?”** (el nombre se configura con `USER_NAME` en `assistant.py`) y escucha.
3. Formula el pedido; al detectar silencio, transcribe y clasifica intención con el modelo **rápido**.
4. Charla → modelo rápido. Acciones/tools → modelo **14B**, confirmación si hace falta, y respuesta por voz.
5. Tras responder, la sesión sigue activa un rato (sin repetir la wake word) hasta un cierre (“gracias”, etc.) o silencio.
6. Vuelve al modo espera (historial corto de conversación).

Para salir: `Ctrl+C`.

### Variables de entorno útiles

| Variable | Uso |
|----------|-----|
| `CURSOR_API_KEY` | Agente Cursor (`run_cursor_agent`) |
| `KIMMI_VSCODE_PATH` / `KIMMI_CURSOR_PATH` / `KIMMI_BROWSER_PATH` / `KIMMI_TERMINAL_PATH` | Rutas a ejecutables si no están en el PATH |
| `KIMMI_BROWSER_PROFILE_WORK` | Directorio de perfil del navegador para `profile=work` |
| `KIMMI_WORKSPACE_ASSISTANT` | Sobrescribe el alias de workspace `assistant` |
| `KIMMI_MQTT_HOST` / `KIMMI_MQTT_PORT` / `KIMMI_MQTT_PREFIX` | Bus MQTT (`127.0.0.1:1883`, prefijo `kimmi`) — estados del pipeline y notificaciones |
| `KIMMI_MQTT_SMOKE_NOTIFY` | `1` = publicar un toast de prueba al llegar a READY |
| `KIMMI_UI_BUS` | Usado por el **desktop**: `dual` (default) / `mqtt` / `stdout` |
| `KIMMI_SEARXNG_URL` | SearXNG base URL (default `http://127.0.0.1:8080`) |
| `ENABLE_DEMO_JOBS` | `1`/`0` — tool `run_demo_long_job` (default on in code) |
| `KIMMI_RESEARCH_MODEL` | Ollama model for research / competition synthesis |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Cloud synthesis for research / competition |

### MQTT (Fase A)

El desktop puede levantar Mosquitto y suscribirse a:

- `kimmi/pipeline/state` — HUD (`starting` / `idle` / `listening` / `processing` / `responding` / `error`)
- `kimmi/notify` — toast de Windows
- `kimmi/tts/active` — refuerzo de animación
- `kimmi/jobs/done` / `kimmi/jobs/failed` — toast de jobs (dedupe con notify)

El engine publica esos topics desde [`mqtt_bus.py`](mqtt_bus.py). Al cerrar el desktop se desconecta MQTT, se mata el engine y, si Kimmi spawneó Mosquitto, también el broker. Detalles en [`../desktop/README.md`](../desktop/README.md).

### Web search y jobs (Fase B)

1. Instala y arranca [SearXNG](https://docs.searxng.org/) en local (JSON API habilitada), p.ej. en `http://127.0.0.1:8080`.
2. Con `ENABLE_TOOLS = True` (por defecto tras Fase B), puedes pedir: *“busca el clima de mañana”* → tool `web_search`.
3. Sin internet la tool devuelve `{"error":"no_internet"}` y Kimmi lo dice; **no inventa** resultados.
4. Sin SearXNG: `searxng_unavailable`.
5. Demo de job largo: *“corre un job demo de treinta segundos”* → `run_demo_long_job` → toast + “¿te leo el resumen?”.

### Research agent (Fase C)

1. *“Investiga sobre X y haz un resumen”* → Kimmi pregunta **local / Claude / ChatGPT**.
2. `start_research(topic, provider)` encola un job: SearXNG + fetch de páginas públicas + síntesis.
   - `local` → Ollama (`KIMMI_RESEARCH_MODEL` o `qwen2.5:14b`)
   - `claude` → requiere `ANTHROPIC_API_KEY`
   - `chatgpt` → requiere `OPENAI_API_KEY`
3. Retrieval **siempre** pasa por SearXNG (las APIs cloud solo sintetizan).
4. Solo se fetchean URLs públicas (se bloquean IPs privadas/loopback).
5. Al terminar: toast + “¿te leo el resumen?” (mismo loop de jobs de Fase B).

| Variable | Uso |
|----------|-----|
| `KIMMI_RESEARCH_MODEL` | Modelo Ollama para síntesis local |
| `KIMMI_CLAUDE_MODEL` | Override modelo Anthropic |
| `KIMMI_CHATGPT_MODEL` | Override modelo OpenAI |
| `ANTHROPIC_API_KEY` | Claude |
| `OPENAI_API_KEY` | ChatGPT |

### Análisis de competencia (Fase D)

1. *“Analizame la competencia de X”* → Kimmi pregunta **local / Claude / ChatGPT** (mismo contrato que research).
2. `start_competition_analysis(target, provider)` encola un job de varios minutos:
   - hasta 5 queries SearXNG, hasta 10 URLs públicas con `web_fetch`, delay ~0.5s entre requests
   - User-Agent de fetch: `KimmiResearchBot` (mismo que research)
   - síntesis con brief estructurado (jugadores, value props, pricing solo si aparece en fuentes, incertidumbre marcada)
3. Fallos parciales de fetch se anotan en `sources.json` y el job sigue; sin fuentes usables → error claro.
4. Mismas env keys que Fase C (`KIMMI_RESEARCH_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
5. Al terminar: toast + “¿te leo?”; no inventa el informe ni precios antes de tiempo.

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
- Confirmación obligatoria en launch/write/cursor/scaffold/remember_fact y Docker/clipboard con efecto

Cachés locales (embeddings, docs) van a `.kimmi_cache/` (ignorado por git).
La memoria permanente (ChromaDB) vive en `memory/` (también ignorada, salvo `.gitkeep`).

---

## Qué puedes pedirle (guía de uso)

Habla en lenguaje natural. Kimmi clasifica la intención y elige la tool; no hace falta decir el nombre exacto, pero estos ejemplos ayudan.

### Memoria permanente (RAG)

- *“Recuerda que la IP de la Raspberry Pi es 192.168.1.50”* → `remember_fact` (pide confirmación)
- Si mencionas un dato importante sin pedirlo, Kimmi puede preguntarte *“¿Quieres que lo recuerde?”*; si dices que sí, lo guarda
- *“¿Cuál es la IP de la Pi?”* → recupera contexto por similitud (RAG) antes de responder, sin tool
- Requiere `ollama pull nomic-embed-text` y el paquete `chromadb`

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
  remember_fact.py    # Guardar hechos en memoria
  memory.py           # ChromaDB + embeddings Ollama (RAG)
  cursor_agent.py
  …                   # Lectura/escritura/listado de archivos
memory/               # Persistencia ChromaDB (gitignored)
kokoro/               # Modelos TTS
```

---

## Consejos de uso

1. **Frases claras y una intención por turno** — el VAD corta al silencio; evita monólogos largos.
2. **Nombres concretos** — contenedores Docker, carpetas y apps (“Cursor”, no “el editor”) mejoran el function calling.
3. **Sandbox** — si pides algo fuera de `ALLOWED_DATA_ROOT`, Kimmi te dirá que no puede.
4. **GPU** — Whisper usa `medium` + CUDA/`float16` en `assistant.py` (más preciso que `small`). Si te falta VRAM, baja `WHISPER_MODEL_SIZE` a `small` o usa CPU/`int8`.
5. **Voz** — las respuestas evitan markdown a propósito, para que el TTS suene natural.
6. **Confirmación** — responde “sí” / “dale” o “no” / “cancelar” cuando Kimmi pregunte antes de actuar.
7. **Micrófono** — habla tras el acuse, frases claras y sin ruido de fondo; el STT mejora mucho con buena entrada de audio.

---

## Licencia / estado

Proyecto experimental de asistente local. Las tools y allowlists están pensadas para productividad personal con límites explícitos de seguridad; amplíalas con cuidado.

# Kimmi — asistente de voz local

Kimmi es un asistente de IA por voz que corre **en tu máquina**. Escucha una wake word, transcribe lo que dices, razona con un LLM local ([Ollama](https://ollama.com)) y puede usar **herramientas** acotadas: abrir apps, buscar en la web, jobs en background (research / competencia), Docker, etc.

Respuestas cortas y habladas (TTS), en **español o inglés** (cambio por comando o detección automática). Las acciones con efecto lateral piden **confirmación** por voz.

La UI habitual es el [desktop HUD](../desktop/README.md) (Mosquitto + MQTT + motor como subprocess). También puedes arrancar solo el engine desde esta carpeta.

## Instalación de dependencias

Desde `engine/` (con el venv del proyecto):

Instala primero los builds CUDA de PyTorch (no vienen de PyPI):

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Nota: Si no tienes GPU NVIDIA, instala la versión CPU desde PyPI (o adapta el comando según tu GPU):

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

Después:

```powershell
python -m pip install -r requirements.txt
```

---

## Qué incluye

| Pieza | Rol |
|-------|-----|
| **openWakeWord** | Detecta la wake word (`hey_kee_me`) |
| **Silero VAD** | Fin de turno / barge-in |
| **faster-whisper** (`medium`) | Speech-to-text (GPU recomendada) |
| **Ollama** (`qwen2.5:3b` + `qwen2.5:14b`) | Chat rápido; 14B para tools / resumen post-tool |
| **Kokoro ONNX** (o Piper) | Text-to-speech ES/EN |
| **MQTT** (`mqtt_bus.py`) | Estados del pipeline + toasts (desktop) |
| **JobManager** (`jobs.py`) | Tareas largas en background + avisos |
| **`tools/`** | Acciones locales sandboxed / allowlist |

---

## Requisitos

- Windows (foco de ventanas y parte del portapapeles son Win32; el resto es portable)
- Python 3.12+ con el venv del engine (`engine/proj-env`)
- [Ollama](https://ollama.com) en marcha, con:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:14b
```

Opcional pero recomendado:

- **SearXNG** en local (JSON API) — web search, research y competencia
- **Mosquitto** — si usas el desktop con bus MQTT (`KIMMI_UI_BUS=dual` o `mqtt`)
- **ripgrep** (`rg`) — búsquedas regex en el código
- **Docker CLI** — listar, logs, stats y compose
- Embeddings para RAG / búsqueda semántica:

```bash
ollama pull nomic-embed-text
```

- `CURSOR_API_KEY` — agente Cursor (`cursor-sdk` ya está en `requirements.txt`)
- Micrófono y altavoces/auriculares

Modelos de voz Kokoro esperados en:

- `kokoro/kokoro-v1_0.onnx`
- `kokoro/voices-v1_0.bin`

---

## Cómo ejecutar Kimmi

### Solo engine

Desde `engine/`:

```powershell
.\proj-env\Scripts\Activate.ps1
python assistant.py
```

### Desktop (recomendado)

Ver [`../desktop/README.md`](../desktop/README.md): arranca Mosquitto (si hace falta), MQTT y el engine como subprocess.

Cuando el motor está listo:

```text
=== KIMMI READY ===
```

1. Di **“Hey Kimmi”** (modelo `hey_kee_me`), o el pedido en la misma frase (**“Hey Kimmi, qué tal estás?”**): si hay habla justo después de la wake word, Kimmi **no** dice el acuse y transcribe el comando pegado (pre-roll + continuación).
2. Si solo la llamas (silencio después), responde al momento **“Sí, Fede?”** (nombre = `USER_NAME` en `assistant.py`) y escucha.
3. Al silencio, transcribe. Cierres (“gracias”, “no worries”, “no hay problema”…) → ack corto **sin LLM**.
4. Charla clara → modelo **3B** sin tools (heurística `looks_like_chat_only`). Acciones → **14B + tools** (con filler hablado si tarda). Con `ENABLE_INTENT_ROUTER=True` se clasifica intent y se limita el subset de tools.
5. Tras responder, la sesión sigue activa un rato (sin repetir wake word) hasta cierre o silencio; puedes interrumpir el TTS (**barge-in**).
6. Jobs largos (research / competencia / demo): “te aviso” → toast + “¿te leo?” cuando terminan.

Para salir: `Ctrl+C` (o cierra el desktop: corta MQTT, engine y Mosquitto propio).

### Variables de entorno útiles

| Variable | Uso |
|----------|-----|
| `CURSOR_API_KEY` | Agente Cursor (`run_cursor_agent`) |
| `KIMMI_VSCODE_PATH` / `KIMMI_CURSOR_PATH` / `KIMMI_BROWSER_PATH` / `KIMMI_TERMINAL_PATH` | Rutas a ejecutables si no están en el PATH |
| `KIMMI_BROWSER_PROFILE_WORK` | Perfil del navegador para `profile=work` |
| `KIMMI_WORKSPACE_ASSISTANT` | Sobrescribe el alias de workspace `assistant` |
| `KIMMI_MQTT_HOST` / `KIMMI_MQTT_PORT` / `KIMMI_MQTT_PREFIX` | Bus MQTT (`127.0.0.1:1883`, prefijo `kimmi`) |
| `KIMMI_MQTT_SMOKE_NOTIFY` | `1` = toast de prueba al llegar a READY |
| `KIMMI_UI_BUS` | Desktop: `dual` (default) / `mqtt` / `stdout` |
| `KIMMI_MOSQUITTO_BIN` | Ruta a `mosquitto.exe` (desktop) |
| `KIMMI_SEARXNG_URL` | SearXNG (default `http://127.0.0.1:8080`) |
| `ENABLE_DEMO_JOBS` | `1`/`0` — tool `run_demo_long_job` (default on en código) |
| `KIMMI_RESEARCH_MODEL` / `KIMMI_FAST_MODEL` | Modelos Ollama para síntesis / planning de jobs |
| `KIMMI_CLAUDE_MODEL` / `KIMMI_CHATGPT_MODEL` | Override cloud |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Síntesis cloud (research / competencia) |

### MQTT (Fase A)

El desktop puede levantar Mosquitto y suscribirse a:

- `kimmi/pipeline/state` — HUD (`starting` / `idle` / `listening` / `processing` / `responding` / `error`)
- `kimmi/notify` — toast de Windows
- `kimmi/tts/active` — refuerzo de animación
- `kimmi/jobs/done` / `kimmi/jobs/failed` — toast de jobs (dedupe con notify)

El engine publica desde [`mqtt_bus.py`](mqtt_bus.py). Al cerrar el desktop: MQTT → engine → Mosquitto **solo si** Kimmi lo spawneó (si el puerto ya estaba ocupado, no lo mata). Detalles en [`../desktop/README.md`](../desktop/README.md).

### Web search y jobs (Fase B)

1. Instala y arranca [SearXNG](https://docs.searxng.org/) en local (JSON API), p.ej. `http://127.0.0.1:8080`.
2. Con `ENABLE_TOOLS = True`, *“busca el clima de mañana”* → `web_search`.
3. Sin internet → `{"error":"no_internet"}` (Kimmi lo dice; **no inventa** resultados).
4. Sin SearXNG → `searxng_unavailable`.
5. Demo: *“corre un job demo de treinta segundos”* → `run_demo_long_job` → toast + “¿te leo el resumen?”.

### Research agent (Fase C)

1. *“Investiga sobre X y haz un resumen”* → pregunta **local / Claude / ChatGPT**.
2. `start_research(topic, provider)` encola un job: SearXNG + fetch de páginas públicas + síntesis.
   - `local` → Ollama (`KIMMI_RESEARCH_MODEL` o `qwen2.5:14b`)
   - `claude` → `ANTHROPIC_API_KEY`
   - `chatgpt` → `OPENAI_API_KEY`
3. Retrieval **siempre** por SearXNG (cloud solo sintetiza).
4. Solo URLs públicas (bloquea IPs privadas/loopback). User-Agent: `KimmiResearchBot`.
5. Al terminar: toast + “¿te leo el resumen?”.

### Análisis de competencia (Fase D)

1. *“Analizame la competencia de X”* → misma elección de provider.
2. `start_competition_analysis(target, provider)` — job de varios minutos:
   - hasta 5 queries SearXNG, hasta 10 URLs con `web_fetch`, delay ~0.5s
   - brief estructurado; pricing solo si aparece en fuentes; incertidumbre marcada
3. Fallos parciales de fetch → `sources.json` y el job sigue; 0 fuentes → error claro.
4. Mismas env keys que Fase C.
5. Toast + “¿te leo?”; no inventa el informe antes de tiempo.

En [`assistant.py`](assistant.py) puedes ajustar:

- `ALLOWED_DATA_ROOT` — sandbox de archivos
- `USER_NAME` — acuse del wake word
- `ENABLE_INTENT_ROUTER` — clasificar intent (`False` por defecto; con `False` usa heurística chat-only + tools cuando hace falta)
- `ENABLE_TOOLS` — function calling (`True` por defecto)
- `ENABLE_DEMO_JOBS` — job demo
- `FILLER_ENABLED` / `FILLER_AFTER_MS` — relleno hablado si el turno tarda (inmediato en camino tools)
- `FAST_MODEL` / `LLM_MODEL` — `qwen2.5:3b` / `qwen2.5:14b`
- `BARGE_IN_ENABLED` — cortar TTS hablando
- `LANG_AUTO_DETECT_ENABLED` — cambio ES↔EN automático
- `TTS_ENGINE` — `kokoro` o `piper`
- `CURSOR_WORKSPACE` / `CURSOR_MODEL`
- `HISTORY_MAX_TURNS`
- `WHISPER_MODEL_SIZE` / `WHISPER_BEAM_SIZE`

---

## Cómo funciona (flujo)

```text
Micrófono
   │
   ▼
Wake word ──► ¿habla pegada? ──sí──► STT comando (sin acuse)
   │                         └──no──► acuse TTS ──► VAD + grabación
   ▼
Whisper (texto) ──► filtro STT / locale
   │
   ▼
¿Cierre? (gracias / no worries / no hay problema…) ──sí──► ack TTS ──► espera
   │ no
   ▼
ENABLE_INTENT_ROUTER?
   ├─ False (default)
   │     ENABLE_TOOLS?
   │       ├─ no ──► 3B chat (+ filler si tarda) ──► TTS
   │       └─ sí ──► ¿chat-only heurística?
   │              ├─ sí ──► 3B sin tools ──► TTS
   │              └─ no ──► filler tools + 14B + tools ──► TTS
   └─ True ──► clasificar intent ──► chat 3B o subset tools 14B ──► TTS

(en paralelo / entre turnos)
JobManager ──► jobs/done|failed + notify ──► “¿te leo?” ──► read_job_result
```

El router (opcional) vive en [`tools/intent.py`](tools/intent.py). Las tools se registran en [`tools/registry.py`](tools/registry.py). Helpers LLM cloud/local de jobs: [`tools/llm_cloud.py`](tools/llm_cloud.py).

**Límites de seguridad:**

- Archivos: solo dentro de `ALLOWED_DATA_ROOT`
- Apps: allowlist (`vscode`, `cursor`, `browser`, `terminal`)
- Docker: sin `exec` / `rm` / `run` arbitrario; compose sin `down -v`
- Docs web: hosts allowlisted; research/fetch: URLs públicas (anti-SSRF)
- REPL Python: imports restringidos, timeout, sin disco ni red
- Confirmación en launch/write/cursor/scaffold/remember_fact y Docker/clipboard con efecto

Cachés locales (embeddings, docs, jobs, clips TTS) → `.kimmi_cache/` (gitignored).  
Memoria permanente (ChromaDB) → `memory/` (gitignored salvo `.gitkeep`).

---

## Qué puedes pedirle (guía de uso)

Habla en lenguaje natural. No hace falta el nombre exacto de la tool.

### Web, jobs, research y competencia

- *“Busca el clima de mañana”* / *“¿Qué noticias hay de …?”* → `web_search` (SearXNG)
- *“Investiga sobre X y haz un resumen”* → provider → `start_research`
- *“Analizame la competencia de X”* → provider → `start_competition_analysis`
- *“Corre un job demo de treinta segundos”* → `run_demo_long_job`
- *“¿Cómo va el job?”* / *“Léeme el resumen”* → `list_jobs` / `get_job_status` / `read_job_result`

### Memoria permanente (RAG)

- *“Recuerda que la IP de la Raspberry Pi es 192.168.1.50”* → `remember_fact` (pide confirmación)
- Si mencionas un dato importante sin pedirlo, puede preguntar *“¿Quieres que lo recuerde?”*
- *“¿Cuál es la IP de la Pi?”* → RAG antes de responder, sin tool
- Requiere `ollama pull nomic-embed-text` y `chromadb`

### Archivos y código

- *“Lista lo que hay en la carpeta tools”* → `list_directory`
- *“Lee los archivos de notas”* → `read_directory_files`
- *“Escribe un README en borrador.md con …”* → `write_file` (confirmación)
- *“Busca la función launch_app en el código”* → `search_codebase` (`text` / `regex`)
- *“¿En qué archivos se menciona Docker?”* → `search_codebase` (`files_only`)
- *“Busca código relacionado con el portapapeles”* → `search_codebase` (`semantic`)

### Entorno de trabajo

- *“Abre Cursor en este proyecto”* / *“Abre VS Code”* → `launch_app` (confirmación)
- *“Enfoca el navegador”* → `focus_window`
- *“Abre el navegador con el perfil de trabajo”* → `launch_app` + `profile=work`
- Alias de workspace: `assistant` apunta al proyecto (configurable por env)

### Portapapeles y cálculos

- *“¿Qué hay en el portapapeles?”* / *“Copia este texto …”* → `manage_clipboard`
- *“Añade al portapapeles …”* / *“Limpia el portapapeles”* / *“Historial del portapapeles”*
- *“Pasa 25 grados Celsius a Fahrenheit”* → `exec_python_repl`

### Proyectos nuevos

- *“Crea un proyecto FastAPI llamado api-demo”* → `scaffold_project` (confirmación)  
  Plantillas: `python`, `nodejs`, `nextjs`, `micropython`, `fastapi`, `esp32`  
  Opcional: `open_in` Cursor/VS Code

### Docker (debug)

- *“Lista los contenedores”* → `list`
- *“Muéstrame los últimos logs de mi-contenedor”* → `logs`
- *“¿Cuánta CPU/memoria usan?”* → `stats`
- *“Inspecciona / reinicia X”* → `inspect` / `restart` / `start` / `top`  
  (`start` / `restart` / compose up-down piden confirmación)
- Compose (dentro del sandbox): *“Levanta el compose”*, *“logs del servicio web”*, *“baja el stack”*

### Documentación

- *“Resúmeme la docs de FastAPI sobre dependency injection”* → `fetch_web_docs`  
- Alias: `python`, `nextjs`, `react`, `docker`, `micropython`, `tailwind`, `kicad`, …

### Agente Cursor

- *“Refactoriza X con el agente de Cursor”* → `run_cursor_agent` (confirmación)  
  Requiere `CURSOR_API_KEY`. Respuesta breve por defecto (mejor para TTS).

### Idioma

- *“Let’s speak in English”* / *“Habla en español”* → cambio de locale
- Detección automática opcional (`LANG_AUTO_DETECT_ENABLED`) con histéresis

---

## Estructura del repo (resumen)

```text
assistant.py          # Bucle: audio + chat/tools + jobs announce + TTS
audio_out.py          # Sink, barge-in, streaming TTS
clip_cache.py         # Frases fijas pre-sintetizadas (acks, fillers)
locale_config.py      # ES/EN: frases, regex de cierre, STT prompts
mqtt_bus.py           # Publicación MQTT (pipeline / notify / jobs)
jobs.py               # JobManager + cola de avisos
perf.py               # Marcas de latencia por turno
tools/
  intent.py           # Router opcional + confirmación
  registry.py         # Schemas, handlers, system prompt
  connectivity.py     # Probe de red
  web_search.py       # SearXNG
  web_fetch.py        # Fetch de páginas públicas
  llm_cloud.py        # Ollama / Claude / ChatGPT (jobs)
  research_agent.py   # start_research + worker
  competition_agent.py
  remember_fact.py / memory.py
  …                   # apps, files, docker, clipboard, etc.
memory/               # ChromaDB (gitignored)
.kimmi_cache/         # Jobs, clips, docs cache
kokoro/               # Modelos TTS
```

Roadmaps: [`ROADMAP.md`](ROADMAP.md) (voz/core), [`ROADMAP_2.md`](ROADMAP_2.md) (MQTT, web, research, competencia).

---

## Consejos de uso

1. **Frases claras y una intención por turno** — el VAD corta al silencio; evita monólogos largos.
2. **Nombres concretos** — contenedores, carpetas y apps (“Cursor”, no “el editor”) mejoran el function calling.
3. **Sandbox** — fuera de `ALLOWED_DATA_ROOT`, Kimmi dirá que no puede.
4. **GPU** — Whisper `medium` + CUDA/`float16`. Si falta VRAM, baja a `small` o CPU/`int8`.
5. **Voz** — sin markdown a propósito (TTS). Si el 14B tarda, deberías oír un filler (“Déjame comprobarlo” / “Let me check that”).
6. **Confirmación** — “sí” / “dale” o “no” / “cancelar”.
7. **SearXNG** — sin él fallan web search, research y competencia (errores controlados, sin inventar).
8. **Micrófono** — habla tras el acuse (si lo hay); buen audio ayuda mucho al STT.

---

## Licencia / estado

Proyecto experimental de asistente local. Las tools y allowlists están pensadas para productividad personal con límites explícitos de seguridad; amplíalas con cuidado.

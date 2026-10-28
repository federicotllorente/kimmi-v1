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
| **faster-whisper** (`small` + `tiny`) | STT final; idioma + parciales opcionales — ver [Speech-to-text](#speech-to-text-stt) |
| **Ollama** (`qwen2.5:7b`) | Un solo modelo para intent, chat y tools (cabe entero en una GPU de 8 GB) |
| **Kokoro ONNX** (o Piper) | Text-to-speech ES/EN |
| **MQTT** (`mqtt_bus.py`) | Estados del pipeline + toasts (desktop) |
| **JobManager** (`jobs.py`) | Tareas largas en background + avisos |
| **Memoria** (ChromaDB + soft-auto) | Hechos a largo plazo + RAG; ver [Memoria de Kimmi](#memoria-de-kimmi) |
| **`tools/`** | Acciones locales sandboxed / allowlist |

---

## Requisitos

- Windows (foco de ventanas y parte del portapapeles son Win32; el resto es portable)
- Python 3.12+ con el venv del engine (`engine/proj-env`)
- [Ollama](https://ollama.com) en marcha, con:

```bash
ollama pull qwen2.5:7b
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
| `KIMMI_SOFT_MEMORY` | `1` (default) = soft-auto memory en background; `0` la desactiva |
| `KIMMI_PARTIAL_STT` | `1` = STT parcial en rolling (`[You~]:`) mientras hablas (default off); detalle en [Speech-to-text](#speech-to-text-stt) |

### MQTT (Fase A)

El desktop puede levantar Mosquitto y suscribirse a:

- `kimmi/pipeline/state` — HUD (`starting` / `idle` / `listening` / `processing` / `responding` / `error`)
- `kimmi/notify` — toast de Windows
- `kimmi/tts/active` — refuerzo de animación
- `kimmi/jobs/done` / `kimmi/jobs/failed` — toast de jobs (dedupe con notify)
- `kimmi/chat/message` — `{role: user|assistant, text, turn_id}`: burbujas de la ventana de chat
- `kimmi/chat/intent` — `{intent, confidence?, reason, turn_id}`: qué decidió el router
- `kimmi/tools/call` — `{call_id, name, arguments, status: running|done|error|cancelled, result, duration_ms?}`: tarjetas de tools (el mismo `call_id` actualiza la tarjeta)

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
   - `local` → Ollama (`KIMMI_RESEARCH_MODEL` o `qwen2.5:7b`)
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
- `FAST_MODEL` / `LLM_MODEL` — ambos `qwen2.5:7b`
- `BARGE_IN_ENABLED` — cortar TTS hablando
- `LANG_AUTO_DETECT_ENABLED` — cambio ES↔EN automático
- `TTS_ENGINE` — `kokoro` o `piper`
- `CURSOR_WORKSPACE` / `CURSOR_MODEL`
- `HISTORY_MAX_TURNS`
- `WHISPER_MODEL_SIZE` / `WHISPER_BEAM_SIZE`
- `ENABLE_PARTIAL_STT` vía env `KIMMI_PARTIAL_STT` (STT parcial; ver [Speech-to-text](#speech-to-text-stt))

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

## Speech-to-text (STT)

Kimmi no transcribe en streaming de punta a punta: **primero corta el turno con VAD**, y **después** pasa el PCM a Whisper. El texto que llega al LLM es siempre el del STT **final** (`small`). Opcionalmente, con un feature flag, puede emitir hipótesis parciales **mientras** grabas (solo feedback / métricas; no alimentan el LLM).

### Piezas

| Pieza | Rol |
|-------|-----|
| Silero VAD | Decide cuándo empieza y termina la frase (`record_user_audio_vad`) |
| faster-whisper `small` | Transcript autoritativo del turno (`transcribe_audio` / `transcribe_for_turn`) |
| faster-whisper `tiny` | Detección de idioma (~15 ms) y, si el flag está on, STT parcial rolling |
| Locale (`locale_config`) | Idioma fijo para Whisper (`es` / `en`), `initial_prompt` con nombre y apps |

Audio de captura: **16 kHz**, mono, PCM int16. El VAD trabaja en chunks de 512 muestras (~32 ms).

### Camino feliz (sin parciales)

```text
Mic → VAD (Silero)
        │  silencio post-habla / timeout / max duración
        ▼
PCM completo del turno
        │
        ├─ (opcional) tiny.detect_language → ¿cambiar locale?
        ▼
small.transcribe(language=locale activo)
        │  filtro no_speech / avg_logprob / longitud mínima
        ▼
[You]: "…"  →  LLM / tools / cierre
```

1. **Grabación** — `record_user_audio_vad` en [`assistant.py`](assistant.py): descarta silencio inicial, guarda un pre-roll corto (~0.3 s) para no cortar la primera sílaba, acumula frames mientras hay voz y corta tras ~1 s de silencio (`SILENCE_LIMIT_SEC`) o al tope de seguridad (`MAX_RECORD_SEC`).
2. **Wake pegado** — si el comando viene justo tras “Hey Kimmi”, el ring buffer de la wake word se pasa como `seed_audio` y se continúa grabando; a veces no hay acuse TTS.
3. **STT final** — `transcribe_for_turn`:
   - Si `LANG_AUTO_DETECT_ENABLED`, el modelo `tiny` estima idioma (con histéresis / umbrales de duración y probabilidad).
   - Whisper `small` transcribe con `language` del locale (nunca `language=None` en el camino feliz), `beam_size=1`, `temperature=0.0`, sin `vad_filter` (Silero ya recortó).
   - `initial_prompt` sesga nombres propios (p. ej. `USER_NAME`, apps de la allowlist).
4. **Calidad** — segmentos con `no_speech_prob` alto o `avg_logprob` muy bajo se descartan; transcripts demasiado cortos marcan `ok=False` y Kimmi pide repetir.
5. **Confirmaciones** — `_confirm_action` llama a `transcribe_audio` **directo** (sin auto-switch de idioma), para que un “sí/no” dudoso no vuelque la sesión a otro locale.

Constantes útiles en [`assistant.py`](assistant.py): `WHISPER_MODEL_SIZE`, `WHISPER_BEAM_SIZE`, `LANG_*`, `SILENCE_LIMIT_SEC`, `VAD_PRE_ROLL_SEC`.

### STT parcial (feature flag)

Por defecto **apagado**. Actívalo solo para probar feedback en vivo y medir coste:

```powershell
$env:KIMMI_PARTIAL_STT = "1"
python assistant.py
```

Apagar (baseline):

```powershell
Remove-Item Env:KIMMI_PARTIAL_STT -ErrorAction SilentlyContinue
```

También aparece en el banner `Flags: … PARTIAL_STT=True|False` al llegar a `KIMMI READY`.

#### Qué hace

Mientras `record_user_audio_vad` captura, un worker en background ([`partial_stt.py`](partial_stt.py)) reutiliza el Whisper **`tiny`** (el mismo de detección de idioma) bajo un lock:

1. Cada chunk de voz se añade a un buffer PCM compartido (`feed`).
2. Cada ~**800 ms** (`PARTIAL_STT_INTERVAL_MS`), si hay al menos ~**0.8 s** de audio (`PARTIAL_STT_MIN_AUDIO_SEC`) y no hay otra transcripción en curso, corre `tiny.transcribe` sobre una **copia** del buffer.
3. Si el texto cambia, imprime una línea distinta del final:

```text
[You~]: open cursor please
[You~]: open cursor in this project
```

4. Al cortar el VAD, el worker hace **`stop` + join** antes de devolver el PCM. Luego el camino normal (`small`) produce el `[You]:` autoritativo.
5. Resumen de coste: `[partial_stt]: calls=N total_ms=M`, y en la línea `[perf]` del turno: `partial_n` / `partial_ms` (junto a `stt`).

```text
Mic → VAD ──feed──► buffer PCM ──tick──► tiny → [You~]: …
         │                                    (best-effort)
         └─ silencio → stop worker → small → [You]: … → LLM
```

#### Qué no hace

- **No** sustituye el transcript final.
- **No** arranca el LLM con el texto parcial (no hay prefetch).
- **No** publica MQTT ni anima el HUD del desktop (el bridge sigue mirando `[You]:` / `[Kimmi]:`; `[You~]:` es inerte para el estado).
- Si un tick tarda más que el intervalo, el siguiente se **salta** (no encola): el hilo de captura nunca espera a Whisper.

#### A/B y contención de GPU

Compara la misma frase con flag off/on mirando `[perf]`: `stt`, `ttfa`, `llm_ttft`, y con el flag on también `partial_n` / `partial_ms`. Si con el flag on el `stt` final sube mucho, es señal de pelea por la GPU entre `tiny` (parciales) y `small` (final); el diseño actual prioriza aislamiento por lock + join, no un segundo modelo en VRAM.

### Markers de stdout relacionados

| Marker | Significado |
|--------|-------------|
| `[VAD]:` | Inicio/fin de habla o timeout de escucha |
| `[You~]:` | Hipótesis parcial (solo con `KIMMI_PARTIAL_STT=1`) |
| `[You]:` | Transcript final del turno |
| `[partial_stt]:` | Resumen de llamadas / ms del worker parcial |
| `[perf]` | Latencias del turno (`stt`, `partial_*`, …); inerte para el HUD |

Detalle de latencia del pipeline de voz: [`ROADMAP.md`](ROADMAP.md).

---

## Memoria de Kimmi

Kimmi combina **dos memorias** distintas. Confundirlas es la causa habitual de “¿por qué no se acuerda?”.

| Capa | Dónde vive | Qué guarda | Duración |
|------|------------|------------|----------|
| Corto plazo | Lista en RAM (`conversation_history`) | Últimos turnos user/assistant | Solo la sesión activa (~6 pares; `HISTORY_MAX_TURNS`) |
| Largo plazo | ChromaDB en [`memory/`](memory/) | Hechos durables (IP, preferencias, hardware, proyecto…) | Entre reinicios y sesiones |

El corto plazo alimenta el LLM como historial de chat. El largo plazo es un **RAG** clásico: embeber → buscar por similitud → inyectar bullets en el mensaje del usuario (no en el system prompt, para no romper el KV cache de Ollama).

### Stack

- **Almacén:** ChromaDB (`PersistentClient`), colección `kimmi_knowledge`, espacio cosine.
- **Embeddings:** Ollama + `nomic-embed-text` (`keep_alive=24h`). Hay que tener el modelo: `ollama pull nomic-embed-text`.
- **Código:** [`tools/memory.py`](tools/memory.py) (CRUD + RAG), [`tools/soft_memory.py`](tools/soft_memory.py) (auto-guardado), [`tools/remember_fact.py`](tools/remember_fact.py) (tool explícita).
- **Metadata por hecho:** `category`, `created_at`, `source` (`explicit` \| `soft_auto`).

### Lectura (RAG, cada turno)

Antes del LLM, en el camino caliente:

1. Gates baratos: query muy corta, saludos/`rag_skip_re`, o colección vacía → no se paga embedding (`rag≈0`).
2. Si hay datos: embedding de lo que dijo el usuario + top‑k en Chroma (hasta 3).
3. Filtro por distancia cosine (`MAX_DISTANCE ≈ 0.45`) y margen respecto al mejor hit.
4. El bloque se pega al **mensaje de usuario** (“Relevant memories retrieved…”).

Preguntar *“¿Cuál es la IP de la Pi?”* es charla normal: **no** llama tools; responde con lo que el RAG inyectó (si hay hit).

### Escritura: dos caminos

```text
Usuario menciona un hecho
        │
        ├─ “Recuerda que…” / confirma oferta sensible
        │     → intent/tool remember_fact (+ confirmación por voz)
        │     → Chroma (source=explicit)
        │
        └─ Hecho durable de bajo riesgo, sin pedirlo
              → responde normal (TTS)
              → en background: soft-auto → Chroma (source=soft_auto)
```

#### 1) Explícito — `remember_fact`

- El usuario pide guardar (*“recuerda que…”*, *“anota que…”*) o confirma tras una oferta.
- Siempre pide **confirmación por voz** (está en `ALWAYS_CONFIRM_TOOLS`).
- Categorías tipicas del schema: `general`, `network`, `hardware`, `preference`, `project`.

#### 2) Soft-auto (opción A, default on)

Tras el turno (TTS ya sonó), un worker de **1 hilo** (`ThreadPoolExecutor`, no el `JobManager`) puede guardar en silencio:

1. **Gate** (sin LLM): texto corto (menos de ~20 chars), cierre/saludo, secreto detectado, o sin señales de dato estable (IP, “mi/my”, preferencias, hardware, números…) → no hace nada.
2. **Extractor:** `qwen2.5:7b` devuelve JSON con 0–2 hechos; solo parafrasea al **usuario**; categorías allowlist `preference|network|hardware|project|general`.
3. **Filtros:** secretos por regex, groundedness mínima, longitud.
4. **Dedup:** si ya hay un vecino con distancia ≤ `0.25`, no escribe.
5. **Write** en Chroma con `source=soft_auto`. Solo logs (`[Memory]: soft-auto saved|skipped|…`); **nunca habla**.

No corre en cierres, cambio de idioma, ni si ese turno ya guardó con `remember_fact`. Kill-switch: `KIMMI_SOFT_MEMORY=0` (también `false` / `no` / `off`).

### Secretos y privacidad

Soft-auto **no** guarda passwords, PINs, API keys, tokens, etc. (prompt del extractor + regex). Para eso el modelo puede ofrecer recordar y, si el usuario acepta, pasa por `remember_fact` con confirmación.

Hoy **no** hay tool de “olvida X” / listar memorias. En desarrollo: borrar el contenido de `memory/` (salvo `.gitkeep`) resetea la base.

### Requisitos y límites

- Paquete `chromadb` + Ollama con `nomic-embed-text`.
- Embeddings anglocéntricos: el recall cross-lingual (hecho en ES, pregunta en EN) puede degradarse; migrar a un embedder multilingüe implicaría reindexar.
- No indexa PDFs, el repo ni el historial entero: solo hechos que entran por soft-auto o `remember_fact`.
- El research agent (SearXNG) es otro canal; no usa Chroma.

### Ejemplos rápidos

| Dices | Qué pasa |
|-------|----------|
| *“La IP de la Pi es 192.168.1.50”* | Soft-auto puede guardarlo en background; luego *“¿cuál es la IP?”* → RAG |
| *“Recuerda que la IP es …”* | `remember_fact` + confirmación por voz |
| *“Mi password del lab es …”* | Soft-auto no guarda; puede ofrecer y pedir confirmación |
| *“Hola”* / *“gracias”* | Sin extractor ni RAG caro (gates) |

---

## Qué puedes pedirle (guía de uso)

Habla en lenguaje natural. No hace falta el nombre exacto de la tool.

### Web, jobs, research y competencia

- *“Busca el clima de mañana”* / *“¿Qué noticias hay de …?”* → `web_search` (SearXNG)
- *“Investiga sobre X y haz un resumen”* → provider → `start_research`
- *“Analizame la competencia de X”* → provider → `start_competition_analysis`
- *“Corre un job demo de treinta segundos”* → `run_demo_long_job`
- *“¿Cómo va el job?”* / *“Léeme el resumen”* → `list_jobs` / `get_job_status` / `read_job_result`

### Memoria

Detalle completo en [Memoria de Kimmi](#memoria-de-kimmi). Resumen:

- Soft-auto guarda preferencias / IPs / hardware / proyecto sin que lo pidas (`KIMMI_SOFT_MEMORY=0` para apagar).
- *“Recuerda que …”* → `remember_fact` (confirmación).
- *“¿Cuál es la IP de la Pi?”* → RAG, sin tool.
- Secretos no van por soft-auto.

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
partial_stt.py        # Rolling STT parcial (flag KIMMI_PARTIAL_STT)
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
  remember_fact.py / memory.py / soft_memory.py
  …                   # apps, files, docker, clipboard, etc.
memory/               # ChromaDB (gitignored)
.kimmi_cache/         # Jobs, clips, docs cache
kokoro/               # Modelos TTS
```

Roadmaps: [`ROADMAP.md`](ROADMAP.md) (voz/core), [`ROADMAP_2.md`](ROADMAP_2.md) (MQTT, web, research, competencia).

---

## Tests (evals)

Scripts en [`tests/`](tests/) que llaman a los modelos reales y puntúan el comportamiento. Úsalos **antes y después** de tocar un prompt, el schema de una tool, un modelo o Whisper, y compara los números.

| Script | Qué mide |
|---|---|
| `tests/eval_intent.py` | Atajo de charla (sin LLM) + clasificador de intents. Separa frases parecidas a los ejemplos del prompt de frases nuevas, para detectar si el prompt solo "memoriza". |
| `tests/eval_tools.py` | Turno de tools completo (reescritura de follow-ups, llamadas forzadas): tool correcta y argumentos correctos. Ejecuta la calculadora y la traducción de `find_object` para comprobar el resultado final. |
| `tests/eval_stt.py` | WER y latencia de Whisper por tamaño. Usa frases sintetizadas con Kokoro y, si existen, tus grabaciones en `tests/audio/` (`es_algo.wav` + `es_algo.txt`, 16 kHz mono). |

Desde `engine/`:

```bash
proj-env\Scripts\python.exe tests\eval_intent.py
proj-env\Scripts\python.exe tests\eval_tools.py --model qwen2.5:14b
proj-env\Scripts\python.exe tests\eval_stt.py --sizes medium small
```

`eval_intent.py` y `eval_tools.py` devuelven código 1 si algún caso falla.

---

## Consejos de uso

1. **Frases claras y una intención por turno** — el VAD corta al silencio; evita monólogos largos.
2. **Nombres concretos** — contenedores, carpetas y apps (“Cursor”, no “el editor”) mejoran el function calling.
3. **Sandbox** — fuera de `ALLOWED_DATA_ROOT`, Kimmi dirá que no puede.
4. **GPU** — Whisper `small` + CUDA/`float16` (`medium` es algo más preciso pero usa ~1,1 GB más; mídelo con `tests/eval_stt.py`). Si falta VRAM, baja a CPU/`int8`. El `tiny` (idioma / parciales) ya está cargado en el boot.
5. **Voz** — sin markdown a propósito (TTS). Si el 14B tarda, deberías oír un filler (“Déjame comprobarlo” / “Let me check that”).
6. **Confirmación** — “sí” / “dale” o “no” / “cancelar”.
7. **SearXNG** — sin él fallan web search, research y competencia (errores controlados, sin inventar).
8. **Micrófono** — habla tras el acuse (si lo hay); buen audio ayuda mucho al STT.
9. **STT parcial** — `KIMMI_PARTIAL_STT=1` para ver `[You~]:` mientras hablas; el LLM sigue usando solo el `[You]:` final.

---

## Licencia / estado

Proyecto experimental de asistente local. Las tools y allowlists están pensadas para productividad personal con límites explícitos de seguridad; amplíalas con cuidado.

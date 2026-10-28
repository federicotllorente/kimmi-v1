# Kimmi Desktop HUD

Small always-on-top OpenGL window that visualizes the local voice pipeline.

**Active UI path** for this monorepo (the web `backend/` + `frontend/` stack is deprecated).

## Setup

```powershell
cd desktop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Mosquitto (MQTT broker)

The desktop **starts Mosquitto** when the UI bus needs MQTT (`KIMMI_UI_BUS=dual` or `mqtt`), and **stops it on exit** if Kimmi spawned it.

1. Install [Mosquitto](https://mosquitto.org/download/) for Windows.
2. Ensure `mosquitto.exe` is on `PATH`, or set `KIMMI_MOSQUITTO_BIN` to the full path
   (e.g. `C:\Program Files\mosquitto\mosquitto.exe`).
3. Kimmi does **not** download Mosquitto; it only launches the binary.

If port `1883` is already in use, Kimmi reuses that broker and will **not** kill it on exit.

## Run

Requires `engine/proj-env`, Ollama, and models ready (same as running the engine alone). The HUD uses its own `.venv` for PySide6; the voice engine still runs under `engine/proj-env`.

```powershell
cd desktop
.\.venv\Scripts\Activate.ps1
python app.py
```

Startup order: **Mosquitto → MQTT client → engine subprocess**.

A 200×200 window opens and animates the wave from pipeline status (`idle` / `listening` / `processing` / `responding`). A system-tray icon shows Windows toasts for `kimmi/notify` messages.

A second window, **KIMMI · CHAT** ([`chat_window.py`](chat_window.py)), opens to its right: your messages and Kimmi's replies as bubbles, a small `ROUTER` line with the chosen intent, and a card for every tool call (arguments, status, duration and a collapsible result; errors open by themselves). It is fed only over MQTT (`kimmi/chat/*`, `kimmi/tools/call`), so it stays empty with `KIMMI_UI_BUS=stdout`. Closing it keeps Kimmi running: reopen it from the tray menu (**Show chat**).

The window can be resized by dragging any edge or corner: it always stays square ([`aspect_lock.py`](aspect_lock.py)) and the wave, strokes and labels scale with it. Colors and fonts live in [`theme.py`](theme.py) (Orbitron + Rajdhani from `assets/fonts/`, SIL Open Font License); keep them in sync with `robot-vision/ui_theme.py`.

Closing the window (or Quit from the tray menu) stops:

1. MQTT desktop client  
2. Engine subprocess  
3. Mosquitto child (only if Kimmi started it)  
4. Tray icon  

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `KIMMI_UI_BUS` | `dual` | `mqtt` = HUD state only from MQTT; `stdout` = legacy markers; `dual` = MQTT preferred, stdout fallback |
| `KIMMI_MQTT_HOST` | `127.0.0.1` | Broker host |
| `KIMMI_MQTT_PORT` | `1883` | Broker port |
| `KIMMI_MQTT_PREFIX` | `kimmi` | Topic prefix |
| `KIMMI_MOSQUITTO_BIN` | (PATH) | Full path to `mosquitto` if not on PATH |
| `KIMMI_MQTT_SMOKE_NOTIFY` | unset | Set to `1` on the **engine** process to publish a toast when READY |

Jobs (engine Phase B) also publish `kimmi/jobs/done` and `kimmi/jobs/failed`; the desktop shows a toast (deduped with `kimmi/notify` for the same `job_id`).

Example smoke toast (engine publishes; desktop must be running):

```powershell
$env:KIMMI_MQTT_SMOKE_NOTIFY = "1"
# then start desktop (which starts the engine with inherited env)
python app.py
```

Or publish manually while the desktop is open:

```powershell
mosquitto_pub -h 127.0.0.1 -t kimmi/notify -m "{\"title\":\"Kimmi\",\"body\":\"Hello toast\",\"level\":\"info\"}"
```

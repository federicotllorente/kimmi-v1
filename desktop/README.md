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

## Run

Requires `engine/proj-env`, Ollama, and models ready (same as running the engine alone). The HUD uses its own `.venv` for PySide6; the voice engine still runs under `engine/proj-env`.

```powershell
cd desktop
.\.venv\Scripts\Activate.ps1
python app.py
```

A 200×200 window opens, starts `engine/assistant.py`, and animates the wave from pipeline status (`idle` / `listening` / `processing` / `responding`). Closing the window stops the engine.

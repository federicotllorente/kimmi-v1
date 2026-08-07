# Kimmi Bridge (FastAPI)

> **Deprecated.** This web bridge was a prototype. The active UI is the desktop HUD in [`../desktop/`](../desktop/). Keep this folder only as reference.

Thin WebSocket bridge between the local voice engine and the UI.

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Requires `engine/proj-env` and Ollama ready (same as running the engine alone).

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

- Health: `GET http://localhost:8000/health`
- Status WS: `ws://localhost:8000/ws/status`

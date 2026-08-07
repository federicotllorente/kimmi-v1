# Kimmi UI (Vite + React)

> **Deprecated.** This web UI was a prototype. The active UI is the desktop HUD in [`../desktop/`](../desktop/). Keep this folder only as reference.

Single-screen HUD that visualizes the local voice pipeline over WebSocket.

## Setup

```powershell
cd frontend
npm install
```

Optional env (defaults to local backend):

```powershell
# .env
VITE_WS_URL=ws://localhost:8000/ws/status
```

## Run

Start the FastAPI bridge first (`backend/`), then:

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .engine_bridge import EngineBridge
from .ws_hub import WsHub

hub = WsHub()
bridge = EngineBridge(hub)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        await bridge.start()
    except FileNotFoundError as exc:
        # Allow API/WS to come up even if the engine env is missing (dev UX).
        print(f"[backend] Engine not started: {exc}", flush=True)
    yield
    await bridge.stop()


app = FastAPI(title="Kimmi Bridge", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "state": bridge.state}


@app.websocket("/ws/status")
async def ws_status(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        await ws.send_json(bridge.snapshot())
        while True:
            # Block until the client disconnects (or sends a keepalive ping).
            await ws.receive()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)

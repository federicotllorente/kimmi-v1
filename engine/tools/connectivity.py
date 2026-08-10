"""Internet connectivity probes for network tools."""
from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional, Tuple

# Fast TCP targets (no ICMP — often blocked on Windows without admin).
_PROBE_TARGETS: Tuple[Tuple[str, int], ...] = (
    ("1.1.1.1", 443),
    ("8.8.8.8", 53),
)


def check_internet(timeout: float = 2.0) -> Dict[str, Any]:
    """Return {ok: bool, error?: str, message?: str}."""
    last_err = "timeout"
    for host, port in _PROBE_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return {"ok": True}
        except OSError as exc:
            last_err = str(exc) or "connection_failed"
    return {
        "ok": False,
        "error": "no_internet",
        "message": f"No network connectivity ({last_err})",
    }


def require_internet(timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """None if online; otherwise a structured error dict for tool results."""
    result = check_internet(timeout=timeout)
    if result.get("ok"):
        return None
    return {
        "error": result.get("error") or "no_internet",
        "message": result.get("message") or "No network connectivity",
    }


def error_json(error: str, message: str, **extra: Any) -> str:
    """Serialize a tool error as JSON (English; LLM reformulates for speech)."""
    payload: Dict[str, Any] = {"error": error, "message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)

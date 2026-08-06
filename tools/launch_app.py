"""Lanza aplicaciones de la allowlist (VS Code, Cursor, navegador, terminal).

Requiere: stdlib. Rutas opcionales vía JARVIS_VSCODE_PATH, JARVIS_CURSOR_PATH,
JARVIS_BROWSER_PATH, JARVIS_TERMINAL_PATH.
"""
from __future__ import annotations

import os
import subprocess
from urllib.parse import urlparse

from tools.app_allowlist import get_app, list_allowed_app_ids, resolve_launch_command


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def launch_app(app_id: str, path: str = "") -> str:
    """Abre una app permitida. `path` es proyecto/URL según la app."""
    app = get_app(app_id)
    if app is None:
        allowed = ", ".join(list_allowed_app_ids())
        return f"App no permitida. Usa una de: {allowed}."

    cmd = resolve_launch_command(app.id)
    if not cmd:
        return (
            f"No se encontró el ejecutable de {app.display_name}. "
            "Configura la variable de entorno correspondiente o instálalo."
        )

    argv = list(cmd)
    target = (path or "").strip()

    if target:
        if _is_http_url(target):
            if not app.accepts_url:
                return f"{app.display_name} no acepta URLs."
            argv.append(target)
        else:
            if not app.accepts_path:
                return f"{app.display_name} no acepta rutas de proyecto."
            expanded = os.path.abspath(os.path.expanduser(target))
            if not os.path.exists(expanded):
                return f"La ruta '{target}' no existe."
            argv.append(expanded)

    try:
        subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        return f"No se pudo abrir {app.display_name}: {e}"

    extra = f" con '{target}'" if target else ""
    return f"Abriendo {app.display_name}{extra}."

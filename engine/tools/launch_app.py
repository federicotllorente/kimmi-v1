"""Lanza aplicaciones de la allowlist (VS Code, Cursor, navegador, terminal).

Requiere: stdlib. Rutas opcionales vía JARVIS_VSCODE_PATH, JARVIS_CURSOR_PATH,
JARVIS_BROWSER_PATH, JARVIS_TERMINAL_PATH.
Aliases de workspace y perfiles: ver tools.app_allowlist.
"""
from __future__ import annotations

import os
import subprocess
import time
from urllib.parse import urlparse

from tools.app_allowlist import (
    browser_profile_args,
    get_app,
    list_allowed_app_ids,
    resolve_launch_command,
    resolve_workspace_alias,
)


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def launch_app(
    app_id: str,
    path: str = "",
    focus_after: bool = True,
    profile: str = "default",
) -> str:
    """Abre una app permitida. `path` es proyecto/URL/alias según la app."""
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
    resolved_label = target

    if app.id == "browser":
        profile_flags = browser_profile_args(profile)
        if profile and profile.strip().lower() not in ("", "default", "work"):
            return "Perfil de navegador no permitido. Usa 'default' o 'work'."
        argv.extend(profile_flags)

    if target:
        if _is_http_url(target):
            if not app.accepts_url:
                return f"{app.display_name} no acepta URLs."
            argv.append(target)
        else:
            if not app.accepts_path:
                return f"{app.display_name} no acepta rutas de proyecto."
            alias_path = resolve_workspace_alias(target)
            expanded = (
                alias_path
                if alias_path
                else os.path.abspath(os.path.expanduser(target))
            )
            if not os.path.exists(expanded):
                return f"La ruta '{target}' no existe."
            argv.append(expanded)
            resolved_label = expanded

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

    extra = f" con '{resolved_label}'" if target else ""
    msg = f"Abriendo {app.display_name}{extra}."

    if focus_after:
        time.sleep(1.0)
        try:
            from tools.focus_window import focus_window

            focus_msg = focus_window(app_id=app.id)
            if not focus_msg.startswith("No hay ventana"):
                msg = f"{msg} {focus_msg}"
        except Exception:
            pass

    return msg

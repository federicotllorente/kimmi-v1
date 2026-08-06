"""Inspecciona, inicia o reinicia contenedores Docker (sin exec/rm/run).

Requiere: Docker CLI (`docker`) en PATH y daemon activo.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from typing import List

TIMEOUT_SEC = 30
MAX_OUTPUT_CHARS = 2000
ALLOWED_ACTIONS = ("list", "inspect", "start", "restart")
_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def manage_docker_container(action: str, container: str = "") -> str:
    """action: list | inspect | start | restart."""
    action = (action or "").strip().lower()
    if action not in ALLOWED_ACTIONS:
        return f"Acción no válida. Usa: {', '.join(ALLOWED_ACTIONS)}."

    if not shutil.which("docker"):
        return "Docker CLI no está instalado o no está en PATH."

    if action != "list":
        name = (container or "").strip()
        if not name or not _CONTAINER_RE.match(name):
            return "Indica un nombre o id de contenedor válido."

    if action == "list":
        return _run_docker(
            ["ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"]
        )

    if action == "inspect":
        return _run_docker(
            [
                "inspect",
                "--format",
                "Name={{.Name}} Status={{.State.Status}} Image={{.Config.Image}} "
                "Ports={{json .NetworkSettings.Ports}}",
                container.strip(),
            ]
        )

    if action == "start":
        out = _run_docker(["start", container.strip()])
        if out.startswith("Error") or out.startswith("Docker"):
            return out
        return f"Contenedor iniciado: {out.strip() or container.strip()}"

    # restart
    out = _run_docker(["restart", container.strip()])
    if out.startswith("Error") or out.startswith("Docker"):
        return out
    return f"Contenedor reiniciado: {out.strip() or container.strip()}"


def _run_docker(args: List[str]) -> str:
    try:
        completed = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return f"Docker agotó el tiempo ({TIMEOUT_SEC}s)."
    except FileNotFoundError:
        return "Docker CLI no está instalado o no está en PATH."
    except Exception as e:
        return f"Error al ejecutar Docker: {e}"

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        msg = stderr or stdout or f"código {completed.returncode}"
        if "Cannot connect" in msg or "daemon" in msg.lower():
            return "No se pudo conectar al daemon de Docker. ¿Está en ejecución?"
        return _truncate(f"Error Docker: {msg}")

    if not stdout:
        return "Sin salida de Docker."
    return _truncate(stdout)


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "... [truncado]"
    return text

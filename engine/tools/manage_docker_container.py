"""Inspecciona, inicia, reinicia y depura contenedores Docker / Compose.

Acciones: list, inspect, start, restart, logs, stats, top,
compose_ps, compose_logs, compose_up, compose_down, compose_restart.
Sin exec/rm/run ni compose down -v.
Requiere: Docker CLI en PATH. Compose files solo bajo allowed_root.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import List

from tools.common import truncate
from tools.sandbox import resolve_safe_path

TIMEOUT_SEC = 30
COMPOSE_UP_TIMEOUT_SEC = 120
MAX_OUTPUT_CHARS = 2000
MAX_TAIL = 300
DEFAULT_TAIL = 100

ALLOWED_ACTIONS = (
    "list",
    "inspect",
    "start",
    "restart",
    "logs",
    "stats",
    "top",
    "compose_ps",
    "compose_logs",
    "compose_up",
    "compose_down",
    "compose_restart",
)

_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_COMPOSE_NAME_RE = re.compile(
    r"^(compose.*|docker-compose.*)\.(yml|yaml)$",
    re.IGNORECASE,
)


def manage_docker_container(
    action: str,
    container: str = "",
    allowed_root: str = ".",
    compose_file: str = "",
    service: str = "",
    tail: int = DEFAULT_TAIL,
) -> str:
    """Gestiona contenedores/compose para debug diario."""
    action = (action or "").strip().lower()
    if action not in ALLOWED_ACTIONS:
        return f"Acción no válida. Usa: {', '.join(ALLOWED_ACTIONS)}."

    if not shutil.which("docker"):
        return "Docker CLI no está instalado o no está en PATH."

    try:
        tail_n = int(tail) if tail is not None else DEFAULT_TAIL
    except (TypeError, ValueError):
        tail_n = DEFAULT_TAIL
    tail_n = max(1, min(tail_n, MAX_TAIL))

    needs_container = action in ("inspect", "start", "restart", "logs", "top")
    if needs_container:
        name = (container or "").strip()
        if not name or not _CONTAINER_RE.match(name):
            return "Indica un nombre o id de contenedor válido."

    if action.startswith("compose_"):
        compose_path, err = _resolve_compose_file(compose_file, allowed_root)
        if err:
            return err
        return _compose_action(action, compose_path, service.strip(), tail_n)

    if action == "list":
        return _run_docker(
            ["ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"]
        )

    if action == "inspect":
        return _run_docker(
            [
                "inspect",
                "--format",
                "Name={{.Name}} Status={{.State.Status}} Health={{.State.Health.Status}} "
                "Image={{.Config.Image}} Ports={{json .NetworkSettings.Ports}} "
                "StartedAt={{.State.StartedAt}} Error={{.State.Error}}",
                container.strip(),
            ]
        )

    if action == "start":
        out = _run_docker(["start", container.strip()])
        if _is_error(out):
            return out
        return f"Contenedor iniciado: {out.strip() or container.strip()}"

    if action == "restart":
        out = _run_docker(["restart", container.strip()])
        if _is_error(out):
            return out
        return f"Contenedor reiniciado: {out.strip() or container.strip()}"

    if action == "logs":
        return _run_docker(
            [
                "logs",
                "--tail",
                str(tail_n),
                "--timestamps",
                container.strip(),
            ]
        )

    if action == "stats":
        args = ["stats", "--no-stream"]
        if container and container.strip():
            if not _CONTAINER_RE.match(container.strip()):
                return "Indica un nombre o id de contenedor válido."
            args.append(container.strip())
        return _run_docker(args)

    if action == "top":
        return _run_docker(["top", container.strip()])

    return f"Acción no implementada: {action}"


def _resolve_compose_file(compose_file: str, allowed_root: str) -> tuple:
    path = (compose_file or "").strip() or "docker-compose.yml"
    safe, error = resolve_safe_path(path, allowed_root)
    if error:
        return None, error
    if not os.path.isfile(safe):
        # Intentar compose.yaml
        if not compose_file:
            alt, alt_err = resolve_safe_path("compose.yaml", allowed_root)
            if not alt_err and alt and os.path.isfile(alt):
                safe = alt
            else:
                return None, f"No existe el compose file '{path}'."
        else:
            return None, f"No existe el compose file '{path}'."

    base = os.path.basename(safe)
    if not _COMPOSE_NAME_RE.match(base):
        return None, (
            "Nombre de compose no permitido. Usa compose*.yml/yaml "
            "o docker-compose*.yml/yaml."
        )
    return safe, None


def _compose_action(
    action: str,
    compose_path: str,
    service: str,
    tail_n: int,
) -> str:
    base = ["compose", "-f", compose_path]

    if action == "compose_ps":
        return _run_docker(base + ["ps"], timeout=TIMEOUT_SEC)

    if action == "compose_logs":
        cmd = base + ["logs", "--tail", str(tail_n), "--no-color"]
        if service:
            if not _CONTAINER_RE.match(service):
                return "Nombre de servicio inválido."
            cmd.append(service)
        return _run_docker(cmd, timeout=TIMEOUT_SEC)

    if action == "compose_up":
        return _run_docker(base + ["up", "-d"], timeout=COMPOSE_UP_TIMEOUT_SEC)

    if action == "compose_down":
        # Explicitamente sin -v
        return _run_docker(base + ["down"], timeout=TIMEOUT_SEC)

    if action == "compose_restart":
        cmd = base + ["restart"]
        if service:
            if not _CONTAINER_RE.match(service):
                return "Nombre de servicio inválido."
            cmd.append(service)
        return _run_docker(cmd, timeout=TIMEOUT_SEC)

    return f"Acción compose desconocida: {action}"


def _is_error(out: str) -> bool:
    return out.startswith("Error") or out.startswith("Docker") or out.startswith(
        "No se pudo"
    )


def _run_docker(args: List[str], timeout: int = TIMEOUT_SEC) -> str:
    try:
        completed = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Docker agotó el tiempo ({timeout}s)."
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
        return truncate(f"Error Docker: {msg}", MAX_OUTPUT_CHARS)

    if not stdout and stderr:
        # compose often writes progress to stderr
        return truncate(stderr, MAX_OUTPUT_CHARS)
    if not stdout:
        return "Sin salida de Docker."
    return truncate(stdout, MAX_OUTPUT_CHARS)

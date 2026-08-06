"""Registro central de tools: schemas Ollama + handlers."""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from tools.app_allowlist import list_allowed_app_ids
from tools.cursor_agent import run_cursor_agent
from tools.exec_python_repl import exec_python_repl
from tools.fetch_web_docs import fetch_web_docs
from tools.focus_window import focus_window
from tools.launch_app import launch_app
from tools.list_files import list_directory
from tools.manage_clipboard import manage_clipboard
from tools.manage_docker_container import manage_docker_container
from tools.read_files import read_directory_files
from tools.scaffold_project import TEMPLATES, scaffold_project
from tools.search_codebase import search_codebase
from tools.write_file import write_file


def build_tools(
    allowed_root: str,
    llm_model: str,
    cursor_workspace: str,
    cursor_model: str,
    cursor_api_key: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Callable[..., str]]]:
    """Construye (tools_schema, available_tools) para Ollama function calling."""
    apps = ", ".join(list_allowed_app_ids())
    templates = ", ".join(TEMPLATES)

    def _read(directory_path: str) -> str:
        return read_directory_files(directory_path, allowed_root=allowed_root)

    def _list(path: str, recursive: bool = False) -> str:
        return list_directory(path, allowed_root=allowed_root, recursive=recursive)

    def _write(file_path: str, content: str) -> str:
        return write_file(file_path, content, allowed_root=allowed_root)

    def _cursor(prompt: str, brief: bool = True) -> str:
        return run_cursor_agent(
            prompt,
            workspace=cursor_workspace,
            model=cursor_model,
            api_key=cursor_api_key or os.environ.get("CURSOR_API_KEY"),
            brief=brief,
        )

    def _scaffold(
        template: str,
        name: str,
        parent_path: str = ".",
        open_in: str = "",
    ) -> str:
        return scaffold_project(
            template,
            name,
            allowed_root=allowed_root,
            parent_path=parent_path,
            open_in=open_in,
        )

    def _search(
        query: str,
        mode: str = "regex",
        path: str = ".",
        glob: str = "",
    ) -> str:
        return search_codebase(
            query,
            allowed_root=allowed_root,
            mode=mode,
            path=path,
            glob=glob,
        )

    def _docker(
        action: str,
        container: str = "",
        compose_file: str = "",
        service: str = "",
        tail: int = 100,
    ) -> str:
        return manage_docker_container(
            action,
            container=container,
            allowed_root=allowed_root,
            compose_file=compose_file,
            service=service,
            tail=tail,
        )

    def _docs(
        url: str = "",
        library: str = "",
        topic: str = "",
        mode: str = "summary",
    ) -> str:
        return fetch_web_docs(
            url=url,
            library=library,
            topic=topic,
            llm_model=llm_model,
            mode=mode,
            allowed_root=allowed_root,
        )

    def _launch(
        app_id: str,
        path: str = "",
        focus_after: bool = True,
        profile: str = "default",
    ) -> str:
        return launch_app(
            app_id,
            path=path,
            focus_after=focus_after,
            profile=profile,
        )

    def _repl(code: str, reset_session: bool = False) -> str:
        return exec_python_repl(code, reset_session=reset_session)

    schema: List[dict] = [
        {
            "type": "function",
            "function": {
                "name": "read_directory_files",
                "description": (
                    "Lee el contenido de los archivos de texto dentro del directorio permitido "
                    f"({allowed_root}) o de una subcarpeta suya."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": 'Ruta relativa (ej: ".", "notas").',
                        },
                    },
                    "required": ["directory_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": f"Lista archivos y carpetas dentro de {allowed_root}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Ruta relativa."},
                        "recursive": {
                            "type": "boolean",
                            "description": "Si true, lista el árbol completo.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Escribe o crea un archivo de texto solo dentro del "
                    f"directorio permitido ({allowed_root})."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_cursor_agent",
                "description": (
                    "Delega una tarea de código al agente de Cursor (SDK local) sobre "
                    f"{cursor_workspace}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "brief": {
                            "type": "boolean",
                            "description": "Si true (default), resume corto para voz.",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "launch_app",
                "description": (
                    f"Abre una app allowlisted ({apps}). "
                    "path puede ser carpeta, URL o alias de workspace (ej: assistant)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_id": {"type": "string"},
                        "path": {"type": "string"},
                        "focus_after": {"type": "boolean"},
                        "profile": {
                            "type": "string",
                            "description": "Para browser: default o work.",
                        },
                    },
                    "required": ["app_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "focus_window",
                "description": f"Enfoca una ventana abierta allowlisted ({apps}).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_id": {"type": "string"},
                        "title_substring": {"type": "string"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "manage_clipboard",
                "description": (
                    "Portapapeles: read, write, append, clear o history. Solo I/O de texto."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_python_repl",
                "description": (
                    "Python restringido para cálculos/conversiones (sesión con TTL). "
                    "Sin disco ni red. Ej: (c*9/5)+32 para °C a °F."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "reset_session": {"type": "boolean"},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scaffold_project",
                "description": (
                    f"Crea scaffold ({templates}) solo dentro de {allowed_root}. "
                    "open_in: cursor o vscode."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string"},
                        "name": {"type": "string"},
                        "parent_path": {"type": "string"},
                        "open_in": {"type": "string"},
                    },
                    "required": ["template", "name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": (
                    "Busca en el código: mode text, regex, semantic o files_only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {"type": "string"},
                        "path": {"type": "string"},
                        "glob": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "manage_docker_container",
                "description": (
                    "Docker debug: list, inspect, start, restart, logs, stats, top, "
                    "compose_ps, compose_logs, compose_up, compose_down, compose_restart. "
                    "Compose solo bajo el sandbox."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "container": {"type": "string"},
                        "compose_file": {"type": "string"},
                        "service": {"type": "string"},
                        "tail": {"type": "integer"},
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_web_docs",
                "description": (
                    "Docs web allowlisted. mode summary (default) o extract. "
                    "Pasa url o library+topic (python, nextjs, micropython, tailwind, kicad, ...)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "library": {"type": "string"},
                        "topic": {"type": "string"},
                        "mode": {"type": "string"},
                    },
                    "required": [],
                },
            },
        },
    ]

    handlers: Dict[str, Callable[..., str]] = {
        "read_directory_files": _read,
        "list_directory": _list,
        "write_file": _write,
        "run_cursor_agent": _cursor,
        "launch_app": _launch,
        "focus_window": focus_window,
        "manage_clipboard": manage_clipboard,
        "exec_python_repl": _repl,
        "scaffold_project": _scaffold,
        "search_codebase": _search,
        "manage_docker_container": _docker,
        "fetch_web_docs": _docs,
    }

    return schema, handlers


def build_system_prompt(allowed_root: str) -> str:
    apps = ", ".join(list_allowed_app_ids())
    return (
        "Eres Jarvis, un asistente local conciso, directo y útil. "
        "Responde siempre en español. Usa frases cortas y conversacionales (1 o 2 oraciones máximo). "
        "Evita completamente el uso de caracteres Markdown (sin asteriscos, guiones o numerales) o "
        "listas, ya que tu respuesta será leída por un sintetizador de voz. "
        "Si no entiendes la pregunta, pide que reformule. "
        f"Archivos solo dentro de {allowed_root}. "
        "Puedes delegar código a run_cursor_agent. "
        f"Apps allowlisted: {apps} (launch_app/focus_window; path puede ser alias assistant). "
        "Portapapeles: read, write, append, clear, history. "
        "Python restringido con sesión (exec_python_repl). "
        "Scaffold de proyectos y búsqueda con search_codebase "
        "(mode text, regex, semantic o files_only). "
        "Docker debug: logs, stats, top y compose_* sobre ficheros del sandbox. "
        "Documentación web allowlisted con fetch_web_docs (summary o extract)."
    )

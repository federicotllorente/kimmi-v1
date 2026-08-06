"""Registro central de tools: schemas Ollama + handlers."""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from tools.app_allowlist import list_allowed_app_ids
from tools.cursor_agent import run_cursor_agent
from tools.exec_python_repl import exec_python_repl
from tools.fetch_web_docs import fetch_web_docs
from tools.focus_window import focus_window
from tools.intent import tool_names_for_intent
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
                    "SOLO si el usuario pide explícitamente leer el contenido de archivos "
                    f"en el sandbox ({allowed_root}). NO usar para conversación general ni "
                    "para solo listar nombres."
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
                "description": (
                    "SOLO si el usuario pide listar archivos o carpetas "
                    f"dentro de {allowed_root}. NO usar para charla ni para leer contenido."
                ),
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
                    "SOLO si el usuario pide explícitamente crear o sobrescribir un archivo "
                    f"de texto en {allowed_root}. NO usar para conversación ni borradores no pedidos."
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
                    "SOLO si el usuario pide explícitamente delegar una tarea de código al "
                    f"agente de Cursor sobre {cursor_workspace}. NO para preguntas generales "
                    "ni para editar sin haberlo pedido."
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
                    f"SOLO si el usuario pide abrir o lanzar una app allowlisted ({apps}). "
                    "path puede ser carpeta, URL o alias (ej: assistant). "
                    "NO usar si solo pregunta sobre apps o para enfocar una ya abierta."
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
                "description": (
                    f"SOLO si el usuario pide enfocar o traer al frente una ventana "
                    f"allowlisted ya abierta ({apps}). NO para abrir apps nuevas."
                ),
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
                    "SOLO si el usuario pide leer, escribir, añadir, limpiar o ver el "
                    "historial del portapapeles (read, write, append, clear, history). "
                    "NO usar para conversación general."
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
                    "SOLO si el usuario pide un cálculo, conversión o evaluación numérica. "
                    "Python restringido (sesión con TTL), sin disco ni red. "
                    "NO usar para preguntas que no sean cálculo."
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
                    f"SOLO si el usuario pide crear un proyecto nuevo ({templates}) "
                    f"dentro de {allowed_root}. open_in: cursor o vscode. "
                    "NO usar sin petición explícita de crear scaffold."
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
                    "SOLO si el usuario pide buscar en el código "
                    "(mode text, regex, semantic o files_only). "
                    "NO usar para charla ni para listar carpetas sin búsqueda."
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
                    "SOLO si el usuario pide una acción Docker concreta: list, inspect, "
                    "start, restart, logs, stats, top, compose_ps, compose_logs, "
                    "compose_up, compose_down, compose_restart. Compose solo bajo el sandbox. "
                    "NO usar para conversación sobre Docker sin acción."
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
                    "SOLO si el usuario pide consultar documentación web allowlisted "
                    "(mode summary o extract). Pasa url o library+topic. "
                    "NO usar para preguntas generales sin pedir docs."
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


def tools_for_intent(intent: str, tools_schema: List[dict]) -> List[dict]:
    """Filtra el schema completo al subset del intent clasificado."""
    allowed = set(tool_names_for_intent(intent))
    if not allowed:
        return []
    filtered: List[dict] = []
    for entry in tools_schema:
        name = entry.get("function", {}).get("name")
        if name in allowed:
            filtered.append(entry)
    return filtered


def build_system_prompt(allowed_root: str) -> str:
    apps = ", ".join(list_allowed_app_ids())
    return (
        "Eres Jarvis, un asistente local de voz: conciso, directo y útil. "
        "Responde siempre en español con 1 o 2 oraciones conversacionales. "
        "Sin markdown, asteriscos, guiones de lista ni numerales: tu respuesta se lee en voz alta. "
        "CIERRE: si el usuario agradece, dice que está listo, perfecto, vale, adiós o similar, "
        "responde con un cierre breve (de nada, cuando quieras) y NO preguntes si necesita algo más. "
        "Nunca termines con preguntas del tipo necesitas algo más o en qué más te ayudo. "
        "DISCIPLINA DE HERRAMIENTAS: "
        "Solo llama a una tool si el usuario pide una acción concreta "
        "(abrir, listar, escribir, buscar en código, Docker, portapapeles, cálculo, docs). "
        "Saludos, charla, opiniones o preguntas generales: responde SIN tools. "
        "Si hay duda entre tools o faltan datos: pregunta, no ejecutes. "
        "Prefiere cero tools a una incorrecta. Nunca inventes resultados de tools. "
        f"Archivos solo dentro de {allowed_root}. "
        f"Apps allowlisted: {apps}. "
        "Usa como máximo una tool por turno."
    )

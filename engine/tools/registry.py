"""Registro central de tools: schemas Ollama + handlers."""
from __future__ import annotations

import json
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
from tools.remember_fact import remember_fact
from tools.scaffold_project import TEMPLATES, scaffold_project
from tools.search_codebase import search_codebase
from tools.web_search import web_search
from tools.competition_agent import start_competition_analysis
from tools.research_agent import start_research
from tools.write_file import write_file


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def build_tools(
    allowed_root: str,
    llm_model: str,
    cursor_workspace: str,
    cursor_model: str,
    cursor_api_key: Optional[str] = None,
    enable_demo_jobs: Optional[bool] = None,
) -> Tuple[List[dict], Dict[str, Callable[..., str]]]:
    """Construye (tools_schema, available_tools) para Ollama function calling."""
    from jobs import demo_long_job_worker, get_job_manager

    apps = ", ".join(list_allowed_app_ids())
    templates = ", ".join(TEMPLATES)
    if enable_demo_jobs is None:
        enable_demo_jobs = _env_flag("ENABLE_DEMO_JOBS", True)

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

    def _remember(fact: str, category: str = "general") -> str:
        return remember_fact(fact, category=category)

    def _web_search(
        query: str,
        max_results: int = 5,
        language: str = "",
    ) -> str:
        return web_search(
            query,
            max_results=max_results,
            language=language,
            allowed_root=allowed_root,
        )

    def _demo_job(seconds: int = 30, title: str = "demo job") -> str:
        mgr = get_job_manager()
        job_title = (title or "demo job").strip() or "demo job"
        try:
            secs = int(seconds)
        except (TypeError, ValueError):
            secs = 30

        def _fn(ctx):
            return demo_long_job_worker(ctx, seconds=secs)

        job_id = mgr.start("demo", job_title, _fn, seconds=secs)
        return json.dumps(
            {
                "job_id": job_id,
                "status": "queued",
                "kind": "demo",
                "title": job_title,
                "seconds": secs,
                "message": (
                    "Background demo job started. Tell the user you will notify "
                    "them when it finishes; do not invent the final summary yet."
                ),
            },
            ensure_ascii=False,
        )

    def _job_status(job_id: str) -> str:
        mgr = get_job_manager()
        info = mgr.get(job_id)
        if info is None:
            # Fall back to disk read
            disk = mgr.read_result(job_id)
            if disk.get("error") == "not_found":
                return json.dumps(disk, ensure_ascii=False)
            return json.dumps(
                {
                    "job_id": job_id,
                    "status": disk.get("status"),
                    "title": disk.get("title"),
                },
                ensure_ascii=False,
            )
        return json.dumps(info, ensure_ascii=False)

    def _job_result(job_id: str) -> str:
        mgr = get_job_manager()
        return json.dumps(mgr.read_result(job_id), ensure_ascii=False)

    def _list_jobs(limit: int = 8) -> str:
        mgr = get_job_manager()
        return json.dumps(
            {
                "active": mgr.list_active(),
                "recent": mgr.list_recent(limit=limit),
            },
            ensure_ascii=False,
        )

    def _start_research(topic: str, provider: str = "") -> str:
        return start_research(
            topic=topic,
            provider=provider,
            allowed_root=allowed_root,
        )

    def _start_competition(target: str, provider: str = "") -> str:
        return start_competition_analysis(
            target=target,
            provider=provider,
            allowed_root=allowed_root,
        )

    schema: List[dict] = [
        {
            "type": "function",
            "function": {
                "name": "read_directory_files",
                "description": (
                    "ONLY if the user explicitly asks to read the contents of files "
                    f"in the sandbox ({allowed_root}). Do NOT use for general conversation "
                    "or to just list names."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": 'Relative path (e.g. ".", "notes").',
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
                    "ONLY if the user asks to list files or folders "
                    f"inside {allowed_root}. Do NOT use for small talk or to read contents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path."},
                        "recursive": {
                            "type": "boolean",
                            "description": "If true, list the whole tree.",
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
                    "ONLY if the user explicitly asks to create or overwrite a text file "
                    f"in {allowed_root}. Do NOT use for conversation or unrequested drafts."
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
                    "ONLY if the user explicitly asks to delegate a coding task to the "
                    f"Cursor agent over {cursor_workspace}. NOT for general questions "
                    "or for editing without being asked."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "brief": {
                            "type": "boolean",
                            "description": "If true (default), summarize short for speech.",
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
                    f"ONLY if the user asks to open or launch an allowlisted app ({apps}). "
                    "path can be a folder, a URL or an alias (e.g. assistant). "
                    "Do NOT use if they merely ask about apps, or to focus an open one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app_id": {"type": "string"},
                        "path": {"type": "string"},
                        "focus_after": {"type": "boolean"},
                        "profile": {
                            "type": "string",
                            "description": "For browser: default or work.",
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
                    "ONLY if the user asks to focus or bring forward an already open "
                    f"allowlisted window ({apps}). NOT for opening new apps."
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
                    "ONLY if the user asks to read, write, append to, clear or view the "
                    "clipboard history (read, write, append, clear, history). "
                    "Do NOT use for general conversation."
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
                    "ONLY if the user asks for a calculation, conversion or numeric evaluation. "
                    "Restricted Python (session with a TTL), no disk and no network. "
                    "Do NOT use for questions that are not calculations."
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
                    f"ONLY if the user asks to create a new project ({templates}) "
                    f"inside {allowed_root}. open_in: cursor or vscode. "
                    "Do NOT use without an explicit request to scaffold."
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
                    "ONLY if the user asks to search the code "
                    "(mode text, regex, semantic or files_only). "
                    "Do NOT use for small talk or to list folders without searching."
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
                    "ONLY if the user asks for a concrete Docker action: list, inspect, "
                    "start, restart, logs, stats, top, compose_ps, compose_logs, "
                    "compose_up, compose_down, compose_restart. Compose only under the sandbox. "
                    "Do NOT use for conversation about Docker with no action."
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
                    "ONLY if the user asks to look up allowlisted web documentation "
                    "(mode summary or extract). Pass url, or library plus topic. "
                    "Do NOT use for general questions that do not ask for docs."
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
        {
            "type": "function",
            "function": {
                "name": "remember_fact",
                "description": (
                    "Store a personal detail, preference, device IP, pinout or important "
                    "technical note in long-term memory. "
                    "Use it when the user explicitly asks you to remember something "
                    "(remember that..., note that..., save to memory...). "
                    "Do NOT use it for normal small talk or to invent facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": (
                                "The clear, concrete piece of information to remember "
                                '(e.g. "The Raspberry Pi IP is 192.168.1.50").'
                            ),
                        },
                        "category": {
                            "type": "string",
                            "description": (
                                "Optional category: general, network, hardware, preference, project."
                            ),
                        },
                    },
                    "required": ["fact"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the live web via SearXNG for current facts "
                    "(weather, news, prices, who/what is X). "
                    "Use when the answer needs up-to-date information from the internet. "
                    "Do NOT invent results if the tool returns an error."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in the user's language or English.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of results (1-10, default 5).",
                        },
                        "language": {
                            "type": "string",
                            "description": "Optional SearXNG language code (e.g. en, es).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_job_status",
                "description": (
                    "Check the status of a background job by job_id "
                    "(queued, running, done, failed)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                    },
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_job_result",
                "description": (
                    "Read the summary/result of a finished background job. "
                    "Use when the user asks to hear the report or after they confirm "
                    "they want it read aloud."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                    },
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_jobs",
                "description": (
                    "List active and recent background jobs (research, competition, demo, etc.). "
                    "Use when the user asks how a research or competition job is going."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max recent jobs to include (default 8).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_research",
                "description": (
                    "Start a BACKGROUND research job on a topic (SearXNG retrieval + "
                    "summary). REQUIRED: ask the user first whether to use local "
                    "(Ollama), Claude, or ChatGPT, then pass provider. "
                    "Returns job_id immediately; tell them you will notify them. "
                    "Do NOT invent the final report."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "What to research.",
                        },
                        "provider": {
                            "type": "string",
                            "description": "local | claude | chatgpt (required after user chooses).",
                        },
                    },
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_competition_analysis",
                "description": (
                    "Start a BACKGROUND competitive-analysis job for a company or product "
                    "(SearXNG + public pages + structured brief). REQUIRED: ask the user "
                    "first whether to use local (Ollama), Claude, or ChatGPT, then pass "
                    "provider. Returns job_id immediately; tell them you will notify them. "
                    "Do NOT invent the final report or pricing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Company or product to analyze.",
                        },
                        "provider": {
                            "type": "string",
                            "description": "local | claude | chatgpt (required after user chooses).",
                        },
                    },
                    "required": ["target"],
                },
            },
        },
    ]

    if enable_demo_jobs:
        schema.append(
            {
                "type": "function",
                "function": {
                    "name": "run_demo_long_job",
                    "description": (
                        "Start a DEMO background job that sleeps for N seconds and then "
                        "produces a placeholder summary. Use only when the user asks to "
                        "test a long job / demo background task. Returns immediately with "
                        "job_id; tell them you will notify them when it finishes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seconds": {
                                "type": "integer",
                                "description": "Duration in seconds (5-120, default 30).",
                            },
                            "title": {
                                "type": "string",
                                "description": "Short title for the toast/announcement.",
                            },
                        },
                        "required": [],
                    },
                },
            }
        )

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
        "remember_fact": _remember,
        "web_search": _web_search,
        "get_job_status": _job_status,
        "read_job_result": _job_result,
        "list_jobs": _list_jobs,
        "start_research": _start_research,
        "start_competition_analysis": _start_competition,
    }
    if enable_demo_jobs:
        handlers["run_demo_long_job"] = _demo_job

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


def format_memory_block(memory_context: str) -> str:
    """Bloque de recuerdos que se adjunta al mensaje del usuario.

    Va en el turno del usuario y no en el system para que el prefijo
    [system] + history quede byte-idéntico entre turnos y Ollama pueda
    reutilizar el cache de KV en vez de re-prefillear todo el prompt.
    """
    if not memory_context or not memory_context.strip():
        return ""
    return (
        "Relevant memories retrieved from your long-term memory "
        "(use them only if they help you answer; do not invent others):\n"
        f"{memory_context.strip()}"
    )


def build_system_prompt(allowed_root: str, language_name: str) -> str:
    """One prompt in English, with the reply language as the only variable.

    Kept single-language on purpose: two parallel prompts drift apart, and the
    closing and tool-discipline rules are exactly the ones that must not.
    """
    apps = ", ".join(list_allowed_app_ids())

    return (
        "You are Kimmi, a local voice assistant: concise, direct and useful. "
        f"Always reply in {language_name}, in one or two conversational sentences. "
        "No markdown, asterisks, bullet dashes or numbering: your answer is read out loud. "
        "SOUND HUMAN: use contractions. Never say 'Certainly!', 'Sure thing!', 'As an AI' "
        "or 'How can I assist you today?'. "
        "Write numbers as words ('twenty-three', not '23'), because they are spoken. "
        # Small models leak CJK digits where a numeral would go once the rule above
        # pushes them away from digits. Measured: 1 in 12 replies without this line.
        "Use only the writing system of that language: never mix in another script. "
        "CLOSING: if the user thanks you, says they are fine or done, that they need "
        "nothing else, goodbye or similar: reply ONLY with a short closing "
        "(you're welcome, anytime). "
        "FORBIDDEN in that case: asking whether they need anything else, proposing plans, "
        "trips, new tasks or any other question. "
        "Never end with a question mark after a closing or a thank you. "
        "MEMORY: you have long-term memory through remember_fact. "
        "If the user explicitly asks you to remember or note something down, use remember_fact. "
        "If they mention an important personal or technical detail (IP, pinout, preference, "
        "lab password, etc.) without asking you to save it, ask first whether they want you "
        "to remember it; do not call the tool until they say yes. "
        "If they ask about something that might be in memory, lean on the memories injected below. "
        "NETWORK: Before relying on live information (weather, news, prices, current events), "
        "use web_search. "
        "If a tool returns error \"no_internet\", \"searxng_unavailable\", \"timeout\", or "
        "\"provider_not_configured\", tell the user clearly. Do NOT invent search results, "
        "prices, news, weather, or citations. "
        "BACKGROUND JOBS: When a long task is started (research, competition analysis, "
        "demo job or similar), confirm briefly that you are working on it and that you "
        "will notify them. "
        "Do not pretend the final report is ready. When asked to read a finished job, "
        "use read_job_result; do not fabricate the report. "
        "RESEARCH: For investigate/research/summarize-a-topic requests, first ask whether "
        "to use local (Ollama + SearXNG), Claude, or ChatGPT. Then call start_research "
        "with topic and provider. If the tool returns need_provider_choice or "
        "provider_not_configured, tell the user and offer local. "
        "COMPETITION: For competitors/competition/competitive-landscape requests "
        "(analiza la competencia, competitors of X), first ask local vs Claude vs "
        "ChatGPT, then call start_competition_analysis with target and provider. "
        "Same need_provider_choice / provider_not_configured handling as research. "
        "Do not invent pricing or the brief before the job finishes. "
        "TOOL DISCIPLINE: "
        "Only call a tool if the user asks for a concrete action "
        "(open, list, write, search code, Docker, clipboard, calculation, docs, web search, "
        "research, competition analysis, background job, remember). "
        "Greetings, small talk, opinions or general questions: answer WITHOUT tools. "
        "If you are unsure which tool applies or data is missing: ask, do not execute. "
        "Prefer zero tools over the wrong one. Never invent tool results. "
        f"Files only inside {allowed_root}. "
        f"Allowlisted apps: {apps}. "
        "Use at most one tool per turn."
    )

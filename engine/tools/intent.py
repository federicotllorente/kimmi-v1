"""Router de intención: clasifica el pedido del usuario antes de exponer tools."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import ollama

from locale_config import current_locale

# Intents válidos y tools asociadas (subconjunto del schema completo).
INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "chat": [],
    "clarify": [],
    "apps": ["launch_app", "focus_window"],
    "files": ["list_directory", "read_directory_files", "write_file"],
    "code": ["search_codebase", "run_cursor_agent", "scaffold_project"],
    "docker": ["manage_docker_container"],
    "clipboard": ["manage_clipboard"],
    "calc": ["exec_python_repl"],
    "docs": ["fetch_web_docs"],
    "memory": ["remember_fact"],
}

VALID_INTENTS: Set[str] = set(INTENT_TOOL_MAP.keys())

INTENT_CONFIDENCE_THRESHOLD = 0.55

DOCKER_READ_ACTIONS = frozenset({
    "list",
    "inspect",
    "logs",
    "stats",
    "top",
    "compose_ps",
    "compose_logs",
})

CLIPBOARD_WRITE_ACTIONS = frozenset({"write", "append", "clear"})

ALWAYS_CONFIRM_TOOLS = frozenset({
    "launch_app",
    "write_file",
    "run_cursor_agent",
    "scaffold_project",
    "remember_fact",
})

_CLASSIFY_SYSTEM = (
    "You are an intent classifier for a voice assistant named Kimmi. "
    "Reply ONLY with valid JSON, no markdown and no extra text. "
    "Schema: {\"intent\": string, \"confidence\": number, \"brief_reason\": string}. "
    "intent must be exactly one of: chat, apps, files, code, docker, clipboard, calc, docs, memory, clarify. "
    "Rules: "
    "chat = greeting, small talk, general question with no action on the PC "
    "(including asking about something that should already be in memory, without asking to save it); "
    "apps = open or focus an application; "
    "files = list, read or write files and folders; "
    "code = search code, scaffold, or the Cursor agent; "
    "docker = containers or compose; "
    "clipboard = the clipboard; "
    "calc = calculations or numeric conversions; "
    "docs = web documentation; "
    "memory = the user EXPLICITLY asks to remember, note down or save something to memory "
    "(remember that..., note..., save that...); "
    "clarify = the audio is ambiguous, incomplete or unintelligible. "
    "confidence between 0 and 1. When unsure, use clarify with low confidence. "
    "The user may speak any language; classify the intent regardless."
)


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    brief_reason: str

    @property
    def should_use_tools(self) -> bool:
        if self.intent in ("chat", "clarify"):
            return False
        if self.confidence < INTENT_CONFIDENCE_THRESHOLD:
            return False
        return bool(INTENT_TOOL_MAP.get(self.intent))


def parse_intent_json(raw: str) -> IntentResult:
    """Parsea la respuesta del clasificador; ante fallo → clarify."""
    text = (raw or "").strip()
    if not text:
        return IntentResult("clarify", 0.0, "empty classifier response")

    # Extraer bloque JSON si viene envuelto en texto/markdown.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return IntentResult("clarify", 0.0, "unparseable json")

    intent = str(data.get("intent", "clarify")).strip().lower()
    if intent not in VALID_INTENTS:
        intent = "clarify"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("brief_reason", "") or "")[:200]
    return IntentResult(intent, confidence, reason)


def classify_intent(
    user_text: str,
    model: str,
    recent_context: Optional[List[Dict[str, Any]]] = None,
) -> IntentResult:
    """Clasifica intención con Ollama (sin tools)."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
    ]
    if recent_context:
        # Últimos turnos user/assistant como pista (sin system duplicado).
        for msg in recent_context[-4:]:
            role = msg.get("role")
            if role in ("user", "assistant") and msg.get("content"):
                messages.append({"role": role, "content": str(msg["content"])[:500]})

    messages.append({"role": "user", "content": user_text})

    response = ollama.chat(
        model=model,
        messages=messages,
        keep_alive="24h",
        options={
            "temperature": 0.1,
            "num_predict": 80,  # JSON corto; evita generar de más
        },
    )
    content = response.get("message", {}).get("content", "")
    return parse_intent_json(content)


def tool_names_for_intent(intent: str) -> List[str]:
    return list(INTENT_TOOL_MAP.get(intent, []))


def tool_needs_confirmation(fn_name: str, fn_args: Optional[Dict[str, Any]] = None) -> bool:
    """True si la tool tiene efecto lateral y debe confirmarse por voz."""
    args = fn_args or {}
    if fn_name in ALWAYS_CONFIRM_TOOLS:
        return True
    if fn_name == "manage_docker_container":
        action = str(args.get("action", "")).strip().lower()
        return bool(action) and action not in DOCKER_READ_ACTIONS
    if fn_name == "manage_clipboard":
        action = str(args.get("action", "")).strip().lower()
        return action in CLIPBOARD_WRITE_ACTIONS
    return False


def confirmation_prompt(fn_name: str, fn_args: Optional[Dict[str, Any]] = None) -> str:
    """Frase corta para pedir confirmación por TTS."""
    args = fn_args or {}
    templates = current_locale().confirm

    if fn_name == "launch_app":
        app = args.get("app_id") or templates["default_app"]
        path = args.get("path") or ""
        if path:
            return templates["launch_app_path"].format(app=app, path=path)
        return templates["launch_app"].format(app=app)
    if fn_name == "write_file":
        path = args.get("file_path") or templates["default_path"]
        return templates["write_file"].format(path=path)
    if fn_name == "run_cursor_agent":
        return templates["run_cursor_agent"]
    if fn_name == "scaffold_project":
        name = args.get("name") or templates["default_project"]
        return templates["scaffold_project"].format(name=name)
    if fn_name == "remember_fact":
        fact = str(args.get("fact") or templates["default_fact"]).strip()
        short = fact if len(fact) <= 90 else fact[:87] + "..."
        return templates["remember_fact"].format(fact=short)
    if fn_name == "manage_docker_container":
        action = args.get("action") or templates["default_action"]
        return templates["manage_docker_container"].format(action=action)
    if fn_name == "manage_clipboard":
        action = args.get("action") or templates["default_clipboard_action"]
        return templates["manage_clipboard"].format(action=action)
    return templates["fallback"]


def parse_yes_no(transcript: str) -> Optional[bool]:
    """True=sí, False=no, None=indeterminado."""
    text = (transcript or "").strip().lower()
    if not text:
        return None
    loc = current_locale()
    yes = bool(loc.yes_re.search(text))
    no = bool(loc.no_re.search(text))
    if yes and not no:
        return True
    if no and not yes:
        return False
    if yes and no:
        # "sí no" raro → indeterminado
        return None
    return None


def last_assistant_offered_memory(history: Optional[List[Dict[str, Any]]]) -> bool:
    """True si el último mensaje del asistente ofreció guardar algo en memoria."""
    if not history:
        return False
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        return bool(current_locale().memory_offer_re.search(content))
    return False


def previous_user_fact(history: Optional[List[Dict[str, Any]]]) -> str:
    """Último texto del usuario antes del offer (candidato a recuerdo)."""
    if not history:
        return ""
    # history: ... user(fact), assistant(offer)  OR includes older turns
    saw_assistant = False
    for msg in reversed(history):
        role = msg.get("role")
        if role == "assistant" and not saw_assistant:
            saw_assistant = True
            continue
        if saw_assistant and role == "user":
            return str(msg.get("content") or "").strip()
    return ""


def should_force_memory_save(
    user_text: str,
    history: Optional[List[Dict[str, Any]]],
) -> bool:
    """Usuario confirma tras una oferta de recordar → forzar intent memory."""
    if not last_assistant_offered_memory(history):
        return False
    return parse_yes_no(user_text) is True

"""Router de intención: clasifica el pedido del usuario antes de exponer tools."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import ollama

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

_YES_RE = re.compile(
    r"\b(s[ií]|dale|ok|okay|vale|confirmo|confirma|afirmativo|adelante|hazlo|claro)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"\b(no|nop|cancelar|cancela|cancele|negativo|para|alto|stop)\b",
    re.IGNORECASE,
)

_MEMORY_OFFER_RE = re.compile(
    r"(?i)("
    r"quieres?\s+que\s+(?:lo\s+)?recuerde|"
    r"lo\s+guardo\s+en\s+(?:la\s+)?memoria|"
    r"guardo\s+(?:ese|este)\s+dato|"
    r"anoto\s+(?:eso|ese\s+dato)|"
    r"recuerde\s+(?:eso|ese\s+dato|que)"
    r")"
)

_CLASSIFY_SYSTEM = (
    "Eres un clasificador de intención para un asistente de voz llamado Jarvis. "
    "Responde SOLO con un JSON válido, sin markdown ni texto extra. "
    "Schema: {\"intent\": string, \"confidence\": number, \"brief_reason\": string}. "
    "intent debe ser exactamente uno de: chat, apps, files, code, docker, clipboard, calc, docs, memory, clarify. "
    "Reglas: "
    "chat = saludo, charla, pregunta general sin acción en el PC "
    "(incluye preguntar por un dato que ya debería estar en memoria, sin pedir guardarlo); "
    "apps = abrir o enfocar una aplicación; "
    "files = listar, leer o escribir archivos/carpetas; "
    "code = buscar en código, scaffold o agente Cursor; "
    "docker = contenedores o compose; "
    "clipboard = portapapeles; "
    "calc = cálculos o conversiones numéricas; "
    "docs = documentación web; "
    "memory = el usuario pide EXPLÍCITAMENTE recordar, anotar o guardar un dato en memoria "
    "(recuerda que..., anota..., guarda que...); "
    "clarify = audio ambiguo, incompleto o no se entiende. "
    "confidence entre 0 y 1. Si dudas, usa clarify con confidence baja."
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
    if fn_name == "launch_app":
        app = args.get("app_id", "la aplicación")
        path = args.get("path") or ""
        if path:
            return f"Voy a abrir {app} en {path}. ¿Confirmas?"
        return f"Voy a abrir {app}. ¿Confirmas?"
    if fn_name == "write_file":
        path = args.get("file_path", "un archivo")
        return f"Voy a escribir en {path}. ¿Confirmas?"
    if fn_name == "run_cursor_agent":
        return "Voy a delegar la tarea al agente de Cursor. ¿Confirmas?"
    if fn_name == "scaffold_project":
        name = args.get("name", "el proyecto")
        return f"Voy a crear el scaffold {name}. ¿Confirmas?"
    if fn_name == "remember_fact":
        fact = str(args.get("fact", "ese dato")).strip()
        short = fact if len(fact) <= 90 else fact[:87] + "..."
        return f"Voy a recordar: {short}. ¿Confirmas?"
    if fn_name == "manage_docker_container":
        action = args.get("action", "esa acción")
        return f"Voy a ejecutar Docker {action}. ¿Confirmas?"
    if fn_name == "manage_clipboard":
        action = args.get("action", "modificar")
        return f"Voy a {action} el portapapeles. ¿Confirmas?"
    return "Voy a ejecutar esa acción. ¿Confirmas?"


def parse_yes_no(transcript: str) -> Optional[bool]:
    """True=sí, False=no, None=indeterminado."""
    text = (transcript or "").strip().lower()
    if not text:
        return None
    yes = bool(_YES_RE.search(text))
    no = bool(_NO_RE.search(text))
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
        return bool(_MEMORY_OFFER_RE.search(content))
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

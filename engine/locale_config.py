"""Everything in Kimmi that depends on the user's language, in one place.

The rule this module exists to enforce: no other file should contain a spoken
literal, a language-specific regex, or a hardcoded voice. They all live here,
and callers read them through ``current_locale()``.

Deliberately imports nothing from the project, so ``assistant.py``,
``tools/intent.py``, ``tools/registry.py`` and ``tools/memory.py`` can all import
it without a cycle.

The holder is mutable because the language has to be swappable mid-session.
That means callers must read ``current_locale()`` **per call** instead of binding
module-level constants at import time.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Pattern, Tuple

DEFAULT_CODE = "en"
ENV_VAR = "KIMMI_LANG"


@dataclass(frozen=True)
class Locale:
    code: str
    language_name: str  # goes into the system prompt verbatim

    # -- STT --
    whisper_language: str
    # Template with {name} and {apps}: the app list lives in tools/, which this
    # module must not import.
    stt_initial_prompt: str

    # -- TTS --
    tts_voice: str
    tts_speed: float
    tts_lang: str
    piper_model: Optional[str]  # None when no Piper voice exists for this language

    # -- Spoken phrases --
    wake_ack: Tuple[str, ...]  # templates with {name}
    closing_ack: Tuple[str, ...]
    filler: Tuple[str, ...]
    filler_tools: str
    spoken: Mapping[str, str]
    confirm: Mapping[str, str]

    # -- Language-specific matching --
    wake_phrase_re: Pattern[str]
    closing_re: Pattern[str]
    closing_soft_re: Pattern[str]
    closing_starters: Tuple[str, ...]
    yes_re: Pattern[str]
    no_re: Pattern[str]
    memory_offer_re: Pattern[str]
    rag_skip_re: Pattern[str]


# ---------------------------------------------------------------- English ----

EN = Locale(
    code="en",
    language_name="English",
    whisper_language="en",
    stt_initial_prompt=(
        "The user speaks clear English to Kimmi. "
        "Their name is {name}. "
        "Examples: open the browser, open Cursor, open VS Code, open the terminal, "
        "list the files, read the tools folder, write a file, "
        "search the code, Docker, container logs, clipboard, "
        "remember that the IP is, what IP does the Raspberry have, "
        "what is seventeen times twenty-three, FastAPI documentation, "
        "hey how are you, cancel, yes confirm. "
        "Apps: {apps}."
    ),
    tts_voice="af_heart",
    # 1.2 is calibrated for Spanish; in English it sounds rushed.
    tts_speed=1.0,
    tts_lang="en-us",
    piper_model=None,
    wake_ack=(
        "Yes, {name}?",
        "What's up, {name}?",
        "I'm listening.",
        "Mm-hm?",
        "Go ahead.",
        "I'm here.",
        "What do you need, {name}?",
    ),
    closing_ack=(
        "You got it.",
        "Anytime.",
        "I'm here if you need me.",
        "Sounds good.",
        "See you.",
        "Just say the word.",
        "All set.",
        "My pleasure.",
    ),
    filler=(
        "Give me a second.",
        "Let's see...",
        "Let me check.",
        "One moment.",
    ),
    filler_tools="Let me check that.",
    spoken={
        "not_heard_session_end": "I didn't catch that. Call me whenever you need me.",
        "not_heard_repeat": "I didn't quite catch that. Could you say it again?",
        "not_heard_cancelled": "I didn't catch that. Cancelled.",
        "cancelled": "Cancelled.",
        "confirm_unclear": "I didn't understand the confirmation. Cancelled.",
        "tool_unavailable": "I can't use that tool.",
        "done": "All set.",
        "unclear_request": (
            "I'm not sure what you asked. "
            "Could you put it in one clear sentence?"
        ),
        # Injected as the user message when they confirm a memory offer.
        "force_memory_save": (
            "Yes, use remember_fact to save this exactly: {fact}"
        ),
    },
    confirm={
        "launch_app_path": "I'm going to open {app} in {path}. Confirm?",
        "launch_app": "I'm going to open {app}. Confirm?",
        "write_file": "I'm going to write to {path}. Confirm?",
        "run_cursor_agent": "I'm going to hand the task to the Cursor agent. Confirm?",
        "scaffold_project": "I'm going to create the {name} scaffold. Confirm?",
        "remember_fact": "I'm going to remember: {fact}. Confirm?",
        "manage_docker_container": "I'm going to run Docker {action}. Confirm?",
        "manage_clipboard": "I'm going to {action} the clipboard. Confirm?",
        "fallback": "I'm going to run that action. Confirm?",
        "default_app": "the app",
        "default_path": "a file",
        "default_project": "the project",
        "default_fact": "that",
        "default_action": "that action",
        "default_clipboard_action": "change",
    },
    # Covers the usual English mis-transcriptions of the name.
    wake_phrase_re=re.compile(
        r"^\s*(?:(?:hey|hi|yo|ok|okay|hey\s+there)\s+)?"
        r"(?:kimmi|kimmy|kimmie|kimi|kee\s*me)"
        r"(?:\s*[,.!?]+\s*|\s+|\s*$)",
        re.IGNORECASE,
    ),
    closing_re=re.compile(
        r"(?i)^\s*(?:well[,.]?\s+|ok(?:ay)?[,.]?\s+|cool[,.]?\s+|great[,.]?\s+"
        r"|alright[,.]?\s+|perfect[,.]?\s+)?"
        r"("
        r"thanks(?:\s+(?:a\s+lot|so\s+much|again))?"
        r"|thank\s+you(?:\s+(?:very|so)\s+much)?"
        r"|(?:that|that'?s)\s+(?:all|it)(?:\s+for\s+now)?"
        r"|nothing\s+else(?:\s+for\s+now)?"
        r"|(?:i'?m|i\s+am)\s+(?:good|fine|all\s+set|all\s+good)(?:\s*,?\s*thanks)?"
        r"|all\s+(?:good|set)(?:\s*,?\s*thanks)?"
        r"|we(?:'re|\s+are)\s+done"
        r"|no(?:pe)?\s*,?\s*(?:i'?m|i\s+am)\s+(?:good|fine|all\s+set)"
        r"|not\s+(?:right\s+now|for\s+now|now)"
        r"|i\s+don'?t\s+need\s+anything(?:\s+else)?(?:\s+(?:right\s+)?now)?"
        r"|good\s?bye|bye(?:\s+bye)?|see\s+(?:you|ya)(?:\s+later)?|later"
        r"|cool|great|perfect|ok(?:ay)?|got\s+it|sounds\s+good"
        r")[\s\.\!\,]*$",
    ),
    closing_soft_re=re.compile(
        r"(?i)("
        r"\bthanks(?:\s+(?:a\s+lot|so\s+much))?\s*$"
        r"|\bthank\s+you(?:\s+(?:very|so)\s+much)?\s*$"
        r"|\bi\s+don'?t\s+need\s+anything(?:\s+else)?\b"
        r"|\bnothing\s+else(?:\s+for\s+now)?\b"
        r"|\b(?:that|that'?s)\s+(?:all|it)\b"
        r"|\b(?:i'?m|i\s+am)\s+(?:good|fine|all\s+set)\b"
        r"|\ball\s+(?:good|set)\b"
        r"|\bwe(?:'re|\s+are)\s+done\b"
        r"|\bsee\s+(?:you|ya)(?:\s+later)?\b"
        r"|\bgood\s?bye\b"
        r")",
    ),
    closing_starters=(
        "thanks",
        "thank you",
        "that's all",
        "thats all",
        "that's it",
        "thats it",
        "nothing else",
        "i'm good",
        "im good",
        "all good",
        "all set",
        "see you",
        "see ya",
        "goodbye",
        "good bye",
        "bye",
        "we're done",
        "were done",
        "i don't need anything else",
        "i dont need anything else",
    ),
    yes_re=re.compile(
        r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|affirmative"
        r"|go\s+ahead|do\s+it|please\s+do|absolutely|correct)\b",
        re.IGNORECASE,
    ),
    no_re=re.compile(
        r"\b(no|nope|nah|cancel|don'?t|do\s+not|negative|stop"
        r"|forget\s+it|never\s+mind)\b",
        re.IGNORECASE,
    ),
    memory_offer_re=re.compile(
        r"(?i)("
        r"want\s+me\s+to\s+remember|"
        r"should\s+i\s+remember|"
        r"save\s+(?:that|this)\s+(?:fact|to\s+memory)|"
        r"(?:i'?ll|shall\s+i)\s+save\s+(?:that|this)|"
        r"note\s+(?:that|this)\s+down|"
        r"remember\s+(?:that|this)\s+for\s+you"
        r")"
    ),
    rag_skip_re=re.compile(
        r"(?i)^\s*(?:hi|hey|hello|yo|what'?s\s+up|how\s+are\s+you"
        r"|how'?s\s+it\s+going|good\s+(?:morning|afternoon|evening)"
        r"|yes|yeah|no|nope|ok(?:ay)?|sure|thanks|thank\s+you)"
        r"[\s\.\!\?\,]*$"
    ),
)


# ---------------------------------------------------------------- Spanish ----
# Values copied verbatim from the pre-i18n engine, so Spanish stays a control
# line for spotting regressions.

ES = Locale(
    code="es",
    language_name="Spanish",
    whisper_language="es",
    stt_initial_prompt=(
        "El usuario habla en español claro con Kimmi. "
        "Se llama {name}. "
        "Ejemplos: abre el navegador, abre Cursor, abre VS Code, abre la terminal, "
        "lista los archivos, lee la carpeta tools, escribe un archivo, "
        "busca en el código, Docker, logs del contenedor, portapapeles, "
        "recuerda que la IP es, qué IP tiene la Raspberry, "
        "cuánto es diecisiete por veintitrés, documentación de FastAPI, "
        "hola qué tal, cancela, sí confirma. "
        "Apps: {apps}."
    ),
    tts_voice="ef_dora",
    tts_speed=1.2,
    tts_lang="es",
    piper_model="piper/es_MX-claude-high.onnx",
    wake_ack=(
        "Sí, {name}?",
        "Dime, {name}.",
        "Te escucho.",
        "Ajá.",
        "Dime.",
        "Aquí estoy.",
        "¿Qué necesitas, {name}?",
    ),
    closing_ack=(
        "De nada.",
        "Cuando quieras.",
        "Aquí estaré.",
        "Perfecto.",
        "Hasta luego.",
        "Avísame si me necesitas.",
        "Listo.",
        "Un placer.",
    ),
    filler=(
        "Dame un segundo.",
        "A ver...",
        "Déjame ver.",
        "Un momento.",
    ),
    filler_tools="Déjame comprobarlo.",
    spoken={
        "not_heard_session_end": "No te escuché. Cuando quieras, vuelve a llamarme.",
        "not_heard_repeat": "No te escuché bien, ¿puedes repetir?",
        "not_heard_cancelled": "No te escuché bien. Cancelado.",
        "cancelled": "Cancelado.",
        "confirm_unclear": "No entendí la confirmación. Cancelado.",
        "tool_unavailable": "No puedo usar esa herramienta.",
        "done": "Listo.",
        "unclear_request": (
            "No estoy seguro de lo que pediste. "
            "¿Puedes reformularlo en una frase más concreta?"
        ),
        "force_memory_save": (
            "Sí, usa remember_fact para guardar este dato exactamente: {fact}"
        ),
    },
    confirm={
        "launch_app_path": "Voy a abrir {app} en {path}. ¿Confirmas?",
        "launch_app": "Voy a abrir {app}. ¿Confirmas?",
        "write_file": "Voy a escribir en {path}. ¿Confirmas?",
        "run_cursor_agent": "Voy a delegar la tarea al agente de Cursor. ¿Confirmas?",
        "scaffold_project": "Voy a crear el scaffold {name}. ¿Confirmas?",
        "remember_fact": "Voy a recordar: {fact}. ¿Confirmas?",
        "manage_docker_container": "Voy a ejecutar Docker {action}. ¿Confirmas?",
        "manage_clipboard": "Voy a {action} el portapapeles. ¿Confirmas?",
        "fallback": "Voy a ejecutar esa acción. ¿Confirmas?",
        "default_app": "la aplicación",
        "default_path": "un archivo",
        "default_project": "el proyecto",
        "default_fact": "ese dato",
        "default_action": "esa acción",
        "default_clipboard_action": "modificar",
    },
    wake_phrase_re=re.compile(
        r"^\s*(?:(?:hey|ey|ok|okay|hola)\s+)?(?:kimmi|kimi|kee\s*me|quimi)"
        r"(?:\s*[,.!?]+\s*|\s+|\s*$)",
        re.IGNORECASE,
    ),
    closing_re=re.compile(
        r"(?i)^\s*(?:bueno[,.]?\s+|vale[,.]?\s+|ok(?:ay)?[,.]?\s+|perfecto[,.]?\s+)?"
        r"("
        r"gracias(?:\s+(?:mucho|mil|de\s+nuevo))?"
        r"|(?:muy\s+)?bien(?:\s*,?\s*gracias)?"
        r"|perfecto(?:\s*,?\s*gracias)?"
        r"|vale(?:\s*,?\s*gracias)?"
        r"|ok(?:ay)?(?:\s*,?\s*gracias)?"
        r"|listo(?:\s*,?\s*gracias)?"
        r"|eso\s+es\s+todo"
        r"|nada\s+m[aá]s(?:\s+por\s+ahora)?"
        r"|ya\s+est[aá]"
        r"|ya\s+bastante"
        r"|no\s+(?:necesito|preciso)\s+(?:nada\s+)?m[aá]s(?:\s+por\s+(?:ahora|el\s+momento))?"
        r"|por\s+ahora\s+(?:nada|estoy\s+bien|no(?:\s+necesito)?\s+nada)?"
        r"|(?:ahora\s+)?estoy\s+(?:bien|ok|okay)(?:\s*,?\s*gracias)?"
        r"|chau"
        r"|adi[oó]s"
        r"|hasta\s+luego"
        r"|nos\s+vemos"
        r"|bye"
        r"|cuando\s+quiera(?:s)?"
        r"|no\s+por\s+ahora"
        r")[\s\.\!\,]*$",
    ),
    closing_soft_re=re.compile(
        r"(?i)("
        r"\bgracias(?:\s+(?:mucho|mil))?\s*$"
        r"|\bno\s+(?:necesito|preciso)\s+(?:nada\s+)?m[aá]s\b"
        r"|\bnada\s+m[aá]s\s+por\s+ahora\b"
        r"|\bpor\s+el\s+momento\s+nada\b"
        r"|\bno\s+necesito\s+nada\s+m[aá]s\s+por\s+(?:ahora|el\s+momento)\b"
        r"|\bestoy\s+bien(?:\s*,?\s*gracias)?\b"
        r"|\beso\s+es\s+todo\b"
        r"|\bya\s+est[aá]\b"
        r"|\bhasta\s+luego\b"
        r"|\badi[oó]s\b"
        r")",
    ),
    closing_starters=(
        "gracias",
        "perfecto, gracias",
        "perfecto gracias",
        "vale, gracias",
        "ok, gracias",
        "okay, gracias",
        "listo, gracias",
        "eso es todo",
        "nada más",
        "nada mas",
        "ya está",
        "ya esta",
        "hasta luego",
        "nos vemos",
        "no necesito nada más",
        "no necesito nada mas",
        "estoy bien",
        "ahora estoy bien",
    ),
    yes_re=re.compile(
        r"\b(s[ií]|dale|ok|okay|vale|confirmo|confirma|afirmativo|adelante|hazlo|claro)\b",
        re.IGNORECASE,
    ),
    no_re=re.compile(
        r"\b(no|nop|cancelar|cancela|cancele|negativo|para|alto|stop)\b",
        re.IGNORECASE,
    ),
    memory_offer_re=re.compile(
        r"(?i)("
        r"quieres?\s+que\s+(?:lo\s+)?recuerde|"
        r"lo\s+guardo\s+en\s+(?:la\s+)?memoria|"
        r"guardo\s+(?:ese|este)\s+dato|"
        r"anoto\s+(?:eso|ese\s+dato)|"
        r"recuerde\s+(?:eso|ese\s+dato|que)"
        r")"
    ),
    rag_skip_re=re.compile(
        r"(?i)^\s*(?:hola|buenas|hey|qu[eé]\s+tal|c[oó]mo\s+est[aá]s|buenos\s+d[ií]as"
        r"|buenas\s+tardes|buenas\s+noches|s[ií]|no|vale|ok(?:ay)?|dale|gracias)"
        r"[\s\.\!\?\,]*$"
    ),
)


LOCALES: Dict[str, Locale] = {EN.code: EN, ES.code: ES}


def resolve(code: Optional[str]) -> Locale:
    """Look up a locale, falling back to the default for anything unknown."""
    key = (code or "").strip().lower()
    return LOCALES.get(key) or LOCALES[DEFAULT_CODE]


_current: Locale = resolve(os.getenv(ENV_VAR, DEFAULT_CODE))


def current_locale() -> Locale:
    return _current


def set_locale(code: str) -> Locale:
    """Swap the active language. Callers that cached anything must re-read it."""
    global _current
    _current = resolve(code)
    return _current

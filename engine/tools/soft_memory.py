"""Soft-auto long-term memory: silent background extraction after a voice turn.

Runs off the TTFA path. Kill with KIMMI_SOFT_MEMORY=0.
"""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import ollama

from locale_config import current_locale
from tools.memory import DEDUP_DISTANCE, MAX_FACT_CHARS, memory_db

ALLOWED_CATEGORIES = frozenset({
    "preference",
    "network",
    "hardware",
    "project",
    "general",
})

_MIN_USER_CHARS = 20
_MAX_FACTS = 2
_EXTRACT_NUM_PREDICT = 120
_DEFAULT_MODEL = "qwen2.5:7b"

# Signals that the utterance might contain a durable fact (bilingual, cheap).
_STABLE_SIGNAL_RE = re.compile(
    r"(?ix)"
    r"("
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"  # IPv4
    r"|\b(?:my|mi|mis|nuestro|nuestra)\b"
    r"|\b(?:prefer|prefiero|prefers|always|siempre|never|nunca)\b"
    r"|\b(?:ip|pinout|raspberry|esp32|arduino|ssid|wifi|hostname|puerto|port)\b"
    r"|\b(?:project|proyecto|repo|workspace|lab)\b"
    r"|\b(?:llam[oa]|called|name is|me llamo|se llama)\b"
    r"|\d{2,}"
    r")"
)

_SECRET_RE = re.compile(
    r"(?ix)"
    r"("
    r"\bpassword\b|\bpasswd\b|\bcontrase[nñ]a\b|\bclave\b"
    r"|\bapi[_\s-]?key\b|\bsecret\b|\btoken\b|\bbearer\b"
    r"|\bprivate[_\s-]?key\b|\bpincode\b|\bpin\s*code\b"
    r"|\bcvv\b|\bssn\b"
    r")"
)

_EXTRACT_SYSTEM = (
    "You extract durable personal/technical facts for a voice assistant's long-term memory. "
    "Reply ONLY with valid JSON, no markdown. "
    'Schema: {"facts":[{"text":string,"category":string}]}. '
    "category must be one of: preference, network, hardware, project, general. "
    "Rules: return 0 to 2 facts; empty facts if the turn is small talk or has nothing durable; "
    "paraphrase ONLY information clearly stated by the USER (never invent, never use the "
    "assistant reply as a source of new facts); "
    "never extract passwords, PINs, API keys, tokens, or other secrets; "
    "each text must be a short standalone sentence in the user's language."
)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kimmi-soft-mem")
_lock = threading.Lock()
_memory_written_this_turn = False


def soft_memory_enabled() -> bool:
    raw = (os.environ.get("KIMMI_SOFT_MEMORY") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def mark_memory_written_this_turn() -> None:
    """Call when remember_fact succeeded in the current voice turn."""
    global _memory_written_this_turn
    with _lock:
        _memory_written_this_turn = True


def consume_memory_written_this_turn() -> bool:
    """Return and clear the per-turn remember_fact flag."""
    global _memory_written_this_turn
    with _lock:
        written = _memory_written_this_turn
        _memory_written_this_turn = False
        return written


def reset_memory_written_this_turn() -> None:
    with _lock:
        global _memory_written_this_turn
        _memory_written_this_turn = False


def schedule_soft_memory(
    user_text: str,
    assistant_text: str = "",
    *,
    model: str = _DEFAULT_MODEL,
    skip: bool = False,
) -> None:
    """Enqueue background extraction. Never blocks the voice path."""
    if skip or not soft_memory_enabled():
        return
    if consume_memory_written_this_turn():
        print("  [Memory]: soft-auto skipped (already saved this turn)")
        return
    user = (user_text or "").strip()
    if not _should_extract(user):
        return
    reply = (assistant_text or "").strip()
    try:
        _executor.submit(_run_soft_memory, user, reply, model)
    except Exception as exc:  # noqa: BLE001
        print(f"  [Memory]: soft-auto schedule failed: {exc}")


def _should_extract(user_text: str) -> bool:
    if len(user_text) < _MIN_USER_CHARS:
        return False
    try:
        loc = current_locale()
        if loc.rag_skip_re.match(user_text):
            return False
        if loc.closing_re.match(user_text):
            return False
    except Exception:  # noqa: BLE001
        pass
    if _SECRET_RE.search(user_text):
        # Secrets must never enter soft-auto; explicit remember_fact path only.
        print("  [Memory]: soft-auto skipped (secret-like utterance)")
        return False
    if not _STABLE_SIGNAL_RE.search(user_text):
        return False
    return True


def _run_soft_memory(user_text: str, assistant_text: str, model: str) -> None:
    try:
        facts = _extract_facts(user_text, assistant_text, model)
        if not facts:
            print("  [Memory]: soft-auto skipped (no facts)")
            return
        for fact in facts:
            _persist_fact(fact["text"], fact["category"])
    except Exception as exc:  # noqa: BLE001
        print(f"  [Memory]: soft-auto failed: {exc}")


def _extract_facts(
    user_text: str,
    assistant_text: str,
    model: str,
) -> List[Dict[str, str]]:
    user_payload = f"User said:\n{user_text[:1000]}"
    if assistant_text:
        user_payload += f"\n\nAssistant replied (context only, do not mine new facts):\n{assistant_text[:500]}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user_payload},
        ],
        keep_alive="24h",
        options={
            "temperature": 0.1,
            "num_predict": _EXTRACT_NUM_PREDICT,
        },
    )
    raw = ""
    if isinstance(response, dict):
        raw = str((response.get("message") or {}).get("content") or "")
    else:
        message = getattr(response, "message", None)
        raw = str(getattr(message, "content", "") or "")
    return _parse_facts_json(raw, user_text)


def _parse_facts_json(raw: str, user_text: str) -> List[Dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []

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
        print("  [Memory]: soft-auto skipped (unparseable json)")
        return []

    items = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    user_lower = user_text.lower()
    out: List[Dict[str, str]] = []
    for item in items[:_MAX_FACTS]:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("text") or "").strip()
        category = str(item.get("category") or "general").strip().lower()
        if not fact or len(fact) > MAX_FACT_CHARS:
            continue
        if category not in ALLOWED_CATEGORIES:
            category = "general"
        if _SECRET_RE.search(fact):
            print("  [Memory]: soft-auto skipped fact (secret filter)")
            continue
        # Light groundedness: require at least one contentful token overlap with user.
        if not _overlaps_user(fact, user_lower):
            print("  [Memory]: soft-auto skipped fact (not grounded in user text)")
            continue
        out.append({"text": fact, "category": category})
    return out


def _overlaps_user(fact: str, user_lower: str) -> bool:
    tokens = [
        t for t in re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9.]{3,}", fact.lower())
        if t not in {
            "the", "and", "que", "para", "con", "una", "uno", "los", "las",
            "del", "por", "user", "prefers", "called",
        }
    ]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in user_lower)
    return hits >= max(1, min(2, len(tokens) // 3 + 1))


def _persist_fact(fact: str, category: str) -> None:
    similar = memory_db.find_similar(fact, n=1)
    if similar is not None:
        _doc, dist = similar
        if dist <= DEDUP_DISTANCE:
            print(
                f"  [Memory]: soft-auto skipped (duplicate dist={dist:.3f})"
            )
            return
    result = memory_db.add_memory(fact, category=category, source="soft_auto")
    if result.startswith("Recuerdo guardado"):
        print(f"  [Memory]: soft-auto saved ({category}): {fact[:120]}")
    else:
        print(f"  [Memory]: soft-auto failed write: {result}")

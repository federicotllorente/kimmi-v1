"""Shared Ollama / Claude / ChatGPT helpers for background research jobs."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CLOUD_TIMEOUT_SEC = 90
QUERY_EXPAND_MODEL_DEFAULT = "qwen2.5:3b"
VALID_PROVIDERS = frozenset({"local", "claude", "chatgpt"})


def research_model() -> str:
    return (
        (os.environ.get("KIMMI_RESEARCH_MODEL") or "").strip()
        or (os.environ.get("KIMMI_LLM_MODEL") or "").strip()
        or "qwen2.5:14b"
    )


def fast_model() -> str:
    return (
        (os.environ.get("KIMMI_FAST_MODEL") or "").strip()
        or QUERY_EXPAND_MODEL_DEFAULT
    )


def ollama_chat(model: str, prompt: str, num_predict: int = 600) -> str:
    try:
        import ollama
    except ImportError:
        return ""
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            keep_alive="24h",
            options={"num_predict": num_predict},
        )
        content = (
            resp.get("message", {}).get("content", "") if isinstance(resp, dict) else ""
        )
        if not content:
            content = getattr(getattr(resp, "message", None), "content", "") or ""
        return str(content).strip()
    except Exception:  # noqa: BLE001
        return ""


def claude_chat(prompt: str, max_tokens: int = 1200) -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return ""
    body = json.dumps(
        {
            "model": os.environ.get("KIMMI_CLAUDE_MODEL") or "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=CLOUD_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        parts = data.get("content") or []
        texts = [
            str(p.get("text") or "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(t for t in texts if t).strip()
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""


def openai_chat(prompt: str, max_tokens: int = 1200) -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return ""
    body = json.dumps(
        {
            "model": os.environ.get("KIMMI_CHATGPT_MODEL") or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=CLOUD_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""


def synthesize_with_provider(provider: str, prompt: str, num_predict: int = 900) -> str:
    provider = (provider or "").strip().lower()
    if provider == "local":
        return ollama_chat(research_model(), prompt, num_predict=num_predict)
    if provider == "claude":
        return claude_chat(prompt)
    if provider == "chatgpt":
        return openai_chat(prompt)
    return ""


def check_provider_ready(provider: str) -> str | None:
    """Return an error_json string if provider cannot run; else None."""
    from tools.connectivity import error_json

    provider_s = (provider or "").strip().lower()
    if provider_s not in VALID_PROVIDERS:
        return error_json(
            "invalid_args",
            f"provider must be one of: {', '.join(sorted(VALID_PROVIDERS))}",
        )
    if provider_s == "claude" and not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return error_json(
            "provider_not_configured",
            "ANTHROPIC_API_KEY is not set. Offer local instead.",
            provider="claude",
        )
    if provider_s == "chatgpt" and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        return error_json(
            "provider_not_configured",
            "OPENAI_API_KEY is not set. Offer local instead.",
            provider="chatgpt",
        )
    return None

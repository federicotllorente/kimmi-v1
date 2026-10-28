"""Turn a follow-up ("¿Y el azul?") into a self-contained request.

Used before the tools call, which only sees the system prompt and ONE user
message: Kimmi's earlier replies made the model repeat old answers, and with
the raw follow-up alone it reused the old arguments (searched the red pencil
again). Measured on qwen2.5:7b: 7/8 follow-ups rewritten correctly, ~0.3 s.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import ollama

_REWRITE_SYSTEM = (
    "You rewrite the LAST user message of a voice conversation into ONE complete, "
    "self-contained request, so it can be understood without the earlier messages. "
    "Resolve references ('it', 'that one', 'the blue one', 'and in Sevilla?') using the "
    "earlier user messages. The LAST message always wins: a new color, object, place or app "
    "REPLACES the old one, and you must not repeat or merge in the earlier request. "
    "Keep the user's language. If the last message is already self-contained, return it "
    "unchanged. Never answer it. "
    "Examples: "
    "earlier 'Where is the red pencil?' + last 'And the blue one?' -> 'Where is the blue pencil?'. "
    "earlier '¿Dónde está el lápiz rojo?' + last '¿Y el azul?' -> '¿Dónde está el lápiz azul?'. "
    "earlier '¿Dónde está la goma?' + last '¿Y el estuche?' -> '¿Dónde está el estuche?'. "
    "earlier 'Abre Visual Studio Code' + last 'Y ahora el navegador' -> 'Abre el navegador'. "
    "earlier '¿Qué tiempo hace en Madrid?' + last '¿Y en Sevilla?' -> '¿Qué tiempo hace en Sevilla?'. "
    "earlier '¿Hay una taza?' + last 'Reinicia el contenedor de redis' -> 'Reinicia el contenedor de redis'."
)

_REWRITE_FORMAT: Dict[str, Any] = {
    "type": "object",
    "properties": {"request": {"type": "string"}},
    "required": ["request"],
}

# Only the most recent user messages matter for resolving a reference
MAX_EARLIER_MESSAGES = 3


def rewrite_followup(earlier: List[str], last: str, model: str, num_ctx: int = 4096) -> str:
    """Self-contained version of `last`. Returns `last` unchanged on any error."""
    earlier = [text.strip() for text in earlier if text and text.strip()]
    if not earlier or not last.strip():
        return last

    conversation = "\n".join(
        f"Earlier user message: {text}" for text in earlier[-MAX_EARLIER_MESSAGES:]
    )
    conversation += f"\nLAST message: {last.strip()}"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": conversation},
            ],
            format=_REWRITE_FORMAT,
            keep_alive="24h",
            options={"num_ctx": num_ctx, "temperature": 0},
        )
        request = str(json.loads(response["message"]["content"]).get("request") or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  [Follow-up]: rewrite failed ({exc}); using the original text")
        return last

    return request or last

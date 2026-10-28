"""Build the tools call: one self-contained request, and forced calls.

Kept out of assistant.py so engine/tests can exercise it without loading the
microphone, Whisper or TTS.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import ollama

from tools.followup import rewrite_followup
from tools.registry import MEMORY_BLOCK_HEADER

# Tools the model must always call when they are the only option. The 7B
# preferred to do arithmetic in its head (and got 15% of 230 wrong); forcing
# the call through structured output got 6/6 right.
ALWAYS_FORCED_TOOLS = frozenset({"exec_python_repl"})


def tool_call_messages(
    messages: List[Dict[str, Any]],
    model: str,
    num_ctx: int,
) -> List[Dict[str, Any]]:
    """System prompt + ONE self-contained user request for the tools call.

    The router already decided a tool is needed, so Kimmi's earlier replies are
    left out: after one "I can't see", the model kept repeating it instead of
    calling the tool (2-4 of 8 calls). Follow-ups ("¿Y el azul?") are rewritten
    with the earlier user messages, so the model does not reuse old arguments.
    """
    system_msg, last_msg = messages[0], messages[-1]
    earlier = [
        str(m.get("content") or "") for m in messages[1:-1] if m.get("role") == "user"
    ]

    # Split the memory block off: only the user's words are rewritten
    text, sep, memories = str(last_msg.get("content") or "").partition(
        f"\n\n{MEMORY_BLOCK_HEADER}"
    )
    request = rewrite_followup(earlier, text, model, num_ctx)
    if request != text:
        print(f"  [Follow-up]: {text!r} -> {request!r}")

    return [system_msg, {"role": "user", "content": request + sep + memories}]


def forced_tool(tools: List[dict]) -> Optional[dict]:
    """The tool to force, if the subset is exactly one always-forced tool."""
    if len(tools) != 1:
        return None
    function = tools[0].get("function", {})
    return function if function.get("name") in ALWAYS_FORCED_TOOLS else None


def forced_tool_call(
    tool_messages: List[Dict[str, Any]],
    function: dict,
    model: str,
    num_ctx: int,
) -> Dict[str, Any]:
    """Get the tool's arguments through structured output, so a call always happens.

    Returns a response shaped like ollama.chat with one tool call, so the
    caller does not need to know the difference.
    """
    instruction = {
        "role": "system",
        "content": (
            f"Call the tool `{function['name']}` for the user's request. Reply only with "
            f"its JSON arguments. Tool description: {function.get('description', '')}"
        ),
    }
    response = ollama.chat(
        model=model,
        messages=tool_messages[:1] + [instruction] + tool_messages[1:],
        format=function["parameters"],
        keep_alive="24h",
        options={"num_ctx": num_ctx, "temperature": 0},
    )
    try:
        arguments = json.loads(response["message"]["content"] or "{}")
    except json.JSONDecodeError:
        arguments = {}
    print(f"  [Tools]: forced {function['name']}({arguments})")
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": function["name"], "arguments": arguments}}],
        }
    }


def request_tool_call(
    messages: List[Dict[str, Any]],
    tools: List[dict],
    model: str,
    num_ctx: int,
) -> Dict[str, Any]:
    """The whole tools call: rewrite the request, then force or let the model choose."""
    tool_messages = tool_call_messages(messages, model, num_ctx)
    function = forced_tool(tools)
    if function is not None:
        return forced_tool_call(tool_messages, function, model, num_ctx)
    return ollama.chat(
        model=model,
        messages=tool_messages,
        tools=tools,
        keep_alive="24h",
        options={
            "num_ctx": num_ctx,
            "temperature": 0,  # Tool choice should not depend on luck
        },
    )

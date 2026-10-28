"""Eval of the tools turn: does the model call the right tool with the right arguments?

Run from engine/:
    proj-env\\Scripts\\python.exe tests\\eval_tools.py
    proj-env\\Scripts\\python.exe tests\\eval_tools.py --model qwen2.5:14b

It goes through tools.tool_calls.request_tool_call, the same path as Kimmi:
follow-up rewrite, history filtering and forced calls included. Tools are NOT
executed, except the calculator (to check the number) and the find_object
translation step (to check the English name the camera would receive).
"""
from __future__ import annotations

import re

import _common  # noqa: F401  (sets sys.path)
from _common import ENGINE_DIR, Scoreboard, engine_number, parse_args

from tools.exec_python_repl import exec_python_repl
from tools.registry import build_system_prompt, build_tools, tools_for_intent
from tools.robot_vision import _to_english
from tools.tool_calls import request_tool_call

CANT_SEE = [
    {"role": "user", "content": "What do you see?"},
    {
        "role": "assistant",
        "content": "I don't have direct access to visual information. How can I help?",
    },
]
RED_PENCIL = [
    {"role": "user", "content": "¿Dónde está el lápiz rojo?"},
    {"role": "assistant", "content": "El lápiz rojo está en el centro, a la izquierda."},
]
RED_PENCIL_EN = [
    {"role": "user", "content": "Where is the red pencil?"},
    {"role": "assistant", "content": "It's in the middle, on the left."},
]
MADRID = [
    {"role": "user", "content": "¿Qué tiempo hace en Madrid?"},
    {"role": "assistant", "content": "En Madrid hace veinte grados y está soleado."},
]


def found(name, color=""):
    """Check for find_object: the English name/color the camera would receive."""

    def check(args, model):
        got = _to_english(str(args.get("name") or ""), str(args.get("color") or ""), model)
        return got == (name, color), f"camera gets {got}"

    return check


def app(app_id):
    def check(args, model):
        return args.get("app_id") == app_id, f"app_id={args.get('app_id')!r}"

    return check


def number(expected, tolerance=0.01):
    """Check for the calculator: run the code and compare the result."""

    def check(args, model):
        output = exec_python_repl(str(args.get("code") or "")).strip()
        values = re.findall(r"-?\d+(?:\.\d+)?", output)
        ok = bool(values) and abs(float(values[-1]) - expected) <= tolerance
        return ok, f"result={output[:30]!r}"

    return check


def contains(key, *words):
    def check(args, model):
        value = str(args.get(key) or "").lower()
        return all(w in value for w in words), f"{key}={value!r}"

    return check


def anything(args, model):
    return True, ""


# (intent, history, user text, expected tool, argument check)
CASES = [
    ("vision", CANT_SEE, "What do you see?", "describe_scene", anything),
    ("vision", CANT_SEE, "¿Qué ves?", "describe_scene", anything),
    ("vision", [], "¿Hay una taza delante de la cámara?", "find_object", found("cup")),
    ("vision", [], "¿Dónde está el lápiz rojo?", "find_object", found("pencil", "red")),
    ("vision", [], "¿Ves una botella verde?", "find_object", found("bottle", "green")),
    ("vision", [], "Busca las tijeras", "find_object", found("scissors")),
    ("vision", [], "Where is the blue cup?", "find_object", found("cup", "blue")),
    ("vision", RED_PENCIL, "¿Y el azul?", "find_object", found("pencil", "blue")),
    ("vision", RED_PENCIL_EN, "And the green one?", "find_object", found("pencil", "green")),
    ("apps", [], "Abre Visual Studio Code", "launch_app", app("vscode")),
    ("apps", [], "Open the browser", "launch_app", app("browser")),
    ("apps", [], "Abre una terminal", "launch_app", app("terminal")),
    ("apps", [], "Abre Cursor en el proyecto robot-vision", "launch_app", app("cursor")),
    ("web", [], "¿Qué tiempo hace en Madrid?", "web_search", contains("query", "madrid")),
    ("web", MADRID, "¿Y en Sevilla?", "web_search", contains("query", "sevilla")),
    ("docker", [], "Reinicia el contenedor de postgres", "manage_docker_container",
     contains("container", "postgres")),
    ("memory", [], "Recuerda que la IP del ESP32 es 192.168.1.40", "remember_fact",
     contains("fact", "192.168.1.40")),
    ("calc", [], "¿Cuánto es el quince por ciento de doscientos treinta?", "exec_python_repl",
     number(34.5)),
    ("calc", [], "Convierte cien grados Fahrenheit a Celsius", "exec_python_repl",
     number(37.78)),
    ("calc", [], "¿Cuánto es 2 más 2?", "exec_python_repl", number(4)),
    ("calc", [], "Si tengo tres servos de doce euros y dos de veinte, ¿cuánto gasto?",
     "exec_python_repl", number(76)),
]


def main() -> int:
    args = parse_args(__doc__.splitlines()[0], "LLM_MODEL")
    num_ctx = engine_number("num_ctx")
    print(f"model: {args.model}  num_ctx: {num_ctx}")

    schema, _ = build_tools(
        allowed_root=ENGINE_DIR,
        llm_model=args.model,
        cursor_workspace=ENGINE_DIR,
        cursor_model="eval",
    )
    system = build_system_prompt(ENGINE_DIR, "Spanish")

    def check(case):
        intent, history, text, expected_tool, check_args = case
        messages = (
            [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": text}]
        )
        response = request_tool_call(
            messages, tools_for_intent(intent, schema), args.model, num_ctx
        )
        calls = response["message"].get("tool_calls") or []
        if not calls:
            reply = (response["message"].get("content") or "")[:50]
            return False, f"{text!r:<50} -> NO TOOL: {reply!r}"

        function = calls[0]["function"]
        tool_args = dict(function.get("arguments") or {})
        if function["name"] != expected_tool:
            return False, f"{text!r:<50} -> {function['name']} (expected {expected_tool})"

        ok, detail = check_args(tool_args, args.model)
        return ok, f"{text!r:<50} -> {function['name']} {tool_args} {detail}"

    board = Scoreboard()
    board.run_group("tools turn", CASES, check)
    return board.summary()


if __name__ == "__main__":
    raise SystemExit(main())

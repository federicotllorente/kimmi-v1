"""Eval of the intent routing: small-talk shortcut + LLM classifier.

Run from engine/:
    proj-env\\Scripts\\python.exe tests\\eval_intent.py
    proj-env\\Scripts\\python.exe tests\\eval_intent.py --model qwen2.5:3b

"Similar to prompt examples" cases look like the few-shot examples in
tools/intent.py; "unseen" cases do not. If only the first group improves after
a prompt change, the prompt is memorizing examples instead of generalizing.
"""
from __future__ import annotations

import _common  # noqa: F401  (sets sys.path)
from _common import Scoreboard, parse_args

from locale_config import set_locale
from tools.intent import classify_intent, looks_like_small_talk

# (locale, text, expected): True = answered by chat without calling the classifier
SMALL_TALK_CASES = [
    ("en", "Can you hear me?", True),
    ("en", "Do you hear me now?", True),
    ("en", "Hey Kimmi, are you there?", True),
    ("en", "Hello!", True),
    ("en", "Thanks", True),
    ("en", "Can you hear me open the browser?", False),
    ("en", "Is there a cup?", False),
    ("en", "Hear me out, I need a timer", False),
    ("es", "¿Me escuchas?", True),
    ("es", "¿Me oyes bien?", True),
    ("es", "Hola Kimmi, ¿me escuchas?", True),
    ("es", "¿Estás ahí?", True),
    ("es", "Hola", True),
    ("es", "¿Hay una taza?", False),
    ("es", "¿Qué ves?", False),
    ("es", "Ábrelo", False),
]

PROMPT_LIKE_CASES = [
    ("Can you hear me?", "chat"),
    ("Do you hear me?", "chat"),
    ("I asked you if you can hear me.", "chat"),
    ("¿Me escuchas?", "chat"),
    ("¿Estás ahí?", "chat"),
    ("Hola Kimmi, ¿qué tal?", "chat"),
    ("What is the capital of France?", "chat"),
    ("Explain how a servo motor works", "chat"),
    ("Abre Visual Studio Code", "apps"),
    ("Open the browser", "apps"),
    ("¿Qué tiempo hace en Madrid?", "web"),
    ("¿Cuánto es el quince por ciento de doscientos treinta?", "calc"),
    ("Remember that the ESP32 IP is 192.168.1.40", "memory"),
    ("Lista los archivos de la carpeta notes", "files"),
    ("¿Qué ves en la mesa?", "vision"),
    ("Where is the red pencil?", "vision"),
    ("¿Hay alguna taza delante de la cámara?", "vision"),
    ("Reinicia el contenedor de postgres", "docker"),
    ("y el... eh...", "clarify"),
]

UNSEEN_CASES = [
    ("Kimmi, ¿sigues ahí? No te oigo responder", "chat"),
    ("Tell me a joke about robots", "chat"),
    ("¿Qué diferencia hay entre un ESP32 y un Arduino?", "chat"),
    ("Who wrote Don Quixote?", "chat"),
    ("Lanza Cursor en la carpeta del proyecto", "apps"),
    ("Pon el portapapeles en blanco", "clipboard"),
    ("Search the web for the latest ESP32-S3 price", "web"),
    ("¿Cuántos grados hay ahora en Barcelona?", "web"),
    ("Convierte cien grados Fahrenheit a Celsius", "calc"),
    ("Guarda en memoria que el servo del codo va al pin 14", "memory"),
    ("Show me the logs of the mosquitto container", "docker"),
    ("¿Hay una taza?", "vision"),
    ("¿Ves algún lápiz rojo?", "vision"),
    ("What objects are in front of the robot?", "vision"),
    ("Investiga los mejores servos para brazos robóticos", "research"),
    ("eh... lo de... bueno", "clarify"),
]

# The loop from the logs: Kimmi's canned reply in the classifier context
LOOP_CASES = [
    (
        [
            {"role": "user", "content": "Can you hear me?"},
            {
                "role": "assistant",
                "content": "I'm not sure what you asked. Could you put it in one clear sentence?",
            },
        ],
        "I asked you if you can hear me.",
        "chat",
    ),
]


def main() -> int:
    args = parse_args(__doc__.splitlines()[0], "FAST_MODEL")
    print(f"model: {args.model}")
    board = Scoreboard()

    def check_small_talk(case):
        locale, text, expected = case
        set_locale(locale)
        got = looks_like_small_talk(text, [])
        return got == expected, f"[{locale}] {str(got):<5} {text}"

    def check_classifier(case):
        text, expected = case
        result = classify_intent(text, model=args.model)
        return (
            result.intent == expected,
            f"{result.intent:<9} ({result.confidence:.2f}) expected={expected:<9} {text}",
        )

    def check_loop(case):
        history, text, expected = case
        result = classify_intent(text, model=args.model, recent_context=history)
        return result.intent == expected, f"{result.intent:<9} expected={expected:<9} {text}"

    board.run_group("small-talk shortcut (no LLM)", SMALL_TALK_CASES, check_small_talk)
    set_locale("en")
    board.run_group("classifier: similar to prompt examples", PROMPT_LIKE_CASES, check_classifier)
    board.run_group("classifier: unseen phrases", UNSEEN_CASES, check_classifier)
    board.run_group("classifier: canned reply in history", LOOP_CASES, check_loop)
    return board.summary()


if __name__ == "__main__":
    raise SystemExit(main())

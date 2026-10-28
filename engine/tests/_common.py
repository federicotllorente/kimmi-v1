"""Shared helpers for the eval scripts in engine/tests.

The evals call the real Ollama models, so they measure behavior, not just
code: run them before and after touching a prompt, a tool schema or a model,
and compare the scores instead of judging by ear.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Callable, List, Tuple

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

# The console may be cp1252 on Windows: never crash on "¿" or "á"
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def engine_setting(name: str) -> str:
    """Read a string constant such as FAST_MODEL from assistant.py.

    Importing assistant.py would open the microphone and load Whisper and TTS,
    so the value is read from the source instead.
    """
    with open(os.path.join(ENGINE_DIR, "assistant.py"), encoding="utf-8") as f:
        match = re.search(rf'^{name}\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"{name} not found in assistant.py")
    return match.group(1)


def engine_number(name: str) -> int:
    """Read an int setting like "num_ctx" from LLM_CHAT_OPTIONS in assistant.py."""
    with open(os.path.join(ENGINE_DIR, "assistant.py"), encoding="utf-8") as f:
        match = re.search(rf'"{name}":\s*(\d+)', f.read())
    if not match:
        raise RuntimeError(f"{name} not found in assistant.py")
    return int(match.group(1))


def disable_thinking() -> None:
    """Make every ollama.chat call in this process pass think=False.

    "Thinking" models (qwen3, ...) reason before answering by default, which
    adds seconds per call: too slow for voice. Models without that mode
    accept think=False and ignore it.
    """
    import ollama

    original_chat = ollama.chat

    def chat_without_thinking(*args, **kwargs):
        kwargs.setdefault("think", False)
        return original_chat(*args, **kwargs)

    ollama.chat = chat_without_thinking


def parse_args(description: str, default_model_setting: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=engine_setting(default_model_setting),
        help=f"Ollama model to test (default: {default_model_setting} from assistant.py)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Let thinking models (qwen3...) reason before answering (slower)",
    )
    args = parser.parse_args()
    if not args.think:
        disable_thinking()
    return args


class Scoreboard:
    """Prints one line per case and a summary per group."""

    def __init__(self) -> None:
        self.groups: List[Tuple[str, int, int, float]] = []

    def run_group(self, title: str, cases: list, check: Callable) -> None:
        """check(case) -> (ok: bool, detail: str)"""
        print(f"\n== {title}")
        passed, total_time = 0, 0.0
        for case in cases:
            start = time.time()
            ok, detail = check(case)
            elapsed = time.time() - start
            total_time += elapsed
            passed += bool(ok)
            print(f"  {'OK ' if ok else 'BAD'} {elapsed:5.2f}s  {detail}")
        self.groups.append((title, passed, len(cases), total_time))
        print(f"  -> {passed}/{len(cases)}")

    def summary(self) -> int:
        print("\n== Summary")
        failed = 0
        for title, passed, total, total_time in self.groups:
            avg = total_time / total if total else 0.0
            print(f"  {passed:>3}/{total:<3} avg {avg:4.2f}s  {title}")
            failed += total - passed
        return 1 if failed else 0

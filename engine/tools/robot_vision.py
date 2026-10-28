"""Read what the robot-vision camera sees, through MQTT.

The vision program (robot-vision/main.py) publishes its latest detections,
retained, on robot/vision/objects and listens on robot/vision/targets for
the list of object names it should look for (YOLO-World open vocabulary).

These tools are read-only: they never move anything.
Requires: paho-mqtt, ollama (to translate object names to English).
"""
from __future__ import annotations

import json
import os
import queue
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore

OBJECTS_TOPIC = "robot/vision/objects"
TARGETS_TOPIC = "robot/vision/targets"

# Detections older than this mean the vision program is not running
# (the retained message survives after it exits).
STALE_AFTER_SECONDS = 3.0

# How long to wait for the first (retained) detections message
READ_TIMEOUT_SECONDS = 2.0

# How long to wait for the vision program to apply new targets
TARGETS_TIMEOUT_SECONDS = 5.0

NO_COLOR_VALUES = frozenset({"nil", "none", "null", "any", "unknown", "n/a"})

# The detector (YOLO-World) understands English names, but the 7B model often
# passes the user's words as spoken ("taza", "lapiz") even when told not to.
# A short structured call translates them; measured ~0.2 s, 12/12 correct.
_TRANSLATE_SYSTEM = (
    "Translate an object name and an optional color into English for an object detector. "
    "Object name: English, singular, lowercase, common noun (e.g. taza -> cup, lapiz -> pencil). "
    "Color: one of red, orange, yellow, green, blue, purple, or empty string if none. "
    "If a value is already English, keep it."
)
_TRANSLATE_FORMAT: Dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "color": {"type": "string"}},
    "required": ["name", "color"],
}
_translation_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


class _VisionLink:
    """Short-lived MQTT connection subscribed to the detections topic."""

    def __init__(self) -> None:
        self.messages: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"kimmi-vision-{uuid.uuid4().hex[:8]}",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def __enter__(self) -> "_VisionLink":
        host = _env("KIMMI_MQTT_HOST", "127.0.0.1")
        port = int(_env("KIMMI_MQTT_PORT", "1883"))
        self.client.connect(host, port, keepalive=10)
        self.client.loop_start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if not reason_code.is_failure:
            client.subscribe(OBJECTS_TOPIC)

    def _on_message(self, client, userdata, message) -> None:
        try:
            state = json.loads(message.payload)
        except json.JSONDecodeError:
            return
        if isinstance(state, dict):
            self.messages.put(state)

    def wait_for(
        self,
        accept: Callable[[Dict[str, Any]], bool],
        timeout: float,
    ) -> Optional[Dict[str, Any]]:
        """First detections message that passes accept(), or None on timeout."""
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                state = self.messages.get(timeout=remaining)
            except queue.Empty:
                return None
            if accept(state):
                return state

    def publish_targets(self, targets: List[str]) -> None:
        self.client.publish(TARGETS_TOPIC, json.dumps(targets), qos=1)


def _is_fresh(state: Dict[str, Any]) -> bool:
    try:
        return time.time() - float(state.get("ts", 0)) <= STALE_AFTER_SECONDS
    except (TypeError, ValueError):
        return False


def _summarize(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Only what the LLM needs to answer out loud
    return [
        {
            "name": obj.get("name"),
            "color": obj.get("color"),
            "position": obj.get("cell"),
            "confidence": obj.get("confidence"),
        }
        for obj in objects
    ]


def _read_current_state(link: _VisionLink) -> Optional[Dict[str, Any]]:
    return link.wait_for(_is_fresh, READ_TIMEOUT_SECONDS)


def _offline_result() -> str:
    return _result({
        "error": "vision_offline",
        "message": (
            "The robot camera program is not running or not publishing. "
            "Tell the user to start robot-vision/main.py."
        ),
    })


def _mqtt_error_result(exc: Exception) -> str:
    return _result({
        "error": "mqtt_unavailable",
        "message": f"Could not reach the MQTT broker: {exc}",
    })


def describe_scene() -> str:
    """List every object the camera currently detects."""
    if mqtt is None:
        return _result({"error": "paho-mqtt not installed"})

    try:
        with _VisionLink() as link:
            state = _read_current_state(link)
    except OSError as exc:
        return _mqtt_error_result(exc)

    if state is None:
        return _offline_result()

    return _result({
        "status": "ok",
        "looking_for": state.get("targets", []),
        "objects": _summarize(state.get("objects", [])),
        "note": (
            "Positions are cells of a 3x3 grid as seen by the camera "
            "(TOP/MIDDLE/BOTTOM - LEFT/CENTER/RIGHT)."
        ),
    })


def _to_english(name: str, color: str, model: str) -> Tuple[str, str]:
    """Translate (name, color) to English. Falls back to the input on any error."""
    key = (name, color)
    if key in _translation_cache:
        return _translation_cache[key]

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps({"name": name, "color": color}, ensure_ascii=False),
                },
            ],
            format=_TRANSLATE_FORMAT,
            keep_alive="24h",
            options={"num_ctx": 4096, "temperature": 0},
        )
        data = json.loads(response["message"]["content"])
        translated = (
            str(data.get("name") or name).strip().lower(),
            str(data.get("color") or "").strip().lower(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [Vision]: translation failed ({exc}); using {key}")
        return key

    if translated[1] in NO_COLOR_VALUES:
        translated = (translated[0], "")
    _translation_cache[key] = translated
    return translated


def find_object(name: str, color: str = "", llm_model: str = "qwen2.5:7b") -> str:
    """Find one kind of object, optionally of a given color."""
    if mqtt is None:
        return _result({"error": "paho-mqtt not installed"})

    name = (name or "").strip().lower()
    color = (color or "").strip().lower()

    # Small models sometimes fill an optional argument with "nil" or "none"
    # instead of leaving it out: treat that as "no color requested".
    if color in NO_COLOR_VALUES:
        color = ""

    if not name:
        return _result({"error": "missing_name", "message": "Say which object to find."})

    name, color = _to_english(name, color, llm_model)

    try:
        with _VisionLink() as link:
            state = _read_current_state(link)
            if state is None:
                return _offline_result()

            targets = [str(t).lower() for t in state.get("targets", [])]

            # The vision model only detects the names in its target list,
            # so ask it to also look for this one and wait until it does.
            if name not in targets:
                link.publish_targets(targets + [name])
                state = link.wait_for(
                    lambda s: _is_fresh(s)
                    and name in [str(t).lower() for t in s.get("targets", [])],
                    TARGETS_TIMEOUT_SECONDS,
                )
                if state is None:
                    return _result({
                        "error": "targets_not_applied",
                        "message": (
                            f"The camera did not start looking for '{name}'. "
                            "It may be using the fixed COCO model."
                        ),
                    })
                # The first frame with the new targets may still be settling:
                # take one more message so the detections are fresh.
                state = link.wait_for(_is_fresh, READ_TIMEOUT_SECONDS) or state
    except OSError as exc:
        return _mqtt_error_result(exc)

    same_name = [
        obj for obj in state.get("objects", [])
        if str(obj.get("name", "")).lower() == name
    ]
    matches = [
        obj for obj in same_name
        if not color or str(obj.get("color", "")).lower() == color
    ]

    if matches:
        return _result({"status": "found", "matches": _summarize(matches)})

    if same_name:
        # Found the object, but not in the requested color
        return _result({
            "status": "found_other_color",
            "requested_color": color,
            "matches": _summarize(same_name),
        })

    return _result({"status": "not_found", "name": name, "color": color})

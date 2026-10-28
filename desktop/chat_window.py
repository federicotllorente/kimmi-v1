"""Kimmi's chat window: the conversation plus a card for every tool call.

Fed by MQTT (see engine/mqtt_bus.py):
  kimmi/chat/message  -> a bubble (you / Kimmi)
  kimmi/chat/intent   -> a small "router" line: why Kimmi chose what it did
  kimmi/tools/call    -> a tool card, updated in place from RUNNING to DONE/ERROR
"""
from __future__ import annotations

import html
import json
import time
from typing import Any, Dict

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import theme

# Oldest items are removed beyond this, so a long session stays light
MAX_ITEMS = 200

STATUS_COLORS = {
    "running": theme.AMBER,
    "done": theme.TEAL,
    "error": theme.ERROR,
    "cancelled": theme.MUTED,
}

# Result preview when the card is collapsed
RESULT_PREVIEW_CHARS = 140

SCROLLBAR_CSS = f"""
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {theme.css(theme.CYAN, 90)}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""


def make_label(
    text: str,
    font: QFont,
    color: QColor,
    wrap: bool = False,
    selectable: bool = False,
) -> QLabel:
    label = QLabel(text)
    label.setFont(font)
    label.setStyleSheet(f"color: {theme.css(color)}; background: transparent;")
    label.setWordWrap(wrap)
    if selectable:
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class HudFrame(QFrame):
    """Panel painted like the robot-vision HUD: dark fill, thin border, bright corners."""

    def __init__(self, accent: QColor = theme.CYAN, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._accent = QColor(accent)

    def set_accent(self, color: QColor) -> None:
        self._accent = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)

        fill = QColor(theme.PANEL)
        fill.setAlpha(215)
        painter.fillRect(rect, fill)

        border = QColor(theme.MUTED)
        border.setAlpha(80)
        painter.setPen(QPen(border, 1))
        painter.drawRect(rect)

        # The four "[ ]" corners
        painter.setPen(QPen(self._accent, 2))
        n = min(12.0, rect.width() / 4, rect.height() / 4)
        for x, y, dx, dy in (
            (rect.left(), rect.top(), 1, 1),
            (rect.right(), rect.top(), -1, 1),
            (rect.left(), rect.bottom(), 1, -1),
            (rect.right(), rect.bottom(), -1, -1),
        ):
            painter.drawLine(QPointF(x, y), QPointF(x + dx * n, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + dy * n))
        painter.end()


class MessageItem(QWidget):
    """One chat bubble: yours on the right, Kimmi's on the left."""

    def __init__(self, role: str, text: str, ts: float) -> None:
        super().__init__()
        is_user = role == "user"

        bubble = HudFrame(theme.CYAN if is_user else theme.TEAL)
        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(3)

        who = "YOU" if is_user else "KIMMI"
        clock = time.strftime("%H:%M", time.localtime(ts))
        layout.addWidget(make_label(f"{who}  ·  {clock}", theme.title_font(9), theme.MUTED))
        layout.addWidget(make_label(text, theme.body_font(15, 500), theme.TEXT, wrap=True, selectable=True))

        # The empty stretch on the other side pushes the bubble left or right
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        if is_user:
            row.addStretch(1)
            row.addWidget(bubble, 5)
        else:
            row.addWidget(bubble, 5)
            row.addStretch(1)


class IntentItem(QWidget):
    """Small line: which intent the router chose, and why."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__()
        intent = str(payload.get("intent") or "?")
        confidence = payload.get("confidence")
        reason = str(payload.get("reason") or "")

        detail = intent if confidence is None else f"{intent}  {float(confidence):.2f}"
        if reason:
            detail += f"  —  {reason}"

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)
        tag = make_label("ROUTER", theme.title_font(8), theme.CYAN)
        tag.setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(tag)
        row.addWidget(make_label(detail, theme.body_font(13, 500), theme.MUTED, wrap=True), 1)


class ToolCard(HudFrame):
    """A tool call. Created on RUNNING and updated in place when it finishes."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__(theme.AMBER)
        self._expanded = False
        self._result = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(make_label("TOOL", theme.title_font(8), theme.MUTED))
        self.name = make_label("", theme.title_font(11), theme.CYAN)
        top.addWidget(self.name)
        top.addStretch(1)
        self.duration = make_label("", theme.body_font(13), theme.MUTED)
        top.addWidget(self.duration)
        self.status = QLabel()
        self.status.setFont(theme.title_font(8))
        top.addWidget(self.status)
        layout.addLayout(top)

        self.arguments = make_label("", theme.body_font(14, 500), theme.TEXT, wrap=True, selectable=True)
        self.arguments.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.arguments)

        self.toggle = QToolButton()
        self.toggle.setFont(theme.title_font(8))
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.setStyleSheet(
            f"QToolButton {{ color: {theme.css(theme.CYAN)}; background: transparent; border: none; padding: 0; }}"
        )
        self.toggle.clicked.connect(self._toggle_result)
        layout.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignLeft)

        self.result = make_label("", theme.mono_font(12), theme.TEXT, wrap=True, selectable=True)
        layout.addWidget(self.result)

        self.update_call(payload)

    def update_call(self, payload: Dict[str, Any]) -> None:
        status = str(payload.get("status") or "running")
        color = STATUS_COLORS.get(status, theme.MUTED)
        self.set_accent(color)

        self.name.setText(str(payload.get("name") or "?"))
        self.status.setText(status.upper())
        self.status.setStyleSheet(
            f"color: {theme.css(color)}; border: 1px solid {theme.css(color)};"
            " border-radius: 3px; padding: 1px 6px; background: transparent;"
        )

        duration = payload.get("duration_ms")
        if duration is None:
            self.duration.setText("")
        elif duration < 1000:
            self.duration.setText(f"{duration} ms")
        else:
            self.duration.setText(f"{duration / 1000:.1f} s")

        self.arguments.setText(self._format_arguments(payload.get("arguments") or {}))

        self._result = self._pretty(str(payload.get("result") or ""))
        # Errors open by themselves: that is when you want to read the result
        if status == "error":
            self._expanded = True
        self._refresh_result()

    @staticmethod
    def _format_arguments(arguments: Dict[str, Any]) -> str:
        if not arguments:
            return f"<span style='color:{theme.css(theme.MUTED)}'>no arguments</span>"
        rows = []
        for key, value in arguments.items():
            shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            rows.append(
                f"<span style='color:{theme.css(theme.MUTED)}'>{html.escape(str(key))}</span>"
                f"&nbsp;&nbsp;{html.escape(str(shown))}"
            )
        return "<br>".join(rows)

    @staticmethod
    def _pretty(result: str) -> str:
        # JSON results (most tools) are easier to read indented
        try:
            return json.dumps(json.loads(result), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return result

    def _toggle_result(self) -> None:
        self._expanded = not self._expanded
        self._refresh_result()

    def _refresh_result(self) -> None:
        has_result = bool(self._result)
        self.toggle.setVisible(has_result)
        self.result.setVisible(has_result)
        if not has_result:
            return

        if self._expanded:
            self.toggle.setText("HIDE RESULT")
            self.result.setText(self._result)
        else:
            self.toggle.setText("SHOW RESULT")
            preview = " ".join(self._result.split())
            if len(preview) > RESULT_PREVIEW_CHARS:
                preview = preview[: RESULT_PREVIEW_CHARS - 1] + "…"
            self.result.setText(preview)


class HeaderBar(QWidget):
    """Top bar: KIMMI · CHAT on the left, pipeline state on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(46)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 14, 2)
        row.setSpacing(10)
        row.addWidget(make_label("KIMMI", theme.title_font(15, 800), theme.CYAN))
        row.addWidget(make_label("CHAT", theme.title_font(11, 600), theme.MUTED))
        row.addStretch(1)
        self.state = make_label("", theme.title_font(9), theme.MUTED)
        row.addWidget(self.state)

    def set_state(self, text: str, color: QColor) -> None:
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {theme.css(color)}; background: transparent;")

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.PANEL)
        painter.setPen(QPen(theme.CYAN, 1.5))
        bottom = self.height() - 1
        painter.drawLine(0, bottom, self.width(), bottom)
        painter.end()


class ChatWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KIMMI · CHAT")
        self.resize(440, 640)
        self.setMinimumSize(320, 360)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, theme.DEEP)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self._state = ""
        self._mqtt_connected = True
        self._tool_cards: Dict[str, ToolCard] = {}
        self._stick_to_bottom = True

        self.header = HeaderBar()

        self.items = QWidget()
        self.items.setStyleSheet("background: transparent;")
        self.items_layout = QVBoxLayout(self.items)
        self.items_layout.setContentsMargins(12, 12, 12, 12)
        self.items_layout.setSpacing(10)
        self.empty = make_label(
            "Say “Hey Kimmi” to start.\nYour conversation and every tool Kimmi uses will show up here.",
            theme.body_font(15, 500),
            theme.MUTED,
            wrap=True,
        )
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.items_layout.addWidget(self.empty)
        self.items_layout.addStretch(1)  # keeps items at the top

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(SCROLLBAR_CSS)
        self.scroll.viewport().setStyleSheet("background: transparent;")
        self.scroll.setWidget(self.items)

        bar = self.scroll.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        bar.rangeChanged.connect(self._on_range_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.scroll, 1)

        self.set_status("starting", 0.0, "")

    # ------------------------------------------------------------ public slots
    def add_message(self, role: str, text: str, turn_id: int = 0) -> None:
        self._add(MessageItem(role, text, time.time()))

    def add_intent(self, payload: Dict[str, Any]) -> None:
        self._add(IntentItem(payload))

    def update_tool(self, payload: Dict[str, Any]) -> None:
        call_id = str(payload.get("call_id") or "")
        card = self._tool_cards.get(call_id)
        if card is not None:
            card.update_call(payload)
            return
        card = ToolCard(payload)
        card.setProperty("call_id", call_id)
        self._tool_cards[call_id] = card
        self._add(card)

    def set_status(self, state: str, amplitude: float = 0.0, message: str = "") -> None:
        # The bridge emits ~30 times per second (amplitude): only react to changes
        if state == self._state:
            return
        self._state = state
        self._refresh_header()

    def set_connected(self, connected: bool) -> None:
        self._mqtt_connected = bool(connected)
        self._refresh_header()

    # ------------------------------------------------------------ internals
    def _refresh_header(self) -> None:
        if not self._mqtt_connected:
            # The chat only works over MQTT
            self.header.set_state("MQTT OFF · NO CHAT", theme.ERROR)
            return
        label = theme.STATE_LABELS.get(self._state, self._state.upper())
        color = {
            "listening": theme.CYAN,
            "responding": theme.CYAN,
            "processing": theme.AMBER,
            "error": theme.ERROR,
        }.get(self._state, theme.MUTED)
        self.header.set_state(label, color)

    def _add(self, widget: QWidget) -> None:
        self.empty.hide()
        # Insert before the final stretch
        self.items_layout.insertWidget(self.items_layout.count() - 1, widget)
        self._trim()

    def _trim(self) -> None:
        # Index 0 is the (hidden) empty-state label, the last item is the stretch
        while self.items_layout.count() - 2 > MAX_ITEMS:
            item = self.items_layout.takeAt(1)
            widget = item.widget()
            if widget is None:
                continue
            call_id = widget.property("call_id")
            if call_id:
                self._tool_cards.pop(str(call_id), None)
            widget.deleteLater()

    def _on_scrolled(self, value: int) -> None:
        # Follow new messages only if you are already at the bottom: if you
        # scrolled up to read something, the window does not jump
        bar = self.scroll.verticalScrollBar()
        self._stick_to_bottom = value >= bar.maximum() - 24

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._stick_to_bottom:
            self.scroll.verticalScrollBar().setValue(maximum)

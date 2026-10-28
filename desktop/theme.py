"""Shared look for Kimmi's desktop window.

Keep the palette in sync with robot-vision/ui_theme.py, so both windows look
like parts of the same HUD.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

# ---------------------------------------------------------------- palette
DEEP = QColor(2, 8, 15)  # background
PANEL = QColor(8, 22, 40)  # cards / bars
CYAN = QColor(0, 212, 255)  # main accent (the wave)
GLOW = QColor(0, 180, 255)  # halo around the wave
TEAL = QColor(46, 196, 182)  # "ok" / active (tray icon)
AMBER = QColor(255, 190, 0)  # warnings
ERROR = QColor(255, 64, 96)  # errors
TEXT = QColor(232, 246, 255)  # main text
MUTED = QColor(120, 165, 200)  # secondary text

# Pipeline state -> label shown in the wave window and the chat header
STATE_LABELS = {
    "starting": "BOOTING",
    "idle": "STANDBY",
    "listening": "LISTENING",
    "processing": "THINKING",
    "responding": "SPEAKING",
    "error": "ERROR",
}


def css(color: QColor, alpha: int | None = None) -> str:
    """QColor -> "rgba(r, g, b, a)" for Qt style sheets."""
    a = color.alpha() if alpha is None else alpha
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {a})"


# ---------------------------------------------------------------- fonts
FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

# Orbitron: titles and labels (sci-fi HUD look). Rajdhani: readable body text.
_FONT_FILES = ("Orbitron-Variable.ttf", "Rajdhani-Medium.ttf", "Rajdhani-SemiBold.ttf")
_families: dict[str, str] = {}


def load_fonts() -> None:
    """Register the bundled fonts with Qt. Call once, after QApplication exists."""
    for name in _FONT_FILES:
        font_id = QFontDatabase.addApplicationFont(str(FONTS_DIR / name))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        if families:
            _families[name.split("-")[0]] = families[0]
        else:
            print(f"[desktop] font not loaded: {name}", flush=True)


def title_font(pixel_size: float, weight: int = 700) -> QFont:
    font = QFont(_families.get("Orbitron", "Segoe UI"))
    font.setPixelSize(max(int(pixel_size), 1))
    font.setWeight(QFont.Weight(weight))
    # Wide letter spacing: the classic HUD label look
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 118)
    return font


def body_font(pixel_size: float, weight: int = 600) -> QFont:
    font = QFont(_families.get("Rajdhani", "Segoe UI"))
    font.setPixelSize(max(int(pixel_size), 1))
    font.setWeight(QFont.Weight(weight))
    return font


def mono_font(pixel_size: float) -> QFont:
    """For tool results (JSON): fixed width keeps the indentation readable."""
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(max(int(pixel_size), 1))
    return font

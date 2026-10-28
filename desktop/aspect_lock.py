"""Keep a window's proportions while the user resizes it (Windows only).

While an edge is dragged, Windows sends WM_SIZING with the rectangle the
window is about to take. Fixing that rectangle in place keeps the inner
(client) area at a fixed width/height ratio, so the content scales without
being squashed. The same idea is used in robot-vision/aspect_window.py.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

WM_SIZING = 0x0214

# Which edge or corner is being dragged (wParam of WM_SIZING)
WMSZ_LEFT, WMSZ_RIGHT, WMSZ_TOP = 1, 2, 3
WMSZ_TOPLEFT, WMSZ_TOPRIGHT, WMSZ_BOTTOM = 4, 5, 6
WMSZ_BOTTOMLEFT, WMSZ_BOTTOMRIGHT = 7, 8

if sys.platform == "win32":
    _user32 = ctypes.windll.user32
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]


def _border(hwnd: int) -> tuple[int, int]:
    """Title bar + borders = outer size minus inner (client) size."""
    outer, inner = wintypes.RECT(), wintypes.RECT()
    _user32.GetWindowRect(hwnd, ctypes.byref(outer))
    _user32.GetClientRect(hwnd, ctypes.byref(inner))
    return (
        (outer.right - outer.left) - (inner.right - inner.left),
        (outer.bottom - outer.top) - (inner.bottom - inner.top),
    )


def handle_sizing(message_address: int, aspect: float, min_width: int) -> bool:
    """Call from QWidget.nativeEvent. True if it was a WM_SIZING it fixed.

    The window handle is read from the message itself: calling
    QWidget.winId() inside nativeEvent can create the native window, which
    sends more messages to nativeEvent, recursing until it crashes.
    """
    if sys.platform != "win32":
        return False

    msg = wintypes.MSG.from_address(message_address)
    if msg.message != WM_SIZING:
        return False

    edge = msg.wParam
    rect = wintypes.RECT.from_address(msg.lParam)
    border_w, border_h = _border(msg.hWnd)

    if edge in (WMSZ_TOP, WMSZ_BOTTOM):
        # Dragging top/bottom: the height decides, the width follows
        inner_h = max(rect.bottom - rect.top - border_h, int(min_width / aspect))
        inner_w = int(inner_h * aspect)
    else:
        # Sides and corners: the width decides, the height follows
        inner_w = max(rect.right - rect.left - border_w, min_width)
        inner_h = int(inner_w / aspect)

    # Move only the edges being dragged; the opposite ones stay put
    if edge in (WMSZ_LEFT, WMSZ_TOPLEFT, WMSZ_BOTTOMLEFT):
        rect.left = rect.right - inner_w - border_w
    else:
        rect.right = rect.left + inner_w + border_w

    if edge in (WMSZ_TOP, WMSZ_TOPLEFT, WMSZ_TOPRIGHT):
        rect.top = rect.bottom - inner_h - border_h
    else:
        rect.bottom = rect.top + inner_h + border_h

    return True

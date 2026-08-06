"""Lee o escribe texto en el portapapeles del sistema.

Acciones: read | write | append | clear | history.
Preferencia: pyperclip si está instalado; fallback Win32 vía ctypes.
Requiere: opcionalmente `pip install pyperclip`.
"""
from __future__ import annotations

import platform
import time
from collections import deque
from typing import Deque, Optional

from tools.common import truncate

MAX_READ_CHARS = 2000
MAX_WRITE_BYTES = 50 * 1024
HISTORY_SIZE = 5
HISTORY_ENTRY_CHARS = 200

_HISTORY: Deque[str] = deque(maxlen=HISTORY_SIZE)
_ALLOWED = ("read", "write", "append", "clear", "history")


def manage_clipboard(action: str, text: str = "") -> str:
    """Gestiona el portapapeles (solo I/O; no inyecta teclas)."""
    action = (action or "").strip().lower()
    if action not in _ALLOWED:
        return f"Acción no válida. Usa: {', '.join(_ALLOWED)}."

    if action == "history":
        if not _HISTORY:
            return "Historial de portapapeles vacío."
        lines = []
        for i, item in enumerate(reversed(_HISTORY), start=1):
            lines.append(f"{i}. {truncate(item, HISTORY_ENTRY_CHARS)}")
        return "Historial reciente:\n" + "\n".join(lines)

    if action == "clear":
        err = _set_clipboard("")
        if err:
            return err
        return "Portapapeles vaciado."

    if action == "read":
        content, err = _get_clipboard()
        if err:
            return err
        if content is None or content == "":
            return "El portapapeles está vacío o no contiene texto."
        return truncate(content, MAX_READ_CHARS)

    # write | append
    if text is None:
        return "Debes indicar el texto."

    payload_text = text
    if action == "append":
        current, err = _get_clipboard()
        if err:
            return err
        current = current or ""
        payload_text = current + text

    payload = payload_text.encode("utf-8")
    if len(payload) > MAX_WRITE_BYTES:
        return f"Texto demasiado grande (máximo {MAX_WRITE_BYTES} bytes)."

    err = _set_clipboard(payload_text)
    if err:
        return err

    _HISTORY.append(payload_text)
    verb = "añadido" if action == "append" else "actualizado"
    return f"Portapapeles {verb} ({len(payload)} bytes)."


def _set_clipboard(text: str) -> Optional[str]:
    try:
        import pyperclip

        pyperclip.copy(text)
        return None
    except ImportError:
        pass
    except Exception as e:
        return f"No se pudo escribir el portapapeles (pyperclip): {e}"

    if platform.system() == "Windows":
        return _win_set_clipboard(text)
    return (
        "No hay backend de portapapeles. Instala pyperclip: pip install pyperclip"
    )


def _get_clipboard() -> tuple:
    try:
        import pyperclip

        return pyperclip.paste(), None
    except ImportError:
        pass
    except Exception as e:
        return None, f"No se pudo leer el portapapeles (pyperclip): {e}"

    if platform.system() == "Windows":
        return _win_get_clipboard()
    return None, (
        "No hay backend de portapapeles. Instala pyperclip: pip install pyperclip"
    )


def _win_set_clipboard(text: str) -> Optional[str]:
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.restype = wintypes.BOOL

        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002

        opened = False
        for _ in range(5):
            if user32.OpenClipboard(None):
                opened = True
                break
            time.sleep(0.05)
        if not opened:
            return "No se pudo abrir el portapapeles."
        try:
            user32.EmptyClipboard()
            data = text.encode("utf-16-le") + b"\x00\x00"
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not handle:
                return "GlobalAlloc falló."
            locked = kernel32.GlobalLock(handle)
            if not locked:
                kernel32.GlobalFree(handle)
                return "GlobalLock falló."
            ctypes.memmove(locked, data, len(data))
            kernel32.GlobalUnlock(handle)
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                kernel32.GlobalFree(handle)
                return "SetClipboardData falló."
        finally:
            user32.CloseClipboard()
        return None
    except Exception as e:
        return f"Error Win32 clipboard (write): {e}"


def _win_get_clipboard() -> tuple:
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = wintypes.HANDLE
        user32.CloseClipboard.restype = wintypes.BOOL

        CF_UNICODETEXT = 13

        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None, None

        opened = False
        for _ in range(5):
            if user32.OpenClipboard(None):
                opened = True
                break
            time.sleep(0.05)
        if not opened:
            return None, "No se pudo abrir el portapapeles."
        try:
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return None, None
            locked = kernel32.GlobalLock(handle)
            if not locked:
                return None, "GlobalLock falló al leer el portapapeles."
            try:
                text = ctypes.wstring_at(locked)
            finally:
                kernel32.GlobalUnlock(handle)
            return text, None
        finally:
            user32.CloseClipboard()
    except Exception as e:
        return None, f"Error Win32 clipboard (read): {e}"

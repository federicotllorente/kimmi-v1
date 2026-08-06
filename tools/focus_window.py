"""Enfoca una ventana de una app allowlisted que ya esté abierta.

Windows: ctypes EnumWindows / SetForegroundWindow.
macOS/Linux: mensaje de no implementado (preparado para ampliar).
Requiere: stdlib.
"""
from __future__ import annotations

import platform
from typing import List, Optional, Tuple

from tools.app_allowlist import AppEntry, get_app, list_allowed_app_ids


def focus_window(app_id: str = "", title_substring: str = "") -> str:
    """Enfoca la primera ventana que coincida con app_id y/o title_substring."""
    app_id = (app_id or "").strip().lower()
    title_substring = (title_substring or "").strip()

    if not app_id and not title_substring:
        return "Indica app_id (vscode, cursor, browser, terminal) o title_substring."

    app: Optional[AppEntry] = None
    if app_id:
        app = get_app(app_id)
        if app is None:
            allowed = ", ".join(list_allowed_app_ids())
            return f"App no permitida. Usa una de: {allowed}."

    system = platform.system()
    if system == "Windows":
        return _focus_windows(app, title_substring)
    if system == "Darwin":
        return "focus_window aún no está implementado en macOS."
    return "focus_window aún no está implementado en este sistema."


def _focus_windows(app: Optional[AppEntry], title_substring: str) -> str:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return "ctypes no disponible; no se puede enfocar ventanas."

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
    )
    IsWindowVisible = user32.IsWindowVisible
    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    ShowWindow = user32.ShowWindow
    SetForegroundWindow = user32.SetForegroundWindow
    BringWindowToTop = user32.BringWindowToTop
    AllowSetForegroundWindow = getattr(user32, "AllowSetForegroundWindow", None)

    OpenProcess = kernel32.OpenProcess
    CloseHandle = kernel32.CloseHandle
    QueryFullProcessImageNameW = getattr(kernel32, "QueryFullProcessImageNameW", None)

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SW_RESTORE = 9
    ASFW_ANY = -1

    matches: List[Tuple[int, str]] = []

    def _window_title(hwnd: int) -> str:
        length = GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def _process_basename(hwnd: int) -> str:
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""
        handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not handle:
            return ""
        try:
            if QueryFullProcessImageNameW is None:
                return ""
            size = wintypes.DWORD(260)
            buf = ctypes.create_unicode_buffer(260)
            if QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                path = buf.value
                return path.rsplit("\\", 1)[-1].rsplit(".", 1)[0]
            return ""
        finally:
            CloseHandle(handle)

    def _callback(hwnd, _lparam):
        if not IsWindowVisible(hwnd):
            return True
        title = _window_title(hwnd)
        if not title:
            return True

        if title_substring and title_substring.lower() not in title.lower():
            return True

        if app is not None:
            proc = _process_basename(hwnd)
            allowed_procs = {p.lower() for p in app.process_names}
            title_ok = any(h.lower() in title.lower() for h in app.title_hints)
            proc_ok = proc.lower() in allowed_procs if proc else False
            # Si hay title_substring ya filtramos arriba; aún exigimos app allowlist.
            if not (proc_ok or title_ok):
                return True
        else:
            # Sin app_id: solo title_substring, pero debe mapear a alguna app allowlist.
            from tools.app_allowlist import APP_CATALOG

            allowed = False
            for entry in APP_CATALOG.values():
                proc = _process_basename(hwnd)
                if proc.lower() in {p.lower() for p in entry.process_names}:
                    allowed = True
                    break
                if any(h.lower() in title.lower() for h in entry.title_hints):
                    allowed = True
                    break
            if not allowed:
                return True

        matches.append((int(hwnd), title))
        return True

    EnumWindows(EnumWindowsProc(_callback), 0)

    if not matches:
        label = app.display_name if app else f"título '{title_substring}'"
        return f"No hay ventana abierta coincidente para {label}."

    hwnd, title = matches[0]
    try:
        if AllowSetForegroundWindow is not None:
            AllowSetForegroundWindow(ASFW_ANY)
        ShowWindow(hwnd, SW_RESTORE)
        BringWindowToTop(hwnd)
        SetForegroundWindow(hwnd)
    except Exception as e:
        return f"No se pudo enfocar la ventana: {e}"

    return f"Ventana enfocada: {title}"

"""Allowlist compartida de aplicaciones para launch_app / focus_window.

Windows-first; rutas sobrescribibles por variables de entorno.
Requiere: nada extra (stdlib). En Windows, focus usa ctypes.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AppEntry:
    """Entrada de una app permitida."""

    id: str
    display_name: str
    # Nombres de proceso (sin .exe) usados para enfocar ventanas.
    process_names: tuple
    # Substrings de título típicos (fallback de focus).
    title_hints: tuple
    # Si True, `path` puede ser una carpeta de proyecto.
    accepts_path: bool = False
    # Si True, `path` puede ser una URL http(s).
    accepts_url: bool = False


# Catálogo fijo: solo estas apps pueden lanzarse o enfocarse.
APP_CATALOG: Dict[str, AppEntry] = {
    "vscode": AppEntry(
        id="vscode",
        display_name="VS Code",
        process_names=("Code", "code"),
        title_hints=("Visual Studio Code",),
        accepts_path=True,
    ),
    "cursor": AppEntry(
        id="cursor",
        display_name="Cursor",
        process_names=("Cursor", "cursor"),
        title_hints=("Cursor",),
        accepts_path=True,
    ),
    "browser": AppEntry(
        id="browser",
        display_name="Navegador",
        process_names=("msedge", "chrome", "firefox", "brave"),
        title_hints=("Microsoft Edge", "Google Chrome", "Mozilla Firefox", "Brave"),
        accepts_url=True,
    ),
    "terminal": AppEntry(
        id="terminal",
        display_name="Terminal",
        process_names=("WindowsTerminal", "wt", "powershell", "pwsh", "cmd"),
        title_hints=("Windows Terminal", "PowerShell", "Command Prompt"),
        accepts_path=True,
    ),
}


# Aliases de workspace: clave corta → ruta. Env KIMMI_WORKSPACE_<ALIAS> tiene prioridad.
# Ejemplo: KIMMI_WORKSPACE_ASSISTANT=C:\Users\...\ai-assistant-test
WORKSPACE_ALIASES: Dict[str, str] = {
    "assistant": os.path.realpath("."),
}


def list_allowed_app_ids() -> List[str]:
    return sorted(APP_CATALOG.keys())


def get_app(app_id: str) -> Optional[AppEntry]:
    if not app_id:
        return None
    return APP_CATALOG.get(app_id.strip().lower())


def resolve_workspace_alias(name: str) -> Optional[str]:
    """Resuelve un alias de workspace a ruta absoluta, o None si no existe."""
    if not name:
        return None
    key = name.strip().lower()
    env_key = "KIMMI_WORKSPACE_" + re.sub(r"[^A-Za-z0-9]+", "_", key).upper()
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return os.path.abspath(os.path.expanduser(env_val))
    mapped = WORKSPACE_ALIASES.get(key)
    if mapped:
        return os.path.abspath(os.path.expanduser(mapped))
    return None


def browser_profile_args(profile: str) -> List[str]:
    """Flags allowlisted para perfiles de navegador (default|work)."""
    profile = (profile or "default").strip().lower()
    if profile in ("", "default"):
        return []
    if profile == "work":
        directory = (
            os.environ.get("KIMMI_BROWSER_PROFILE_WORK", "").strip()
            or "Profile 1"
        )
        return [f"--profile-directory={directory}"]
    return []


def _env_path(var_name: str) -> Optional[str]:
    value = os.environ.get(var_name, "").strip()
    return value or None



def resolve_launch_command(app_id: str) -> Optional[List[str]]:
    """Devuelve argv base para lanzar la app, o None si no se encuentra el ejecutable."""
    app = get_app(app_id)
    if app is None:
        return None

    system = platform.system()

    if app_id == "vscode":
        custom = _env_path("KIMMI_VSCODE_PATH")
        if custom:
            return [custom]
        which = shutil.which("code")
        if which:
            return [which]
        if system == "Windows":
            candidates = [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
                r"C:\Program Files\Microsoft VS Code\Code.exe",
            ]
            for path in candidates:
                if os.path.isfile(path):
                    return [path]
        return None

    if app_id == "cursor":
        custom = _env_path("KIMMI_CURSOR_PATH")
        if custom:
            return [custom]
        which = shutil.which("cursor")
        if which:
            return [which]
        if system == "Windows":
            candidates = [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\cursor\Cursor.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Cursor\Cursor.exe"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    return [path]
        return None

    if app_id == "browser":
        custom = _env_path("KIMMI_BROWSER_PATH")
        if custom:
            return [custom]
        if system == "Windows":
            for name in ("msedge", "chrome", "firefox", "brave"):
                which = shutil.which(name)
                if which:
                    return [which]
            edge = os.path.expandvars(
                r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
            )
            if not os.path.isfile(edge):
                edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            if os.path.isfile(edge):
                return [edge]
            chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            if os.path.isfile(chrome):
                return [chrome]
        else:
            for name in ("google-chrome", "chromium", "firefox", "brave"):
                which = shutil.which(name)
                if which:
                    return [which]
        return None

    if app_id == "terminal":
        custom = _env_path("KIMMI_TERMINAL_PATH")
        if custom:
            return [custom]
        if system == "Windows":
            wt = shutil.which("wt")
            if wt:
                return [wt]
            wt_path = os.path.expandvars(
                r"%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe"
            )
            if os.path.isfile(wt_path):
                return [wt_path]
            # Fallback razonable en Windows.
            return ["powershell"]
        which = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal")
        if which:
            return [which]
        return None

    return None

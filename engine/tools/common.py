"""Utilidades compartidas entre tools (truncado, skips, walk seguro)."""
from __future__ import annotations

import os
import re
from typing import Iterator, Optional, Pattern, Tuple

from tools.sandbox import is_allowed_text_file, resolve_safe_path

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "proj-env",
        ".next",
        ".jarvis_cache",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
)

CACHE_DIR_NAME = ".jarvis_cache"


def truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncado]"


def glob_to_re(glob_pat: str) -> Optional[Pattern]:
    if not glob_pat:
        return None
    escaped = re.escape(glob_pat).replace(r"\*\*", ".*").replace(r"\*", ".*")
    try:
        return re.compile(escaped + r"$", re.IGNORECASE)
    except re.error:
        return None


def filter_safe_dirnames(dirpath: str, dirnames: list, allowed_root: str) -> None:
    """Filtra dirnames in-place: skip ruidosos y fuera del sandbox."""
    safe = []
    for dirname in dirnames:
        if dirname in SKIP_DIR_NAMES:
            continue
        full = os.path.join(dirpath, dirname)
        resolved, err = resolve_safe_path(full, allowed_root)
        if err or resolved is None:
            continue
        safe.append(dirname)
    dirnames[:] = safe


def iter_safe_text_files(
    root_path: str,
    allowed_root: str,
    glob: str = "",
) -> Iterator[Tuple[str, str]]:
    """Yields (absolute_safe_path, relpath_from_allowed_root) for text files."""
    glob_re = glob_to_re(glob) if glob else None
    root = os.path.realpath(allowed_root)

    for dirpath, dirnames, filenames in os.walk(root_path):
        filter_safe_dirnames(dirpath, dirnames, allowed_root)
        for filename in sorted(filenames):
            if not is_allowed_text_file(filename):
                continue
            if glob_re:
                rel_full = os.path.relpath(
                    os.path.join(dirpath, filename), root
                ).replace("\\", "/")
                if not (
                    glob_re.search(filename) or glob_re.search(rel_full)
                ):
                    continue
            full = os.path.join(dirpath, filename)
            file_safe, file_error = resolve_safe_path(full, allowed_root)
            if file_error or file_safe is None or not os.path.isfile(file_safe):
                continue
            rel = os.path.relpath(file_safe, root)
            yield file_safe, rel


def ensure_cache_dir(allowed_root: str, *subdirs: str) -> str:
    """Crea y devuelve {allowed_root}/.jarvis_cache[/subdirs]."""
    path = os.path.join(os.path.realpath(allowed_root), CACHE_DIR_NAME, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path

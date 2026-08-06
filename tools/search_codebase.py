"""Búsqueda en el codebase: ripgrep (regex) o embeddings Ollama (semantic).

Requiere: ripgrep (`rg`) en PATH para mode=regex (con fallback textual).
Para semantic: Ollama con modelo de embeddings (ej. nomic-embed-text).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

from tools.sandbox import is_allowed_text_file, resolve_safe_path

MAX_MATCHES = 20
MAX_OUTPUT_CHARS = 2000
RG_TIMEOUT_SEC = 15
SNIPPET_LEN = 160
CHUNK_CHARS = 800
MAX_FILES_SEMANTIC = 80
MAX_CHUNKS = 200
TOP_K = 8
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def search_codebase(
    query: str,
    allowed_root: str,
    mode: str = "regex",
    path: str = ".",
    glob: str = "",
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> str:
    """Busca en el sandbox. mode: 'regex' | 'semantic'."""
    if not query or not str(query).strip():
        return "Debes indicar un texto o patrón de búsqueda."

    mode = (mode or "regex").strip().lower()
    if mode not in ("regex", "semantic"):
        return "Modo no válido. Usa 'regex' o 'semantic'."

    safe_path, error = resolve_safe_path(path or ".", allowed_root)
    if error:
        return error
    if not os.path.exists(safe_path):
        return f"La ruta {path} no existe."
    if not os.path.isdir(safe_path):
        return f"'{path}' no es una carpeta."

    if mode == "regex":
        return _search_regex(query.strip(), allowed_root, safe_path, glob.strip())
    return _search_semantic(
        query.strip(),
        allowed_root,
        safe_path,
        glob.strip(),
        embed_model or DEFAULT_EMBED_MODEL,
    )


def _search_regex(
    query: str,
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
) -> str:
    rg = shutil.which("rg")
    if not rg:
        return _fallback_text_search(query, allowed_root, safe_path, glob_pat)

    root = os.path.realpath(allowed_root)
    cmd = [
        rg,
        "--json",
        "--max-count",
        str(MAX_MATCHES),
        "-e",
        query,
        safe_path,
    ]
    if glob_pat:
        cmd.extend(["--glob", glob_pat])
    else:
        for ext in (".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".txt"):
            cmd.extend(["--glob", f"*{ext}"])

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RG_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return f"Búsqueda agotó el tiempo ({RG_TIMEOUT_SEC}s)."
    except Exception as e:
        return f"Error al ejecutar ripgrep: {e}"

    # 0 = matches, 1 = no matches, 2 = error
    if completed.returncode not in (0, 1):
        err = (completed.stderr or "").strip() or "error desconocido"
        return f"ripgrep falló: {err}"

    formatted = []
    for ln in (completed.stdout or "").splitlines():
        if len(formatted) >= MAX_MATCHES:
            break
        ln = ln.strip()
        if not ln:
            continue
        try:
            event = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        file_path = ((data.get("path") or {}).get("text")) or ""
        line_no = (data.get("line_number") or "?")
        content = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
        try:
            rel = os.path.relpath(file_path, root) if file_path else "?"
        except ValueError:
            rel = file_path
        snippet = content.strip()
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[:SNIPPET_LEN] + "..."
        formatted.append(f"{rel}:{line_no}: {snippet}")

    if not formatted:
        return f"No se encontraron coincidencias para '{query}'."

    header = f"Se encontraron {len(formatted)} coincidencia(s):\n"
    return _truncate(header + "\n".join(formatted))


def _fallback_text_search(
    query: str,
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
) -> str:
    """Fallback sin rg: búsqueda por regex en archivos permitidos."""
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Expresión regular inválida: {e}. Además, ripgrep no está en PATH."

    root = os.path.realpath(allowed_root)
    matches: List[str] = []
    glob_re = _glob_to_re(glob_pat) if glob_pat else None

    for dirpath, dirnames, filenames in os.walk(safe_path):
        safe_dirs = []
        for dirname in dirnames:
            full = os.path.join(dirpath, dirname)
            resolved, err = resolve_safe_path(full, allowed_root)
            if err or resolved is None:
                continue
            safe_dirs.append(dirname)
        dirnames[:] = safe_dirs

        for filename in filenames:
            if not is_allowed_text_file(filename):
                continue
            if glob_re and not glob_re.search(filename):
                continue
            full = os.path.join(dirpath, filename)
            file_safe, file_error = resolve_safe_path(full, allowed_root)
            if file_error or file_safe is None or not os.path.isfile(file_safe):
                continue
            rel = os.path.relpath(file_safe, root)
            try:
                with open(file_safe, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, start=1):
                        if pattern.search(line):
                            snippet = line.strip()
                            if len(snippet) > SNIPPET_LEN:
                                snippet = snippet[:SNIPPET_LEN] + "..."
                            matches.append(f"{rel}:{line_no}: {snippet}")
                            if len(matches) >= MAX_MATCHES:
                                return (
                                    "ripgrep no disponible; fallback textual. "
                                    f"Mostrando {MAX_MATCHES}+ coincidencias:\n"
                                    + "\n".join(matches)
                                )
            except Exception:
                continue

    if not matches:
        return (
            "ripgrep no está en PATH y el fallback no encontró coincidencias "
            f"para '{query}'."
        )
    return (
        f"ripgrep no disponible; fallback textual. "
        f"Se encontraron {len(matches)} coincidencia(s):\n" + "\n".join(matches)
    )


def _search_semantic(
    query: str,
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
    embed_model: str,
) -> str:
    try:
        import ollama
    except ImportError:
        return "Falta el paquete ollama para búsqueda semántica."

    chunks = _collect_chunks(allowed_root, safe_path, glob_pat)
    if not chunks:
        return "No hay archivos de texto indexables en esa ruta."

    try:
        q_emb = _embed(ollama, embed_model, query)
    except Exception as e:
        return (
            f"No se pudo obtener embedding de la consulta ({embed_model}): {e}. "
            "Asegura que el modelo esté descargado (ej. ollama pull nomic-embed-text)."
        )

    scored: List[Tuple[float, str, str]] = []
    for rel, snippet, text in chunks:
        try:
            emb = _embed(ollama, embed_model, text)
            score = _cosine(q_emb, emb)
            scored.append((score, rel, snippet))
        except Exception:
            continue

    if not scored:
        return "No se pudieron calcular embeddings de los archivos."

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    lines = [
        f"{rel} (score={score:.3f}): {snippet}"
        for score, rel, snippet in top
        if score > 0.05
    ]
    if not lines:
        return f"Sin resultados semánticos relevantes para '{query}'."
    return _truncate(
        f"Top {len(lines)} resultado(s) semánticos:\n" + "\n".join(lines)
    )


def _collect_chunks(
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
) -> List[Tuple[str, str, str]]:
    root = os.path.realpath(allowed_root)
    glob_re = _glob_to_re(glob_pat) if glob_pat else None
    chunks: List[Tuple[str, str, str]] = []
    files_seen = 0

    for dirpath, dirnames, filenames in os.walk(safe_path):
        safe_dirs = []
        for dirname in dirnames:
            # Saltar dirs ruidosos
            if dirname in (".git", "node_modules", "__pycache__", ".venv", "venv", ".next"):
                continue
            full = os.path.join(dirpath, dirname)
            resolved, err = resolve_safe_path(full, allowed_root)
            if err or resolved is None:
                continue
            safe_dirs.append(dirname)
        dirnames[:] = safe_dirs

        for filename in sorted(filenames):
            if not is_allowed_text_file(filename):
                continue
            if glob_re and not glob_re.search(filename):
                continue
            full = os.path.join(dirpath, filename)
            file_safe, file_error = resolve_safe_path(full, allowed_root)
            if file_error or file_safe is None or not os.path.isfile(file_safe):
                continue
            files_seen += 1
            if files_seen > MAX_FILES_SEMANTIC:
                return chunks
            rel = os.path.relpath(file_safe, root)
            try:
                with open(file_safe, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(20_000)
            except Exception:
                continue
            for i in range(0, len(content), CHUNK_CHARS):
                piece = content[i : i + CHUNK_CHARS].strip()
                if len(piece) < 40:
                    continue
                snippet = piece.replace("\n", " ")
                if len(snippet) > SNIPPET_LEN:
                    snippet = snippet[:SNIPPET_LEN] + "..."
                chunks.append((rel, snippet, piece))
                if len(chunks) >= MAX_CHUNKS:
                    return chunks
    return chunks


def _embed(ollama_mod, model: str, text: str) -> List[float]:
    resp = ollama_mod.embeddings(model=model, prompt=text)
    # API puede devolver dict con 'embedding'
    if isinstance(resp, dict) and "embedding" in resp:
        return list(resp["embedding"])
    embedding = getattr(resp, "embedding", None)
    if embedding is not None:
        return list(embedding)
    raise RuntimeError("Respuesta de embeddings sin vector.")


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _glob_to_re(glob_pat: str) -> Optional[re.Pattern]:
    # Soporte simple: *.py, **/*.ts
    escaped = re.escape(glob_pat).replace(r"\*\*", ".*").replace(r"\*", ".*")
    try:
        return re.compile(escaped + r"$", re.IGNORECASE)
    except re.error:
        return None


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "... [truncado]"
    return text

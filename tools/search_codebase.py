"""Búsqueda unificada en el codebase: text, regex, semantic, files_only.

Requiere: ripgrep (`rg`) en PATH para mode=regex (con fallback).
Para semantic: Ollama + modelo embeddings (ej. nomic-embed-text).
Caché semántica en {allowed_root}/.jarvis_cache/embeddings.json.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from tools.common import CACHE_DIR_NAME, iter_safe_text_files, truncate
from tools.sandbox import resolve_safe_path

MAX_MATCHES = 20
MAX_OUTPUT_CHARS = 2000
RG_TIMEOUT_SEC = 15
SNIPPET_LEN = 160
CHUNK_CHARS = 800
MAX_FILES_SEMANTIC = 80
MAX_CHUNKS = 200
TOP_K = 8
DEFAULT_EMBED_MODEL = "nomic-embed-text"
VALID_MODES = ("text", "regex", "semantic", "files_only")


def search_codebase(
    query: str,
    allowed_root: str,
    mode: str = "regex",
    path: str = ".",
    glob: str = "",
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> str:
    """Busca en el sandbox. mode: text | regex | semantic | files_only."""
    if not query or not str(query).strip():
        return "Debes indicar un texto o patrón de búsqueda."

    mode = (mode or "regex").strip().lower()
    if mode not in VALID_MODES:
        return f"Modo no válido. Usa: {', '.join(VALID_MODES)}."

    safe_path, error = resolve_safe_path(path or ".", allowed_root)
    if error:
        return error
    if not os.path.exists(safe_path):
        return f"La ruta {path} no existe."
    if not os.path.isdir(safe_path):
        return f"'{path}' no es una carpeta."

    q = query.strip()
    g = (glob or "").strip()

    if mode == "text":
        return _search_literal(q, allowed_root, safe_path, g, files_only=False)
    if mode == "files_only":
        # Prefer literal match for files_only unless query looks like regex metachars.
        if re.search(r"[\\.*+?^${}|()\[\]]", q):
            return _search_regex(q, allowed_root, safe_path, g, files_only=True)
        return _search_literal(q, allowed_root, safe_path, g, files_only=True)
    if mode == "regex":
        return _search_regex(q, allowed_root, safe_path, g, files_only=False)
    return _search_semantic(
        q,
        allowed_root,
        safe_path,
        g,
        embed_model or DEFAULT_EMBED_MODEL,
    )


def _search_literal(
    query: str,
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
    files_only: bool,
) -> str:
    needle = query.lower()
    matches: List[str] = []
    matched_files: List[str] = []

    for file_safe, rel in iter_safe_text_files(safe_path, allowed_root, glob_pat):
        try:
            with open(file_safe, "r", encoding="utf-8", errors="replace") as f:
                hit = False
                for line_no, line in enumerate(f, start=1):
                    if needle in line.lower():
                        hit = True
                        if files_only:
                            break
                        snippet = line.strip()
                        if len(snippet) > SNIPPET_LEN:
                            snippet = snippet[:SNIPPET_LEN] + "..."
                        matches.append(f"{rel}:{line_no}: {snippet}")
                        if len(matches) >= MAX_MATCHES:
                            return truncate(
                                f"Se encontraron {MAX_MATCHES}+ coincidencias:\n"
                                + "\n".join(matches),
                                MAX_OUTPUT_CHARS,
                            )
                if files_only and hit:
                    matched_files.append(rel)
                    if len(matched_files) >= MAX_MATCHES:
                        break
        except Exception as e:
            if not files_only:
                matches.append(f"{rel}: no se pudo leer ({e})")

    if files_only:
        if not matched_files:
            return f"No se encontraron archivos con '{query}'."
        return truncate(
            f"Se encontraron {len(matched_files)} archivo(s):\n"
            + "\n".join(matched_files),
            MAX_OUTPUT_CHARS,
        )

    if not matches:
        return f"No se encontraron coincidencias para '{query}'."
    return truncate(
        f"Se encontraron {len(matches)} coincidencia(s):\n" + "\n".join(matches),
        MAX_OUTPUT_CHARS,
    )


def _search_regex(
    query: str,
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
    files_only: bool,
) -> str:
    rg = shutil.which("rg")
    if not rg:
        return _fallback_regex(query, allowed_root, safe_path, glob_pat, files_only)

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
    # Excluir dirs ruidosos
    for skip in (
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".next",
        CACHE_DIR_NAME,
    ):
        cmd.extend(["--glob", f"!{skip}/**"])

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

    if completed.returncode not in (0, 1):
        err = (completed.stderr or "").strip() or "error desconocido"
        return f"ripgrep falló: {err}"

    formatted: List[str] = []
    files_seen: List[str] = []
    for ln in (completed.stdout or "").splitlines():
        if (files_only and len(files_seen) >= MAX_MATCHES) or (
            not files_only and len(formatted) >= MAX_MATCHES
        ):
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
        try:
            rel = os.path.relpath(file_path, root) if file_path else "?"
        except ValueError:
            rel = file_path
        if files_only:
            if rel not in files_seen:
                files_seen.append(rel)
            continue
        line_no = data.get("line_number") or "?"
        content = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
        snippet = content.strip()
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[:SNIPPET_LEN] + "..."
        formatted.append(f"{rel}:{line_no}: {snippet}")

    if files_only:
        if not files_seen:
            return f"No se encontraron archivos para '{query}'."
        return truncate(
            f"Se encontraron {len(files_seen)} archivo(s):\n" + "\n".join(files_seen),
            MAX_OUTPUT_CHARS,
        )

    if not formatted:
        return f"No se encontraron coincidencias para '{query}'."
    return truncate(
        f"Se encontraron {len(formatted)} coincidencia(s):\n" + "\n".join(formatted),
        MAX_OUTPUT_CHARS,
    )


def _fallback_regex(
    query: str,
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
    files_only: bool,
) -> str:
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Expresión regular inválida: {e}. Además, ripgrep no está en PATH."

    matches: List[str] = []
    matched_files: List[str] = []

    for file_safe, rel in iter_safe_text_files(safe_path, allowed_root, glob_pat):
        try:
            with open(file_safe, "r", encoding="utf-8", errors="replace") as f:
                hit = False
                for line_no, line in enumerate(f, start=1):
                    if pattern.search(line):
                        hit = True
                        if files_only:
                            break
                        snippet = line.strip()
                        if len(snippet) > SNIPPET_LEN:
                            snippet = snippet[:SNIPPET_LEN] + "..."
                        matches.append(f"{rel}:{line_no}: {snippet}")
                        if len(matches) >= MAX_MATCHES:
                            return truncate(
                                "ripgrep no disponible; fallback. "
                                f"Mostrando {MAX_MATCHES}+ coincidencias:\n"
                                + "\n".join(matches),
                                MAX_OUTPUT_CHARS,
                            )
                if files_only and hit:
                    matched_files.append(rel)
                    if len(matched_files) >= MAX_MATCHES:
                        break
        except Exception:
            continue

    if files_only:
        if not matched_files:
            return f"No se encontraron archivos para '{query}'."
        return truncate(
            "ripgrep no disponible; fallback. "
            f"Se encontraron {len(matched_files)} archivo(s):\n"
            + "\n".join(matched_files),
            MAX_OUTPUT_CHARS,
        )

    if not matches:
        return (
            "ripgrep no está en PATH y el fallback no encontró coincidencias "
            f"para '{query}'."
        )
    return truncate(
        "ripgrep no disponible; fallback. "
        f"Se encontraron {len(matches)} coincidencia(s):\n" + "\n".join(matches),
        MAX_OUTPUT_CHARS,
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

    cache = _load_embed_cache(allowed_root)
    cache_dirty = False

    try:
        q_emb = _embed(ollama, embed_model, query)
    except Exception as e:
        return (
            f"No se pudo obtener embedding de la consulta ({embed_model}): {e}. "
            "Asegura que el modelo esté descargado (ej. ollama pull nomic-embed-text)."
        )

    scored: List[Tuple[float, str, str]] = []
    for rel, snippet, text, mtime in chunks:
        key = f"{embed_model}::{rel}::{mtime}"
        emb = cache.get(key)
        if emb is None:
            try:
                emb = _embed(ollama, embed_model, text)
                cache[key] = emb
                cache_dirty = True
            except Exception:
                continue
        score = _cosine(q_emb, emb)
        scored.append((score, rel, snippet))

    if cache_dirty:
        _save_embed_cache(allowed_root, cache)

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
    return truncate(
        f"Top {len(lines)} resultado(s) semánticos:\n" + "\n".join(lines),
        MAX_OUTPUT_CHARS,
    )


def _collect_chunks(
    allowed_root: str,
    safe_path: str,
    glob_pat: str,
) -> List[Tuple[str, str, str, float]]:
    chunks: List[Tuple[str, str, str, float]] = []
    files_seen = 0

    for file_safe, rel in iter_safe_text_files(safe_path, allowed_root, glob_pat):
        files_seen += 1
        if files_seen > MAX_FILES_SEMANTIC:
            break
        try:
            mtime = os.path.getmtime(file_safe)
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
            # chunk index in key via mtime of whole file + offset in text hash-ish
            chunk_key_mtime = mtime + (i * 1e-6)
            chunks.append((rel, snippet, piece, chunk_key_mtime))
            if len(chunks) >= MAX_CHUNKS:
                return chunks
    return chunks


def _embed_cache_path(allowed_root: str) -> str:
    d = os.path.join(os.path.realpath(allowed_root), CACHE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "embeddings.json")


def _load_embed_cache(allowed_root: str) -> Dict[str, List[float]]:
    path = _embed_cache_path(allowed_root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_embed_cache(allowed_root: str, cache: Dict[str, List[float]]) -> None:
    # Limitar tamaño del cache (últimas 500 entradas)
    if len(cache) > 500:
        keys = list(cache.keys())[-500:]
        cache = {k: cache[k] for k in keys}
    path = _embed_cache_path(allowed_root)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _embed(ollama_mod, model: str, text: str) -> List[float]:
    resp = ollama_mod.embeddings(model=model, prompt=text)
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

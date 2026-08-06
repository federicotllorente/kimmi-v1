import os

from tools.sandbox import is_allowed_text_file, resolve_safe_path

MAX_MATCHES = 20
SNIPPET_LEN = 120


def search_files(query: str, allowed_root: str, path: str = ".") -> str:
    """Busca texto (case-insensitive) en archivos permitidos bajo el sandbox."""
    if not query or not query.strip():
        return "Debes indicar un texto de búsqueda."

    safe_path, error = resolve_safe_path(path, allowed_root)
    if error:
        return error

    if not os.path.exists(safe_path):
        return f"La ruta {path} no existe."

    if not os.path.isdir(safe_path):
        return f"'{path}' no es una carpeta."

    root = os.path.realpath(allowed_root)
    needle = query.strip().lower()
    matches = []

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

            full = os.path.join(dirpath, filename)
            file_safe, file_error = resolve_safe_path(full, allowed_root)
            if file_error or file_safe is None or not os.path.isfile(file_safe):
                continue

            rel = os.path.relpath(file_safe, root)
            try:
                with open(file_safe, 'r', encoding='utf-8', errors='replace') as f:
                    for line_no, line in enumerate(f, start=1):
                        if needle in line.lower():
                            snippet = line.strip()
                            if len(snippet) > SNIPPET_LEN:
                                snippet = snippet[:SNIPPET_LEN] + "..."
                            matches.append(f"{rel}:{line_no}: {snippet}")
                            if len(matches) >= MAX_MATCHES:
                                return (
                                    f"Se encontraron {MAX_MATCHES}+ coincidencias (mostrando las primeras):\n"
                                    + "\n".join(matches)
                                )
            except Exception as e:
                matches.append(f"{rel}: no se pudo leer ({e})")
                if len(matches) >= MAX_MATCHES:
                    break

    if not matches:
        return f"No se encontraron coincidencias para '{query}'."

    return f"Se encontraron {len(matches)} coincidencia(s):\n" + "\n".join(matches)

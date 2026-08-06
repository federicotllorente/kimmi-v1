import os

from tools.sandbox import resolve_safe_path


def list_directory(path: str, allowed_root: str, recursive: bool = False) -> str:
    """Lista archivos y carpetas dentro del sandbox."""
    safe_path, error = resolve_safe_path(path, allowed_root)
    if error:
        return error

    if not os.path.exists(safe_path):
        return f"La carpeta {path} no existe."

    if not os.path.isdir(safe_path):
        return f"'{path}' no es una carpeta."

    root = os.path.realpath(allowed_root)
    entries = []

    if recursive:
        for dirpath, dirnames, filenames in os.walk(safe_path):
            # Filtrar dirs que escapen del sandbox (symlinks).
            safe_dirs = []
            for dirname in dirnames:
                full = os.path.join(dirpath, dirname)
                resolved, err = resolve_safe_path(full, allowed_root)
                if err or resolved is None:
                    continue
                safe_dirs.append(dirname)
                rel = os.path.relpath(resolved, root)
                entries.append(f"[dir] {rel}")
            dirnames[:] = safe_dirs

            for filename in filenames:
                full = os.path.join(dirpath, filename)
                resolved, err = resolve_safe_path(full, allowed_root)
                if err or resolved is None:
                    continue
                rel = os.path.relpath(resolved, root)
                entries.append(f"[file] {rel}")
    else:
        for name in sorted(os.listdir(safe_path)):
            full = os.path.join(safe_path, name)
            resolved, err = resolve_safe_path(full, allowed_root)
            if err or resolved is None:
                continue
            rel = os.path.relpath(resolved, root)
            kind = "dir" if os.path.isdir(resolved) else "file"
            entries.append(f"[{kind}] {rel}")

    if not entries:
        return "La carpeta está vacía."

    return "\n".join(entries)

import os

from tools.sandbox import is_allowed_text_file, resolve_safe_path

MAX_CONTENT_BYTES = 50 * 1024  # 50 KB

def write_file(file_path: str, content: str, allowed_root: str) -> str:
    """Escribe o crea un archivo de texto solo dentro del sandbox."""
    if content is None:
        return "El contenido no puede ser nulo."

    content_bytes = content.encode('utf-8')
    if len(content_bytes) > MAX_CONTENT_BYTES:
        return f"Contenido demasiado grande (máximo {MAX_CONTENT_BYTES} bytes)."

    filename = os.path.basename(file_path)
    if not filename or filename in ('.', '..'):
        return "Debes indicar un nombre de archivo válido."

    if not is_allowed_text_file(filename):
        return "Extensión no permitida. Usa .txt, .md, .json o .py."

    safe_path, error = resolve_safe_path(file_path, allowed_root)
    if error:
        return error

    if os.path.isdir(safe_path):
        return f"'{file_path}' es una carpeta, no un archivo."

    root = os.path.realpath(allowed_root)
    parent = os.path.dirname(safe_path)

    if parent:
        parent_safe, parent_error = resolve_safe_path(parent, allowed_root)
        if parent_error:
            return parent_error
        try:
            os.makedirs(parent_safe, exist_ok=True)
        except Exception as e:
            return f"No se pudo crear el directorio padre: {e}"

    try:
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        return f"No se pudo escribir el archivo: {e}"

    rel = os.path.relpath(safe_path, root)
    return f"Archivo escrito correctamente: {rel} ({len(content_bytes)} bytes)."

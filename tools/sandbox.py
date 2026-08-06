import os
from typing import Optional, Tuple

ALLOWED_TEXT_EXTENSIONS = ('.txt', '.md', '.json', '.py', '.js', '.jsx', '.ts', '.tsx')

def resolve_safe_path(requested_path: str, allowed_root: str) -> Tuple[Optional[str], Optional[str]]:
    """Resuelve una ruta y comprueba que quede dentro de allowed_root.

    Returns:
        (ruta_segura, None) si es válida, o (None, mensaje_error) si no lo es.
    """
    root = os.path.realpath(allowed_root)

    if not os.path.isdir(root):
        return None, f"El directorio permitido '{allowed_root}' no existe o no es una carpeta."

    # Rutas relativas se interpretan respecto al sandbox; absolutas se validan igual.
    if os.path.isabs(requested_path):
        candidate = requested_path
    else:
        candidate = os.path.join(root, requested_path)

    resolved = os.path.realpath(candidate)

    try:
        common = os.path.commonpath([root, resolved])
    except ValueError:
        # En Windows, rutas en distintos drives no tienen path común.
        return None, "Acceso denegado: la ruta solicitada está fuera del directorio permitido."

    if common != root:
        return None, "Acceso denegado: la ruta solicitada está fuera del directorio permitido."

    return resolved, None


def is_allowed_text_file(filename: str) -> bool:
    """True si el nombre termina en una extensión de texto permitida."""
    return filename.lower().endswith(ALLOWED_TEXT_EXTENSIONS)

import os

from tools.sandbox import is_allowed_text_file, resolve_safe_path


def read_directory_files(directory_path: str, allowed_root: str) -> str:
    """Lee el texto de los archivos dentro de una carpeta, limitada a allowed_root."""
    safe_path, error = resolve_safe_path(directory_path, allowed_root)
    if error:
        return error

    if not os.path.exists(safe_path):
        return f"La carpeta {directory_path} no existe."

    if not os.path.isdir(safe_path):
        return f"'{directory_path}' no es una carpeta."

    output = []
    for filename in os.listdir(safe_path):
        filepath = os.path.join(safe_path, filename)
        # Revalidar cada archivo por si hubiera symlinks que salgan del sandbox.
        file_safe, file_error = resolve_safe_path(filepath, allowed_root)
        if file_error or file_safe is None:
            continue

        if os.path.isfile(file_safe) and is_allowed_text_file(filename):
            try:
                with open(file_safe, 'r', encoding='utf-8') as f:
                    content = f.read(500)  # Leemos los primeros 500 caracteres
                    output.append(f"Archivo '{filename}':\n{content}\n")
            except Exception as e:
                output.append(f"No se pudo leer {filename}: {str(e)}")

    return "\n".join(output) if output else "La carpeta está vacía o no tiene archivos de texto compatibles."

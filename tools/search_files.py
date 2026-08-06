"""Compat wrapper: búsqueda literal → search_codebase(mode='text')."""
from tools.search_codebase import search_codebase


def search_files(query: str, allowed_root: str, path: str = ".") -> str:
    """Busca texto (case-insensitive) en archivos permitidos bajo el sandbox."""
    return search_codebase(
        query,
        allowed_root=allowed_root,
        mode="text",
        path=path or ".",
    )

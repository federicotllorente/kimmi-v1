"""Tool: guardar un hecho en la memoria permanente (ChromaDB / RAG)."""
from __future__ import annotations

from tools.memory import memory_db


def remember_fact(fact: str, category: str = "general") -> str:
    """Guarda un dato, preferencia, IP, pinout o nota técnica importante."""
    return memory_db.add_memory(fact, category=category, source="explicit")

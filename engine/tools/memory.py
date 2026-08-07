"""Memoria vectorial persistente de Jarvis (ChromaDB + embeddings Ollama).

Guarda hechos en engine/memory/ y recupera los más relevantes por similitud.
Requiere: chromadb, ollama, y el modelo nomic-embed-text (`ollama pull nomic-embed-text`).
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, List, Optional

import chromadb
import ollama

_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(_ENGINE_DIR, "memory")
DEFAULT_EMBED_MODEL = "nomic-embed-text"
# Distancia máxima (cosine en Chroma ≈ 1 - similitud). Hits peores se descartan.
MAX_DISTANCE = 0.45
MAX_FACT_CHARS = 800
MAX_CONTEXT_CHARS = 1200


class JarvisMemory:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.embed_model = embed_model
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="jarvis_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    def _get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        if isinstance(response, dict) and "embedding" in response:
            return list(response["embedding"])
        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            return list(embedding)
        raise RuntimeError("Ollama no devolvió embedding.")

    def add_memory(self, text: str, category: str = "general") -> str:
        """Guarda un nuevo dato o preferencia en la memoria vectorial."""
        fact = (text or "").strip()
        if not fact:
            return "No hay nada que recordar: el texto está vacío."
        if len(fact) > MAX_FACT_CHARS:
            return f"El dato es demasiado largo (máximo {MAX_FACT_CHARS} caracteres)."

        category = (category or "general").strip().lower()[:64] or "general"
        doc_id = f"mem_{uuid.uuid4().hex[:12]}"

        try:
            vector = self._get_embedding(fact)
        except Exception as exc:  # noqa: BLE001
            return (
                f"No pude generar el embedding ({exc}). "
                f"¿Está Ollama en marcha y el modelo {self.embed_model} descargado?"
            )

        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[vector],
                documents=[fact],
                metadatas=[
                    {
                        "category": category,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return f"No se pudo guardar en ChromaDB: {exc}"

        return f"Recuerdo guardado ({doc_id})."

    def search_memory(self, query: str, n_results: int = 3) -> str:
        """Devuelve bullets con recuerdos relevantes, o cadena vacía."""
        query = (query or "").strip()
        if not query:
            return ""

        try:
            count = self.collection.count()
        except Exception:
            return ""
        if count == 0:
            return ""

        k = max(1, min(int(n_results), 5, count))
        try:
            query_vector = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [Memoria]: error en búsqueda RAG: {exc}")
            return ""

        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not documents:
            return ""

        lines: List[str] = []
        best_dist: Optional[float] = None
        scored: List[tuple] = []
        for i, doc in enumerate(documents):
            if not doc or not str(doc).strip():
                continue
            dist: Optional[float] = None
            if i < len(distances) and distances[i] is not None:
                try:
                    dist = float(distances[i])
                except (TypeError, ValueError):
                    dist = None
            # Con espacio cosine, Chroma reporta distancia ≈ 1 - cos_sim (0..2).
            if dist is not None and dist > MAX_DISTANCE:
                continue
            if dist is not None:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
            scored.append((dist if dist is not None else 999.0, str(doc).strip()))

        # Solo vecinos cercanos al mejor hit (evita relleno irrelevante).
        margin = 0.12
        for dist, doc in scored:
            if best_dist is not None and dist > best_dist + margin:
                continue
            lines.append(f"- {doc}")

        if not lines:
            return ""

        text = "\n".join(lines)
        if len(text) > MAX_CONTEXT_CHARS:
            text = text[: MAX_CONTEXT_CHARS - 3] + "..."
        return text

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0


# Instancia compartida por el proceso (tool + RAG del bucle).
memory_db = JarvisMemory()

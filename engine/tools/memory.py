"""Memoria vectorial persistente de Kimmi (ChromaDB + embeddings Ollama).

Guarda hechos en engine/memory/ y recupera los más relevantes por similitud.
Requiere: chromadb, ollama, y el modelo nomic-embed-text (`ollama pull nomic-embed-text`).
"""
from __future__ import annotations

import os
import time
import uuid
from collections import OrderedDict
from typing import List, Optional, Tuple

import chromadb
import ollama

from locale_config import current_locale

_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(_ENGINE_DIR, "memory")
DEFAULT_EMBED_MODEL = "nomic-embed-text"
# Distancia máxima (cosine en Chroma ≈ 1 - similitud). Hits peores se descartan.
MAX_DISTANCE = 0.45
# Soft-auto / dedup: below this cosine distance the candidate is treated as a duplicate.
DEDUP_DISTANCE = 0.25
MAX_FACT_CHARS = 800
MAX_CONTEXT_CHARS = 1200
# Mantener el modelo de embeddings cargado: sin esto usa el default de 5 min y
# se descarga/recarga compitiendo con el LLM de chat.
EMBED_KEEP_ALIVE = "24h"
# Cache de resultados recientes (query normalizada -> contexto).
QUERY_CACHE_SIZE = 32

# Un embedding por turno es caro. Saludos y muletillas nunca aportan contexto.
# El patrón por idioma vive en el locale como rag_skip_re.
_SKIP_QUERY_MIN_CHARS = 8

VALID_SOURCES = frozenset({"explicit", "soft_auto"})


class KimmiMemory:
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
            name="kimmi_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        # Contar en Chroma en cada turno es innecesario: solo cambia al escribir.
        self._count_cache: Optional[int] = None
        self._query_cache: "OrderedDict[str, str]" = OrderedDict()

    def warmup(self) -> None:
        """Cargar el modelo de embeddings antes del primer turno real."""
        self._get_embedding("warm")

    def _get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(
            model=self.embed_model,
            prompt=text,
            keep_alive=EMBED_KEEP_ALIVE,
        )
        if isinstance(response, dict) and "embedding" in response:
            return list(response["embedding"])
        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            return list(embedding)
        raise RuntimeError("Ollama no devolvió embedding.")

    def add_memory(
        self,
        text: str,
        category: str = "general",
        source: str = "explicit",
    ) -> str:
        """Guarda un nuevo dato o preferencia en la memoria vectorial."""
        fact = (text or "").strip()
        if not fact:
            return "No hay nada que recordar: el texto está vacío."
        if len(fact) > MAX_FACT_CHARS:
            return f"El dato es demasiado largo (máximo {MAX_FACT_CHARS} caracteres)."

        category = (category or "general").strip().lower()[:64] or "general"
        src = (source or "explicit").strip().lower()
        if src not in VALID_SOURCES:
            src = "explicit"
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
                        "source": src,
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            return f"No se pudo guardar en ChromaDB: {exc}"

        self._count_cache = None
        self._query_cache.clear()
        return f"Recuerdo guardado ({doc_id})."

    def find_similar(
        self,
        text: str,
        n: int = 1,
    ) -> Optional[Tuple[str, float]]:
        """Nearest stored fact and cosine distance, or None if empty/unavailable."""
        fact = (text or "").strip()
        if not fact:
            return None
        count = self._cached_count()
        if count == 0:
            return None
        k = max(1, min(int(n), 5, count))
        try:
            query_vector = self._get_embedding(fact)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [Memory]: find_similar failed: {exc}")
            return None

        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not documents or not str(documents[0] or "").strip():
            return None
        dist = 999.0
        if distances and distances[0] is not None:
            try:
                dist = float(distances[0])
            except (TypeError, ValueError):
                dist = 999.0
        return (str(documents[0]).strip(), dist)

    def _cached_count(self) -> int:
        if self._count_cache is None:
            try:
                self._count_cache = int(self.collection.count())
            except Exception:  # noqa: BLE001
                return 0
        return self._count_cache

    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return self._query_cache[key]
        return None

    def _cache_put(self, key: str, value: str) -> None:
        self._query_cache[key] = value
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)

    def search_memory(self, query: str, n_results: int = 3) -> str:
        """Devuelve bullets con recuerdos relevantes, o cadena vacía."""
        query = (query or "").strip()
        if not query:
            return ""

        # Gate barato antes de pagar el embedding.
        if len(query) < _SKIP_QUERY_MIN_CHARS:
            return ""
        if current_locale().rag_skip_re.match(query):
            return ""

        count = self._cached_count()
        if count == 0:
            return ""

        cache_key = query.lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        k = max(1, min(int(n_results), 5, count))
        try:
            query_vector = self._get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [Memory]: RAG search failed: {exc}")
            return ""

        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        if not documents:
            self._cache_put(cache_key, "")
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
            self._cache_put(cache_key, "")
            return ""

        text = "\n".join(lines)
        if len(text) > MAX_CONTEXT_CHARS:
            text = text[: MAX_CONTEXT_CHARS - 3] + "..."
        self._cache_put(cache_key, text)
        return text

    def count(self) -> int:
        return self._cached_count()


# Instancia compartida por el proceso (tool + RAG del bucle).
memory_db = KimmiMemory()

"""In-process vector index over persisted embeddings.

:class:`VectorIndex` keeps a cached matrix of the store's embedding vectors and
does top-k cosine search — accelerated with NumPy when it is installed, falling
back to pure Python otherwise. This is the "candidate retrieval" stage of hybrid
recall: it narrows millions of vectors to a handful, which the memory OS then
re-ranks with lexical signals.

:class:`FaissVectorIndex` is an optional drop-in for very large corpora; it lazy
-imports ``faiss`` and raises a clear error if the package is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from plasticity_agent.memory.embeddings import EmbeddingBackend
from plasticity_agent.memory.retrieval import cosine_similarity

if TYPE_CHECKING:
    from plasticity_agent.memory.store import MemoryStore

try:  # optional acceleration
    import numpy as _np
except Exception:  # noqa: BLE001 - numpy is optional
    _np = None  # type: ignore[assignment]


class VectorIndex:
    """Cosine top-k search over the store's persisted vectors."""

    def __init__(self, store: MemoryStore, embedder: EmbeddingBackend) -> None:
        self.store = store
        self.embedder = embedder
        self._ids: list[str] = []
        self._matrix: Any = None
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def add(self, memory_id: str, vector: Sequence[float]) -> None:
        self.store.upsert_vector(memory_id, vector)
        self._dirty = True

    def embed_text(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0]

    def _ensure(self) -> None:
        if not self._dirty and self._matrix is not None:
            return
        vectors = self.store.all_vectors()
        self._ids = list(vectors.keys())
        rows = [vectors[i] for i in self._ids]
        if _np is not None and rows:
            self._matrix = _np.asarray(rows, dtype="float32")
        else:
            self._matrix = rows
        self._dirty = False

    def search(self, query_vector: Sequence[float], k: int = 10) -> list[tuple[str, float]]:
        self._ensure()
        if not self._ids:
            return []
        if _np is not None and isinstance(self._matrix, _np.ndarray):
            query = _np.asarray(query_vector, dtype="float32")
            query_norm = float(_np.linalg.norm(query)) or 1.0
            matrix_norms = _np.linalg.norm(self._matrix, axis=1)
            matrix_norms[matrix_norms == 0] = 1.0
            sims = (self._matrix @ query) / matrix_norms / query_norm
            order = sims.argsort()[::-1][:k]
            return [(self._ids[i], float(sims[i])) for i in order]
        scored = [
            (self._ids[i], cosine_similarity(query_vector, row))
            for i, row in enumerate(self._matrix)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


class FaissVectorIndex:
    """Optional FAISS-backed index for large corpora (lazy import)."""

    def __init__(self, store: MemoryStore, embedder: EmbeddingBackend) -> None:
        self.store = store
        self.embedder = embedder
        self._faiss: Any | None = None
        self._index: Any | None = None
        self._ids: list[str] = []

    def _require(self) -> Any:
        if self._faiss is None:
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "FaissVectorIndex requires the optional 'faiss-cpu' package. "
                    "Install it with `pip install faiss-cpu`."
                ) from exc
            self._faiss = faiss
        return self._faiss

    def embed_text(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0]

    def rebuild(self) -> None:  # pragma: no cover - exercised only with faiss installed
        faiss = self._require()
        import numpy as np

        vectors = self.store.all_vectors()
        self._ids = list(vectors.keys())
        if not self._ids:
            self._index = None
            return
        matrix = np.asarray([vectors[i] for i in self._ids], dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index

    def search(  # pragma: no cover - requires faiss
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[str, float]]:
        faiss = self._require()
        import numpy as np

        if self._index is None:
            self.rebuild()
        if self._index is None:
            return []
        query = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, k)
        return [
            (self._ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0], strict=False)
            if idx != -1
        ]

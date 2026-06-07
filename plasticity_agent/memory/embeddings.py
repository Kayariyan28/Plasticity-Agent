"""Embedding backends for dense/semantic retrieval.

Three real, pluggable backends:

- :class:`HashingEmbeddingBackend` — deterministic feature-hashing embeddings
  with **no dependencies**. Real dense vectors (cosine works), good for offline
  use and tests; it captures n-gram structure rather than deep semantics.
- :class:`SentenceTransformerBackend` — true semantic embeddings via the
  optional ``sentence-transformers`` package (lazy import, clear error if absent).
- :class:`CallableEmbeddingBackend` — wrap any ``fn(list[str]) -> list[vector]``
  (e.g. an OpenAI/Cohere embeddings endpoint).

Use :func:`get_embedder` to resolve a spec string/object into a backend.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from plasticity_agent.memory.retrieval import tokenize


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that maps texts to fixed-length vectors."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        return vector
    return [component / norm for component in vector]


class HashingEmbeddingBackend:
    """Deterministic, dependency-free feature-hashing embeddings."""

    name = "hashing"

    def __init__(self, dim: int = 256, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = max(1, ngram)

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        features = list(tokens)
        for n in range(2, self.ngram + 1):
            features.extend(
                "_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
        return features

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class CallableEmbeddingBackend:
    """Wrap a ``fn(list[str]) -> list[vector]`` callable as a backend."""

    name = "callable"

    def __init__(self, embed_fn: Callable[[Sequence[str]], Any], *, dim: int | None = None) -> None:
        self._fn = embed_fn
        self.dim = dim or 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [[float(x) for x in vec] for vec in self._fn(list(texts))]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return vectors


class SentenceTransformerBackend:
    """Semantic embeddings via the optional ``sentence-transformers`` package."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model: Any | None = None
        self.dim = 0

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "SentenceTransformerBackend requires the optional "
                    "'sentence-transformers' package. Install it with "
                    "`pip install sentence-transformers`."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in vec] for vec in vectors]


def get_embedder(
    spec: EmbeddingBackend | Callable[..., Any] | str | None,
) -> EmbeddingBackend | None:
    """Resolve ``spec`` into an :class:`EmbeddingBackend` (or ``None``).

    Accepts: ``None`` (lexical only), a backend instance, a callable, or a string
    like ``"hashing"``, ``"hashing:512"``, ``"st:all-MiniLM-L6-v2"``.
    """

    if spec is None:
        return None
    if hasattr(spec, "embed") and callable(spec.embed):
        return spec  # type: ignore[return-value]
    if callable(spec):
        return CallableEmbeddingBackend(spec)
    if isinstance(spec, str):
        text = spec.strip()
        if text in {"hashing", "hash"}:
            return HashingEmbeddingBackend()
        if text.startswith("hashing:"):
            return HashingEmbeddingBackend(dim=int(text.split(":", 1)[1]))
        if text.startswith(("st:", "sentence-transformers:")):
            return SentenceTransformerBackend(text.split(":", 1)[1])
        raise ValueError(f"unknown embedder spec: {spec!r}")
    raise TypeError(f"cannot build an embedder from {type(spec)!r}")

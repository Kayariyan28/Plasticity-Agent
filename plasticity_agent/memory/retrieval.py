"""Lexical retrieval and the shared text utilities.

v0.1.0 deliberately ships *no* vector database. Recall uses deterministic
lexical similarity (a blend of Jaccard and overlap coefficient) plus light
salience priors. The tokenizer and ``lexical_similarity`` here are the single
source of truth reused by contradiction detection and consolidation.

The :class:`RetrievalBackend` protocol is the documented hook for a future
embedding/vector adapter — swap it in without touching the rest of the stack.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

from plasticity_agent.memory.schemas import Memory, MemorySearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "as", "at", "by", "from", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "they", "we", "me", "my", "our", "your", "their",
    "do", "does", "did", "so", "such", "than", "too", "very", "can", "will",
    "just", "about", "into", "over", "after", "before", "up", "down", "out",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens."""

    return [
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def lexical_similarity(a: str, b: str) -> float:
    """Similarity in ``[0, 1]`` blending Jaccard and overlap coefficient.

    The overlap-coefficient term keeps short queries from being unfairly
    penalised against long memories.
    """

    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    intersection = len(sa & sb)
    if intersection == 0:
        return 0.0
    jaccard = intersection / len(sa | sb)
    overlap = intersection / min(len(sa), len(sb))
    return max(0.0, min(1.0, 0.5 * jaccard + 0.5 * overlap))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is empty/degenerate)."""

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


@runtime_checkable
class RetrievalBackend(Protocol):
    """Pluggable scorer hook — an alternative to lexical similarity.

    Implement ``score(query, content) -> float`` to back retrieval with your own
    (e.g. cross-encoder) relevance model. For dense vector retrieval, prefer the
    ``query_vector``/``vector_of`` path of :func:`search_memories`.
    """

    def score(self, query: str, content: str) -> float: ...


def search_memories(
    query: str,
    memories: Iterable[Memory],
    *,
    limit: int = 5,
    min_score: float = 0.0,
    backend: RetrievalBackend | None = None,
    query_vector: Sequence[float] | None = None,
    vector_of: Callable[[Memory], Sequence[float] | None] | None = None,
    alpha: float = 0.5,
) -> list[MemorySearchResult]:
    """Rank ``memories`` against ``query`` and return the top hits.

    Lexical by default. When ``query_vector`` and ``vector_of`` are supplied the
    relevance is a hybrid blend ``(1 - alpha)*lexical + alpha*cosine`` — this is
    how dense/semantic retrieval is mixed in without losing keyword precision.
    """

    query_tokens = token_set(query)
    use_vectors = query_vector is not None and vector_of is not None
    results: list[MemorySearchResult] = []

    for memory in memories:
        lexical = backend.score(query, memory.content) if backend else lexical_similarity(
            query, memory.content
        )
        vector_score = 0.0
        if query_vector is not None and vector_of is not None:
            vector = vector_of(memory)
            if vector is not None:
                vector_score = max(0.0, cosine_similarity(query_vector, vector))

        relevance = (1.0 - alpha) * lexical + alpha * vector_score if use_vectors else lexical
        tag_hit = next((tag for tag in memory.tags if tag.lower() in query_tokens), None)
        if relevance <= 0.0 and tag_hit is None:
            continue

        score = 0.78 * relevance + (0.12 if tag_hit else 0.0) + 0.10 * memory.salience
        score = max(0.0, min(1.0, score))
        if score < min_score:
            continue

        overlap_terms = sorted(query_tokens & token_set(memory.content))
        reason_parts: list[str] = []
        if overlap_terms:
            reason_parts.append(f"lexical overlap on: {', '.join(overlap_terms[:4])}")
        if use_vectors and vector_score > 0.0:
            reason_parts.append(f"semantic cos={vector_score:.2f}")
        if tag_hit:
            reason_parts.append(f"tag {tag_hit}")
        reason = " · ".join(reason_parts) or "salience prior"

        results.append(MemorySearchResult(memory=memory, score=score, match_reason=reason))

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]

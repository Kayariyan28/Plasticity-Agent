"""Tests for embeddings, hybrid recall, and the vector index."""

from __future__ import annotations

import math

from plasticity_agent.memory.embeddings import (
    CallableEmbeddingBackend,
    HashingEmbeddingBackend,
    get_embedder,
)
from plasticity_agent.memory.memory_os import MemoryOS
from plasticity_agent.memory.retrieval import cosine_similarity


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    backend = HashingEmbeddingBackend(dim=64)
    vectors = backend.embed(["hello there world", "hello there world"])
    assert len(vectors[0]) == 64
    assert vectors[0] == vectors[1]  # deterministic
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert abs(norm - 1.0) < 1e-6


def test_cosine_similarity_bounds() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


def test_get_embedder_specs() -> None:
    assert get_embedder(None) is None
    assert isinstance(get_embedder("hashing"), HashingEmbeddingBackend)
    assert get_embedder("hashing:128").dim == 128


def test_callable_embedding_backend_infers_dim() -> None:
    backend = CallableEmbeddingBackend(lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    assert backend.embed(["x"]) == [[1.0, 0.0, 0.0]]
    assert backend.dim == 3


def test_memory_os_hybrid_recall_persists_vectors(tmp_path) -> None:
    memory = MemoryOS(memory_dir=str(tmp_path / "m"), embedder="hashing")
    try:
        memory.record("Cache warmup reduced p95 latency", "semantic")
        memory.record("Unrelated note about gardening tomatoes", "episodic")
        results = memory.recall("latency cache warmup")
        assert results
        top = results[0].memory.content.lower()
        assert "latency" in top or "cache" in top
        assert memory.store.vector_count() == 2
        assert "semantic" in results[0].match_reason or "lexical" in results[0].match_reason
    finally:
        memory.close()


def test_vector_index_ranks_semantically_closer_first(tmp_path) -> None:
    memory = MemoryOS(memory_dir=str(tmp_path / "m"), embedder="hashing")
    try:
        api = memory.record("payment api timeout error during checkout")
        memory.record("gardening tips for growing tomatoes")
        index = memory.vector_index
        assert index is not None
        query_vector = index.embed_text("api timeout checkout")
        hits = index.search(query_vector, k=2)
        assert hits
        assert hits[0][0] == api.id
    finally:
        memory.close()


def test_reindex_embeddings(tmp_path) -> None:
    memory = MemoryOS(memory_dir=str(tmp_path / "m"), embedder="hashing")
    try:
        memory.record("one")
        memory.record("two")
        assert memory.reindex_embeddings() == 2
        assert memory.store.vector_count() == 2
    finally:
        memory.close()

# Retrieval (Lexical + Vector)

Recall is **lexical by default** and becomes **hybrid lexical+vector** the moment you configure
an embedding backend — no other code changes.

## Enable vectors

```python
from plasticity_agent import PlasticAgent

# Zero-dependency hashing vectors (offline, deterministic):
agent = PlasticAgent(name="copilot", embeddings="hashing")

# True semantic vectors (pip install sentence-transformers):
agent = PlasticAgent(name="copilot", embeddings="st:all-MiniLM-L6-v2")

# Any embeddings endpoint:
agent = PlasticAgent(name="copilot", embeddings=lambda texts: my_embed(texts))
```

`MemoryOS(embedder=...)` accepts the same values.

## How hybrid recall works

1. Each memory's embedding is computed on `record()` and **persisted** in a `vectors` table.
2. On `recall(query)` the query is embedded and relevance is blended:

```text
relevance = (1 - alpha) * lexical_similarity + alpha * cosine_similarity      # alpha defaults 0.5
score     = 0.78 * relevance + 0.12 * tag_hit + 0.10 * salience_prior
```

3. For large corpora (over ~512 memories) a `VectorIndex` first narrows to the top vector
   candidates, which are then re-ranked — so recall scales beyond "load everything".

`match_reason` tells you what fired, e.g. `lexical overlap on: cache, latency · semantic cos=0.41`.

## Backends

| Backend | Dependency | Notes |
| --- | --- | --- |
| `HashingEmbeddingBackend` | none | Deterministic n-gram feature hashing; great offline/tests |
| `SentenceTransformerBackend` | `sentence-transformers` | True semantic embeddings (lazy import) |
| `CallableEmbeddingBackend` | none | Wrap any `fn(list[str]) -> list[vector]` |

## Vector index

```python
index = agent.memory.vector_index          # VectorIndex (NumPy-accelerated if available)
hits = index.search(index.embed_text("api timeout"), k=5)   # [(memory_id, cosine), ...]
agent.memory.reindex_embeddings()           # (re)embed everything, e.g. after a backend swap
```

`FaissVectorIndex` is a drop-in for very large corpora (`pip install faiss-cpu`).

## Custom relevance backends

Implement `RetrievalBackend.score(query, content) -> float` (e.g. a cross-encoder) and pass it to
`search_memories(..., backend=...)` to replace lexical scoring entirely.

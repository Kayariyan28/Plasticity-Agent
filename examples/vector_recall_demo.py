"""Hybrid lexical+vector recall with the zero-dependency hashing embedder.

Swap ``embedder="hashing"`` for ``embedder="st:all-MiniLM-L6-v2"`` (after
``pip install sentence-transformers``) for true semantic matching.

Run:  uv run python examples/vector_recall_demo.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent.memory.memory_os import MemoryOS


def main() -> None:
    memory = MemoryOS(memory_dir=tempfile.mkdtemp(prefix="plasticity_vec_"), embedder="hashing")

    for text, kind in [
        ("Cache warmup reduced p95 latency by 40%", "semantic"),
        ("Retry with exponential backoff fixed the flaky upload", "semantic"),
        ("The team standup is at 9am on weekdays", "episodic"),
    ]:
        memory.record(text, kind)

    print(f"vectors persisted: {memory.store.vector_count()}\n")
    for query in ["latency cache performance", "backoff retry upload"]:
        print(f"query: {query}")
        for hit in memory.recall(query, limit=2):
            print(f"  {hit.score:.3f}  {hit.memory.content[:48]}  ({hit.match_reason})")
        print()

    memory.close()


if __name__ == "__main__":
    main()

"""The Memory OS: the public façade over the whole memory subsystem.

It wires together persistence (SQLite, concurrency-safe), retrieval (lexical or
hybrid lexical+vector), salience, contradiction detection (heuristic or
LLM-backed), decay/forgetting, sleep-like consolidation, and the skill library —
emitting a trace event for every mutation.

Deterministic memory-quality scoring lives here so it can be injected into the
lower-level decay/consolidation passes without creating import cycles.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.core.events import MEMORY_RECALLED, MEMORY_RECORDED, SLEEP_COMPLETED
from plasticity_agent.core.trace import Tracer, load_trace_records
from plasticity_agent.llm.client import LLMClient, coerce_llm
from plasticity_agent.memory.consolidation import (
    SimilarityFn,
    SleepReport,
    consolidate_memories,
    run_sleep_cycle,
)
from plasticity_agent.memory.contradiction import ContradictionChecker
from plasticity_agent.memory.embeddings import EmbeddingBackend, get_embedder
from plasticity_agent.memory.forgetting import DecayStats, decay_memories
from plasticity_agent.memory.retrieval import (
    cosine_similarity,
    lexical_similarity,
    search_memories,
)
from plasticity_agent.memory.salience import IMPORTANT_TAGS, calculate_salience
from plasticity_agent.memory.schemas import (
    Memory,
    MemoryQualityReport,
    MemorySearchResult,
    MemoryType,
    Recommendation,
)
from plasticity_agent.memory.store import MemoryStore
from plasticity_agent.memory.vector_index import VectorIndex


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _recency_score(memory: Memory, *, now: datetime | None = None) -> float:
    """1.0 for brand-new memories, halving roughly every two weeks."""

    return 1.0 / (1.0 + memory.age_days(now=now) / 14.0)


def compute_utility_score(memory: Memory, *, now: datetime | None = None) -> float:
    """The documented memory-quality formula, clamped to ``[0, 1]``."""

    normalized_usage = min(memory.usage_count / 10.0, 1.0)
    positive_reward = max(0.0, min(memory.reward, 1.0))
    score = (
        0.25 * memory.salience
        + 0.20 * memory.confidence
        + 0.15 * normalized_usage
        + 0.15 * positive_reward
        + 0.10 * _recency_score(memory, now=now)
        - 0.20 * memory.contradiction_score
        - 0.10 * memory.decay_rate
    )
    return _clamp01(score)


def _recommend(memory: Memory, utility: float) -> tuple[Recommendation, list[str]]:
    reasons: list[str] = []
    protected = memory.memory_type == "constitutional" or bool(
        {t.lower() for t in memory.tags} & IMPORTANT_TAGS
    )

    if memory.contradiction_score >= 0.6:
        reasons.append(f"high contradiction score ({memory.contradiction_score:.2f})")
        return "review", reasons
    if utility < 0.15 and not protected:
        reasons.append(f"very low utility ({utility:.2f})")
        return "prune", reasons
    if utility < 0.35 and not protected:
        reasons.append(f"low utility ({utility:.2f}); decay candidate")
        return "decay", reasons
    if memory.usage_count >= 4 or memory.salience >= 0.8:
        reasons.append(
            f"high recurrence/salience (usage={memory.usage_count}, salience={memory.salience:.2f})"
        )
        return "consolidate", reasons
    reasons.append(f"healthy memory (utility={utility:.2f})")
    return "keep", reasons


def score_memory_quality(memory: Memory, *, now: datetime | None = None) -> MemoryQualityReport:
    """Full deterministic quality report for a single memory."""

    utility = compute_utility_score(memory, now=now)
    recommendation, reasons = _recommend(memory, utility)
    return MemoryQualityReport(
        memory_id=memory.id,
        utility_score=utility,
        salience=memory.salience,
        confidence=memory.confidence,
        contradiction_score=memory.contradiction_score,
        decay_rate=memory.decay_rate,
        usage_count=memory.usage_count,
        recommendation=recommendation,
        reasons=reasons,
    )


class MemoryOS:
    """Local-first neuroplastic memory: record, recall, evaluate, sleep.

    Pass ``embedder`` (e.g. ``"hashing"``, ``"st:all-MiniLM-L6-v2"``, a backend,
    or an embed callable) to enable hybrid lexical+vector recall, and ``llm`` to
    enable LLM-backed semantic contradiction detection.
    """

    def __init__(
        self,
        memory_dir: str | Path = "./memory",
        *,
        config: PlasticityConfig | None = None,
        tracer: Tracer | None = None,
        llm: LLMClient | Callable[..., Any] | None = None,
        embedder: EmbeddingBackend | Callable[..., Any] | str | None = None,
        vector_alpha: float = 0.5,
    ) -> None:
        from plasticity_agent.learning.skill_library import SkillLibrary

        self.config = config or PlasticityConfig.from_memory_dir(memory_dir)
        self.config.ensure_dirs()
        self.store = MemoryStore(self.config.db_path)
        self.skills = SkillLibrary(self.config.db_path)
        self.tracer = tracer or Tracer(self.config.traces_dir)

        self.llm = coerce_llm(llm)
        self.embedder = get_embedder(embedder)
        self.contradiction = ContradictionChecker(self.llm)
        self.vector_index = VectorIndex(self.store, self.embedder) if self.embedder else None
        self.vector_alpha = vector_alpha
        self._candidate_threshold = 512

    # -- embeddings -----------------------------------------------------------

    def _embed_and_store(self, memories: list[Memory]) -> None:
        if self.embedder is None or not memories:
            return
        vectors = self.embedder.embed([m.content for m in memories])
        for memory, vector in zip(memories, vectors, strict=False):
            self.store.upsert_vector(memory.id, vector)
        if self.vector_index is not None:
            self.vector_index.mark_dirty()

    def reindex_embeddings(self) -> int:
        """(Re)compute and persist embeddings for every memory. Returns count."""

        if self.embedder is None:
            return 0
        memories = self.store.all()
        self._embed_and_store(memories)
        return len(memories)

    def _embedding_similarity(self) -> SimilarityFn | None:
        if self.embedder is None:
            return None
        vectors = self.store.all_vectors()

        def similarity(a: Memory, b: Memory) -> float:
            va, vb = vectors.get(a.id), vectors.get(b.id)
            if va is None or vb is None:
                return lexical_similarity(a.content, b.content)
            return cosine_similarity(va, vb)

        return similarity

    # -- recording ------------------------------------------------------------

    def record(
        self,
        content: str,
        memory_type: MemoryType = "episodic",
        *,
        tags: list[str] | None = None,
        reward: float = 0.0,
        confidence: float = 0.7,
        salience: float | None = None,
        source_trace: str | None = None,
        metadata: dict[str, Any] | None = None,
        check_contradictions: bool = True,
    ) -> Memory:
        """Persist a new memory, auto-scoring salience and contradiction."""

        tags = list(tags or [])
        if salience is None:
            salience = calculate_salience(
                content, reward=reward, tags=tags, confidence=confidence
            )
        memory = Memory(
            content=content,
            memory_type=memory_type,
            tags=tags,
            salience=salience,
            confidence=confidence,
            reward=reward,
            source_trace=source_trace,
            metadata=metadata or {},
        )
        if check_contradictions:
            memory.contradiction_score = self.contradiction.detect(memory, self.store.all())

        self.store.upsert(memory)
        self._embed_and_store([memory])
        self.tracer.emit(
            MEMORY_RECORDED,
            {
                "memory_id": memory.id,
                "memory_type": memory.memory_type,
                "salience": memory.salience,
                "contradiction_score": memory.contradiction_score,
            },
        )
        return memory

    # -- retrieval ------------------------------------------------------------

    def _rank(
        self, query: str, *, limit: int, min_score: float
    ) -> list[MemorySearchResult]:
        if self.embedder is None:
            return search_memories(
                query, self.store.all(), limit=limit, min_score=min_score
            )

        query_vector = self.embedder.embed([query])[0]
        vectors: dict[str, list[float]]
        if self.vector_index is not None and self.store.count() > self._candidate_threshold:
            hits = self.vector_index.search(query_vector, k=max(limit * 8, 64))
            memories = [m for m in (self.store.get(cid) for cid, _ in hits) if m is not None]
            candidate = {m.id: self.store.get_vector(m.id) for m in memories}
            vectors = {mid: vec for mid, vec in candidate.items() if vec is not None}
        else:
            memories = self.store.all()
            vectors = self.store.all_vectors()

        return search_memories(
            query,
            memories,
            limit=limit,
            min_score=min_score,
            query_vector=query_vector,
            vector_of=lambda m: vectors.get(m.id),
            alpha=self.vector_alpha,
        )

    def search(
        self, query: str, *, limit: int = 5, min_score: float = 0.0
    ) -> list[MemorySearchResult]:
        """Pure search — no usage accounting, no trace."""

        return self._rank(query, limit=limit, min_score=min_score)

    def recall(
        self, query: str, *, limit: int = 5, min_score: float = 0.0
    ) -> list[MemorySearchResult]:
        """Active recall — increments usage on hits and emits a trace event."""

        results = self._rank(query, limit=limit, min_score=min_score)
        for result in results:
            result.memory.usage_count += 1
            result.memory.touch()
            self.store.upsert(result.memory)
        self.tracer.emit(
            MEMORY_RECALLED,
            {
                "query": query,
                "hits": len(results),
                "mode": "hybrid" if self.embedder else "lexical",
            },
        )
        return results

    def list_memories(
        self, *, memory_type: MemoryType | None = None, limit: int | None = None
    ) -> list[Memory]:
        memories = self.store.by_type(memory_type) if memory_type else self.store.all()
        return memories[:limit] if limit else memories

    def get(self, memory_id: str) -> Memory | None:
        return self.store.get(memory_id)

    # -- evaluation -----------------------------------------------------------

    def evaluate_memory(self, memory: Memory | str) -> MemoryQualityReport:
        resolved = self.store.get(memory) if isinstance(memory, str) else memory
        if resolved is None:
            raise KeyError(f"unknown memory id: {memory!r}")
        return score_memory_quality(resolved)

    def evaluate_all(self) -> list[MemoryQualityReport]:
        return [score_memory_quality(memory) for memory in self.store.all()]

    # -- plasticity operations ------------------------------------------------

    def decay(self, days_passed: int | None = None) -> DecayStats:
        memories = self.store.all()
        _, stats = decay_memories(
            memories,
            days_passed=days_passed,
            config=self.config,
            utility_fn=compute_utility_score,
        )
        self.store.upsert_many(memories)
        return stats

    def consolidate(self) -> int:
        """Compress reflective clusters into semantic memories. Returns count."""

        memories = self.store.all()
        new_memories, consolidated, candidates = consolidate_memories(
            memories,
            self.load_traces(),
            config=self.config,
            similarity_fn=self._embedding_similarity(),
        )
        self.store.upsert_many(memories)
        self.store.upsert_many(new_memories)
        self._embed_and_store(new_memories)
        for candidate in candidates:
            self.skills.promote_from_trace(
                candidate.name,
                candidate.successful_trace,
                description=candidate.description,
                trigger_patterns=candidate.trigger_patterns,
                confidence=candidate.confidence,
                reward=candidate.reward,
            )
        return consolidated

    def sleep(self, traces_path: str | Path | None = None) -> SleepReport:
        """Run a full sleep/consolidation cycle and return a report."""

        memories = self.store.all()
        trace_records = (
            load_trace_records(traces_path) if traces_path else self.load_traces()
        )
        outcome = run_sleep_cycle(
            memories,
            trace_records,
            config=self.config,
            utility_fn=compute_utility_score,
            similarity_fn=self._embedding_similarity(),
            pair_fn=self.contradiction.pair,
        )

        removed = set(outcome.removed_ids)
        survivors = [m for m in memories if m.id not in removed]
        self.store.upsert_many(survivors)
        self.store.upsert_many(outcome.new_memories)
        self._embed_and_store(outcome.new_memories)
        for memory_id in outcome.removed_ids:
            self.store.delete(memory_id)
        if self.vector_index is not None:
            self.vector_index.mark_dirty()

        for candidate in outcome.skill_candidates:
            self.skills.promote_from_trace(
                candidate.name,
                candidate.successful_trace,
                description=candidate.description,
                trigger_patterns=candidate.trigger_patterns,
                confidence=candidate.confidence,
                reward=candidate.reward,
            )
        for suggestion in outcome.policy_suggestions:
            self.record(
                suggestion,
                "reflective",
                tags=["policy", "advisory"],
                confidence=0.6,
                check_contradictions=False,
            )
        self.tracer.emit(SLEEP_COMPLETED, outcome.report.model_dump(mode="json"))
        return outcome.report

    def prune(
        self,
        *,
        memory_ids: list[str] | None = None,
        min_utility: float | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete memories. The only destructive memory op — explicit by design.

        With ``memory_ids`` it prunes exactly those. Otherwise it prunes very
        low-utility, unprotected memories (constitutional/important are kept).
        """

        if memory_ids is not None:
            if not dry_run:
                for memory_id in memory_ids:
                    self.store.delete(memory_id)
            return list(memory_ids)

        threshold = (
            min_utility if min_utility is not None else self.config.prune_utility_threshold
        )
        pruned: list[str] = []
        for memory in self.store.all():
            protected = memory.memory_type == "constitutional" or bool(
                {t.lower() for t in memory.tags} & IMPORTANT_TAGS
            )
            if protected:
                continue
            if compute_utility_score(memory) < threshold:
                pruned.append(memory.id)
                if not dry_run:
                    self.store.delete(memory.id)
        return pruned

    # -- traces & import/export ----------------------------------------------

    def load_traces(self) -> list[dict[str, Any]]:
        return load_trace_records(self.config.traces_dir)

    def export_jsonl(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.config.memory_dir / "export.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for memory in self.store.all():
                handle.write(memory.model_dump_json() + "\n")
        return target

    def import_jsonl(self, path: str | Path) -> int:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        imported = 0
        new_memories: list[Memory] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                memory = Memory.model_validate(json.loads(line))
                self.store.upsert(memory)
                new_memories.append(memory)
                imported += 1
        self._embed_and_store(new_memories)
        return imported

    def count(self) -> int:
        return self.store.count()

    def close(self) -> None:
        self.store.close()
        self.skills.close()

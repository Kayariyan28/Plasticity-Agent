"""Sleep-like consolidation.

A software analogue of complementary learning systems + synaptic homeostasis.
During "sleep" the agent replays recent experience and:

1. **Decays** stale, rarely-used memories (forgetting curve).
2. **De-duplicates** near-identical memories, merging them into a survivor.
3. **Consolidates** clusters of related reflective memories into semantic gist
   (clustering can use lexical *or* embedding similarity).
4. **Mines** procedural memories and reusable skills from repeated successes.
5. **Detects contradictions** (heuristic or LLM-backed, via an injected pair fn).
6. **Governs the constitution** — flags memories that conflict with
   ``constitutional`` memories.
7. **Suggests** advisory prompt policies from recurring failures.

The module stays pure with respect to persistence, skills, and the LLM: it
returns a :class:`SleepOutcome` describing what should change, and the memory OS
applies it. All report numbers are real counts of work performed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from pydantic import BaseModel

from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.memory.contradiction import contradiction_pair
from plasticity_agent.memory.forgetting import decay_memories
from plasticity_agent.memory.retrieval import lexical_similarity, tokenize
from plasticity_agent.memory.salience import FAILURE_TAGS
from plasticity_agent.memory.schemas import Memory

_CONTRADICTION_THRESHOLD = 0.5
_DEDUP_THRESHOLD = 0.93

SimilarityFn = Callable[[Memory, Memory], float]
PairFn = Callable[[object, object], float]


class SleepReport(BaseModel):
    """Visible summary of one consolidation pass."""

    traces_analyzed: int = 0
    weak_memories_decayed: int = 0
    memories_consolidated: int = 0
    duplicates_merged: int = 0
    contradictions_detected: int = 0
    constitution_conflicts: int = 0
    skills_created: int = 0
    policies_improved: int = 0
    plasticity_score: float = 0.0
    summary: str = ""


@dataclass
class SkillCandidate:
    """A reusable skill proposed by consolidation (persisted by the caller)."""

    name: str
    description: str
    trigger_patterns: list[str]
    successful_trace: dict[str, Any]
    confidence: float
    reward: float


@dataclass
class SleepOutcome:
    """Everything a sleep cycle produced; the memory OS applies it."""

    report: SleepReport
    new_memories: list[Memory] = field(default_factory=list)
    removed_ids: list[str] = field(default_factory=list)
    skill_candidates: list[SkillCandidate] = field(default_factory=list)
    policy_suggestions: list[str] = field(default_factory=list)


def _lexical_similarity(a: Memory, b: Memory) -> float:
    return lexical_similarity(a.content, b.content)


def _slug(text: str, *, max_words: int = 4) -> str:
    words = tokenize(text)[:max_words]
    return "_".join(words) if words else "skill"


def _cluster(
    memories: list[Memory], threshold: float, similarity: SimilarityFn = _lexical_similarity
) -> list[list[Memory]]:
    """Greedy, order-stable single-link clustering by a similarity function."""

    clusters: list[list[Memory]] = []
    for memory in memories:
        for cluster in clusters:
            if similarity(memory, cluster[0]) >= threshold:
                cluster.append(memory)
                break
        else:
            clusters.append([memory])
    return clusters


def _dedup(
    memories: list[Memory], similarity: SimilarityFn, threshold: float = _DEDUP_THRESHOLD
) -> list[str]:
    """Merge near-duplicate (non-constitutional) memories into a survivor.

    Returns the ids of the absorbed duplicates (to be deleted by the caller).
    The survivor inherits tags, usage, and max salience; provenance is recorded.
    """

    candidates = [m for m in memories if m.memory_type != "constitutional"]
    removed: list[str] = []
    for cluster in _cluster(candidates, threshold, similarity):
        if len(cluster) < 2:
            continue
        survivor = max(cluster, key=lambda m: (m.salience, m.usage_count))
        for member in cluster:
            if member.id == survivor.id:
                continue
            survivor.tags = sorted(set(survivor.tags) | set(member.tags))
            survivor.usage_count += member.usage_count
            survivor.salience = max(survivor.salience, member.salience)
            survivor.metadata.setdefault("merged_from", []).append(member.id)
            removed.append(member.id)
    return removed


def _consolidate_reflective(
    memories: list[Memory], config: PlasticityConfig, similarity: SimilarityFn
) -> tuple[list[Memory], int]:
    """Fold clusters of reflective memories into semantic gist memories."""

    reflective = [m for m in memories if m.memory_type == "reflective"]
    new_memories: list[Memory] = []
    consolidated = 0

    for cluster in _cluster(reflective, config.consolidation_similarity, similarity):
        if len(cluster) < config.consolidation_min_cluster:
            continue
        representative = max(cluster, key=lambda m: m.salience)
        tags = sorted({tag for m in cluster for tag in m.tags} | {"semantic", "consolidated"})
        gist = Memory(
            content=f"Consolidated insight ({len(cluster)}x): {representative.content}",
            memory_type="semantic",
            tags=tags,
            salience=min(1.0, representative.salience + 0.1),
            confidence=mean(m.confidence for m in cluster),
            reward=mean(m.reward for m in cluster),
            usage_count=sum(m.usage_count for m in cluster),
            source_trace="sleep:consolidation",
            metadata={"consolidated_from": [m.id for m in cluster]},
        )
        new_memories.append(gist)
        for member in cluster:
            member.metadata["consolidated_into"] = gist.id
            member.salience = max(0.0, member.salience - 0.05)
            consolidated += 1

    return new_memories, consolidated


def _consolidate_successful_traces(
    trace_records: list[dict[str, Any]], config: PlasticityConfig
) -> tuple[list[Memory], list[SkillCandidate]]:
    """Turn repeated successful runs into procedural memories and skills."""

    successes: list[dict[str, Any]] = []
    for record in trace_records:
        if record.get("event_type") != "run_completed":
            continue
        payload = record.get("payload", {}) or {}
        reward = float(payload.get("reward", 0.0) or 0.0)
        task = str(payload.get("task") or payload.get("input") or "").strip()
        if task and reward >= 0.5:
            successes.append(
                {
                    "task": task,
                    "output": payload.get("output", ""),
                    "reward": reward,
                    "run_id": record.get("run_id", "unknown"),
                }
            )

    if not successes:
        return [], []

    wrapped = [Memory(content=item["task"], metadata={"item": item}) for item in successes]
    new_memories: list[Memory] = []
    candidates: list[SkillCandidate] = []

    for cluster in _cluster(wrapped, config.consolidation_similarity):
        if len(cluster) < config.consolidation_min_cluster:
            continue
        items = [m.metadata["item"] for m in cluster]
        rep = max(items, key=lambda it: it["reward"])
        avg_reward = mean(it["reward"] for it in items)
        confidence = min(1.0, 0.5 + 0.1 * len(items))
        procedure = Memory(
            content=(
                f"Procedure ({len(items)} successes): for tasks like "
                f"'{rep['task']}', repeat the validated approach."
            ),
            memory_type="procedural",
            tags=["procedural", "skill", "success"],
            salience=min(1.0, 0.6 + 0.05 * len(items)),
            confidence=confidence,
            reward=avg_reward,
            source_trace="sleep:procedural",
            metadata={"run_ids": [it["run_id"] for it in items]},
        )
        new_memories.append(procedure)
        candidates.append(
            SkillCandidate(
                name=_slug(rep["task"]),
                description=f"Reusable approach for: {rep['task']}",
                trigger_patterns=tokenize(rep["task"])[:6],
                successful_trace=rep,
                confidence=confidence,
                reward=avg_reward,
            )
        )

    return new_memories, candidates


def _detect_contradictions(memories: list[Memory], pair_fn: PairFn) -> int:
    """Update contradiction scores in place and count conflicting pairs."""

    conflicts = 0
    maxima: dict[str, float] = {m.id: 0.0 for m in memories}
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            score = pair_fn(memories[i], memories[j])
            maxima[memories[i].id] = max(maxima[memories[i].id], score)
            maxima[memories[j].id] = max(maxima[memories[j].id], score)
            if score >= _CONTRADICTION_THRESHOLD:
                conflicts += 1
    for memory in memories:
        memory.contradiction_score = maxima[memory.id]
    return conflicts


def _govern_constitution(memories: list[Memory], pair_fn: PairFn) -> int:
    """Flag non-constitutional memories that conflict with the constitution."""

    constitution = [m for m in memories if m.memory_type == "constitutional"]
    if not constitution:
        return 0
    conflicts = 0
    for memory in memories:
        if memory.memory_type == "constitutional":
            continue
        worst = max((pair_fn(memory, rule) for rule in constitution), default=0.0)
        if worst >= _CONTRADICTION_THRESHOLD:
            memory.metadata["constitution_conflict"] = round(worst, 3)
            memory.tags = sorted(set(memory.tags) | {"constitution_conflict"})
            conflicts += 1
    return conflicts


def _suggest_policies(
    memories: list[Memory], config: PlasticityConfig, similarity: SimilarityFn
) -> list[str]:
    """Advisory prompt-policy suggestions mined from recurring failures."""

    failures = [
        m for m in memories if ({tag.lower() for tag in m.tags} & FAILURE_TAGS) or m.reward < 0
    ]
    suggestions: list[str] = []
    for cluster in _cluster(failures, config.consolidation_similarity, similarity):
        if len(cluster) < config.consolidation_min_cluster:
            continue
        topic = " ".join(tokenize(cluster[0].content)[:5]) or "this situation"
        suggestions.append(
            f"POLICY (advisory): recurring failure around '{topic}' "
            f"({len(cluster)}x) — add an explicit pre-check and verify before finalizing."
        )
    return suggestions


def _plasticity_score(
    memories: list[Memory],
    consolidated: int,
    skills_created: int,
    utility_fn: Callable[[Memory], float] | None,
) -> float:
    if not memories:
        return 0.0
    avg_confidence = mean(m.confidence for m in memories)
    avg_contradiction = mean(m.contradiction_score for m in memories)
    consolidation_ratio = min(1.0, consolidated / max(1, len(memories)))
    skill_bonus = min(1.0, skills_created / 4.0)
    if utility_fn:
        avg_quality = mean(utility_fn(m) for m in memories)
    else:
        avg_quality = mean(m.salience for m in memories)
    composite = (
        0.35 * avg_quality
        + 0.20 * avg_confidence
        + 0.20 * (1.0 - avg_contradiction)
        + 0.15 * consolidation_ratio
        + 0.10 * skill_bonus
    )
    return round(100.0 * max(0.0, min(1.0, composite)), 1)


def consolidate_memories(
    memories: list[Memory],
    trace_records: list[dict[str, Any]] | None = None,
    *,
    config: PlasticityConfig | None = None,
    similarity_fn: SimilarityFn | None = None,
) -> tuple[list[Memory], int, list[SkillCandidate]]:
    """Consolidation only (no decay/contradiction): returns new memories,
    the count of source memories folded in, and any skill candidates."""

    config = config or PlasticityConfig()
    similarity = similarity_fn or _lexical_similarity
    semantic_memories, consolidated = _consolidate_reflective(memories, config, similarity)
    procedural_memories, candidates = _consolidate_successful_traces(
        trace_records or [], config
    )
    return semantic_memories + procedural_memories, consolidated, candidates


def run_sleep_cycle(
    memories: list[Memory],
    trace_records: list[dict[str, Any]] | None = None,
    *,
    config: PlasticityConfig | None = None,
    utility_fn: Callable[[Memory], float] | None = None,
    similarity_fn: SimilarityFn | None = None,
    pair_fn: PairFn | None = None,
) -> SleepOutcome:
    """Run a full consolidation pass over ``memories`` and ``trace_records``."""

    config = config or PlasticityConfig()
    trace_records = trace_records or []
    similarity = similarity_fn or _lexical_similarity
    pair = pair_fn or contradiction_pair

    _, decay_stats = decay_memories(memories, config=config, utility_fn=utility_fn)

    removed_ids = _dedup(memories, similarity)
    removed_set = set(removed_ids)
    active = [m for m in memories if m.id not in removed_set]

    semantic_memories, consolidated = _consolidate_reflective(active, config, similarity)
    procedural_memories, skill_candidates = _consolidate_successful_traces(trace_records, config)
    contradictions = _detect_contradictions(active, pair)
    constitution_conflicts = _govern_constitution(active, pair)
    policies = _suggest_policies(active, config, similarity)

    new_memories = semantic_memories + procedural_memories
    skills_created = len(skill_candidates)
    score = _plasticity_score(active + new_memories, consolidated, skills_created, utility_fn)

    summary = (
        f"Analyzed {len(trace_records)} traces; decayed {decay_stats.decayed}, "
        f"merged {len(removed_ids)} duplicates, consolidated {consolidated}, "
        f"found {contradictions} contradictions, {constitution_conflicts} constitution "
        f"conflicts, created {skills_created} skills, suggested {len(policies)} policies. "
        f"Plasticity score {score:.0f}/100."
    )

    report = SleepReport(
        traces_analyzed=len(trace_records),
        weak_memories_decayed=decay_stats.decayed,
        memories_consolidated=consolidated,
        duplicates_merged=len(removed_ids),
        contradictions_detected=contradictions,
        constitution_conflicts=constitution_conflicts,
        skills_created=skills_created,
        policies_improved=len(policies),
        plasticity_score=score,
        summary=summary,
    )
    return SleepOutcome(
        report=report,
        new_memories=new_memories,
        removed_ids=removed_ids,
        skill_candidates=skill_candidates,
        policy_suggestions=policies,
    )

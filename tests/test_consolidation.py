"""Tests for sleep-like consolidation (real counts, not faked)."""

from __future__ import annotations

from plasticity_agent.core.events import RUN_COMPLETED
from plasticity_agent.memory.memory_os import MemoryOS


def _seed(memory: MemoryOS) -> None:
    # Two near-identical reflective memories -> should consolidate into semantic.
    memory.record("The payment API call timed out", "reflective", tags=["failure"])
    memory.record("The payment API call timed out again", "reflective", tags=["failure"])
    # A contradicting pair.
    memory.record("I love coffee", "episodic")
    memory.record("I hate coffee", "episodic")
    # Two similar successful runs -> should mine a skill + procedural memory.
    memory.tracer.emit(RUN_COMPLETED, {"task": "summarize the quarterly report", "reward": 0.9})
    memory.tracer.emit(
        RUN_COMPLETED, {"task": "summarize the quarterly report carefully", "reward": 0.9}
    )


def test_sleep_returns_real_counts(memory: MemoryOS) -> None:
    _seed(memory)
    report = memory.sleep()
    assert report.traces_analyzed >= 2
    assert report.memories_consolidated >= 2
    assert report.contradictions_detected >= 1
    assert report.skills_created >= 1
    assert 0.0 <= report.plasticity_score <= 100.0
    assert report.summary


def test_sleep_creates_semantic_memory(memory: MemoryOS) -> None:
    _seed(memory)
    memory.sleep()
    semantic = memory.list_memories(memory_type="semantic")
    assert any("Consolidated insight" in m.content for m in semantic)


def test_sleep_persists_skills(memory: MemoryOS) -> None:
    _seed(memory)
    memory.sleep()
    assert memory.skills.count() >= 1


def test_consolidate_only_does_not_raise(memory: MemoryOS) -> None:
    _seed(memory)
    consolidated = memory.consolidate()
    assert consolidated >= 2


def test_sleep_on_empty_memory_is_safe(memory: MemoryOS) -> None:
    report = memory.sleep()
    assert report.traces_analyzed == 0
    assert report.memories_consolidated == 0
    assert report.plasticity_score == 0.0


def test_sleep_merges_duplicates(memory: MemoryOS) -> None:
    memory.record("The deployment succeeded on the first attempt", "episodic")
    memory.record("The deployment succeeded on the first attempt", "episodic")
    before = memory.count()
    report = memory.sleep()
    assert report.duplicates_merged >= 1
    assert memory.count() < before + report.memories_consolidated + 5  # dup was removed


def test_sleep_flags_constitution_conflicts(memory: MemoryOS) -> None:
    memory.record("Always deploy on Friday afternoons", "constitutional", tags=["important"])
    memory.record("Never deploy on Friday afternoons under any circumstances", "episodic")
    report = memory.sleep()
    assert report.constitution_conflicts >= 1
    flagged = [
        m for m in memory.list_memories() if "constitution_conflict" in m.tags
    ]
    assert flagged

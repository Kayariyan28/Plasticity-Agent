"""Tests for the MemoryOS façade: record, recall, evaluate, decay, prune, IO."""

from __future__ import annotations

from plasticity_agent.memory.memory_os import MemoryOS


def test_record_and_get(memory: MemoryOS) -> None:
    record = memory.record("remember this", "semantic", tags=["important"])
    assert memory.get(record.id).content == "remember this"
    assert memory.count() == 1


def test_recall_finds_and_increments_usage(memory: MemoryOS) -> None:
    memory.record("Cache warmup reduced p95 latency significantly", "semantic")
    results = memory.recall("latency cache warmup")
    assert results
    assert results[0].score > 0
    assert memory.get(results[0].memory.id).usage_count >= 1


def test_search_is_side_effect_free(memory: MemoryOS) -> None:
    memory.record("alpha beta gamma delta", "episodic")
    memory.search("alpha beta")
    assert memory.list_memories()[0].usage_count == 0


def test_evaluate_memory_and_all(memory: MemoryOS) -> None:
    record = memory.record("important reusable fact", "semantic", tags=["important"], reward=0.9)
    report = memory.evaluate_memory(record)
    assert 0.0 <= report.utility_score <= 1.0
    assert report.recommendation in {"keep", "decay", "consolidate", "review", "prune"}
    assert len(memory.evaluate_all()) == 1


def test_decay_reduces_low_quality_salience(memory: MemoryOS) -> None:
    record = memory.record("forgettable trivia", "episodic", salience=0.5, confidence=0.3)
    before = memory.get(record.id).salience
    memory.decay(days_passed=60)
    after = memory.get(record.id).salience
    assert after < before


def test_constitutional_memory_resists_decay(memory: MemoryOS) -> None:
    record = memory.record("core principle", "constitutional", tags=["important"], salience=0.6)
    before = memory.get(record.id).salience
    memory.decay(days_passed=120)
    assert memory.get(record.id).salience == before


def test_export_import_roundtrip(memory: MemoryOS, tmp_path) -> None:
    memory.record("one")
    memory.record("two")
    path = memory.export_jsonl(tmp_path / "dump.jsonl")

    other = MemoryOS(memory_dir=str(tmp_path / "other"))
    try:
        imported = other.import_jsonl(path)
        assert imported == 2
        assert other.count() == 2
    finally:
        other.close()


def test_prune_is_explicit_and_protects_important(memory: MemoryOS) -> None:
    weak = memory.record("trivial low value note", "episodic", confidence=0.1, salience=0.05)
    protected = memory.record("core principle", "constitutional", tags=["important"])
    pruned = memory.prune(min_utility=0.9)
    assert weak.id in pruned
    assert protected.id not in pruned
    assert memory.get(weak.id) is None
    assert memory.get(protected.id) is not None

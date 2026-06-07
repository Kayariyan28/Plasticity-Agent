"""Tests for the core memory Pydantic schemas."""

from __future__ import annotations

from datetime import datetime

from plasticity_agent.memory.schemas import (
    Memory,
    MemoryQualityReport,
    MemorySearchResult,
)


def test_memory_defaults_and_unit_clamping() -> None:
    memory = Memory(
        content="hello",
        salience=2.0,
        confidence=-1.0,
        contradiction_score=5.0,
        decay_rate=-0.5,
    )
    assert memory.salience == 1.0
    assert memory.confidence == 0.0
    assert memory.contradiction_score == 1.0
    assert memory.decay_rate == 0.0
    assert memory.memory_type == "episodic"
    assert memory.id.startswith("mem_")
    assert isinstance(memory.created_at, datetime)
    assert memory.tags == []
    assert memory.metadata == {}


def test_memory_age_is_non_negative() -> None:
    assert Memory(content="x").age_days() >= 0.0


def test_memory_roundtrip_json() -> None:
    memory = Memory(content="roundtrip", tags=["a", "b"], memory_type="semantic")
    restored = Memory.model_validate_json(memory.model_dump_json())
    assert restored.id == memory.id
    assert restored.content == "roundtrip"
    assert restored.memory_type == "semantic"


def test_quality_report_model() -> None:
    report = MemoryQualityReport(
        memory_id="m1",
        utility_score=0.5,
        salience=0.5,
        confidence=0.5,
        contradiction_score=0.1,
        decay_rate=0.0,
        usage_count=1,
        recommendation="keep",
        reasons=["healthy"],
    )
    assert report.recommendation == "keep"
    assert report.reasons == ["healthy"]


def test_search_result_model() -> None:
    result = MemorySearchResult(memory=Memory(content="x"), score=0.9, match_reason="overlap")
    assert result.memory.content == "x"
    assert result.score == 0.9
    assert result.match_reason == "overlap"

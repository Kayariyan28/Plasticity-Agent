"""Pydantic data models for the plasticity memory subsystem.

These schemas are the contract shared across the whole framework: the memory
store, quality evaluation, consolidation, retrieval, and the agent runtime all
speak in terms of :class:`Memory` and its companions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MemoryType = Literal[
    "episodic",
    "semantic",
    "procedural",
    "reflective",
    "constitutional",
]

Recommendation = Literal["keep", "decay", "consolidate", "review", "prune"]


def utcnow() -> datetime:
    """Timezone-aware UTC ``now`` (used as the single time source)."""

    return datetime.now(UTC)


def new_memory_id() -> str:
    """Generate a short, collision-resistant memory id."""

    return f"mem_{uuid.uuid4().hex[:12]}"


class Memory(BaseModel):
    """A single unit of agent memory.

    The fields beyond ``content`` are *plasticity signals*: they describe how
    valuable, trustworthy, contested, and fresh a memory is. Higher-level
    components (quality scoring, decay, consolidation) read and update them.
    """

    id: str = Field(default_factory=new_memory_id)
    content: str
    memory_type: MemoryType = "episodic"
    tags: list[str] = Field(default_factory=list)
    salience: float = 0.5
    confidence: float = 0.7
    usage_count: int = 0
    contradiction_score: float = 0.0
    decay_rate: float = 0.0
    reward: float = 0.0
    source_trace: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("salience", "confidence", "contradiction_score", "decay_rate")
    @classmethod
    def _clamp_unit(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def touch(self) -> None:
        """Bump ``updated_at`` to the current time."""

        self.updated_at = utcnow()

    def age_days(self, *, now: datetime | None = None) -> float:
        """Age of the memory in fractional days since it was last updated."""

        reference = now or utcnow()
        anchor = self.updated_at
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=UTC)
        return max(0.0, (reference - anchor).total_seconds() / 86400.0)


class MemoryQualityReport(BaseModel):
    """Deterministic quality assessment of a single memory."""

    memory_id: str
    utility_score: float
    salience: float
    confidence: float
    contradiction_score: float
    decay_rate: float
    usage_count: int
    recommendation: Recommendation
    reasons: list[str] = Field(default_factory=list)


class MemorySearchResult(BaseModel):
    """A scored search hit returned by retrieval."""

    memory: Memory
    score: float
    match_reason: str

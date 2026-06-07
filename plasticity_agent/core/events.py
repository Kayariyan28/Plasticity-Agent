"""Trace event schema and the canonical set of event types.

Every meaningful action a Plasticity agent takes is recorded as a
:class:`TraceEvent` and appended to a daily JSONL log. Traces are the raw
material that sleep/consolidation and the energy report analyse later.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


# Canonical event types (kept as plain strings so traces stay forward-compatible).
RUN_STARTED = "run_started"
RUN_COMPLETED = "run_completed"
RUN_FAILED = "run_failed"
MEMORY_RECORDED = "memory_recorded"
MEMORY_RECALLED = "memory_recalled"
REFLECTION_CREATED = "reflection_created"
HEALING_DIAGNOSED = "healing_diagnosed"
REASONING_AUCTION = "reasoning_auction"
SLEEP_COMPLETED = "sleep_completed"
ENERGY_REPORT_CREATED = "energy_report_created"

EVENT_TYPES = (
    RUN_STARTED,
    RUN_COMPLETED,
    RUN_FAILED,
    MEMORY_RECORDED,
    MEMORY_RECALLED,
    REFLECTION_CREATED,
    HEALING_DIAGNOSED,
    REASONING_AUCTION,
    SLEEP_COMPLETED,
    ENERGY_REPORT_CREATED,
)


class TraceEvent(BaseModel):
    """A single structured event in an agent run."""

    id: str = Field(default_factory=new_event_id)
    run_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=_utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)

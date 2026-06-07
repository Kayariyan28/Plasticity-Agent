"""Neuroplastic memory: schemas, store, scoring, decay, consolidation, retrieval."""

from __future__ import annotations

from plasticity_agent.memory.consolidation import SleepReport, run_sleep_cycle
from plasticity_agent.memory.contradiction import detect_contradiction
from plasticity_agent.memory.forgetting import decay_memories
from plasticity_agent.memory.memory_os import (
    MemoryOS,
    compute_utility_score,
    score_memory_quality,
)
from plasticity_agent.memory.retrieval import lexical_similarity, search_memories
from plasticity_agent.memory.salience import calculate_salience
from plasticity_agent.memory.schemas import (
    Memory,
    MemoryQualityReport,
    MemorySearchResult,
)
from plasticity_agent.memory.store import MemoryStore

__all__ = [
    "MemoryOS",
    "MemoryStore",
    "Memory",
    "MemoryQualityReport",
    "MemorySearchResult",
    "SleepReport",
    "run_sleep_cycle",
    "decay_memories",
    "detect_contradiction",
    "calculate_salience",
    "compute_utility_score",
    "score_memory_quality",
    "lexical_similarity",
    "search_memories",
]

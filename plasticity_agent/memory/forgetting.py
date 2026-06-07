"""Memory decay and forgetting.

A software analogue of synaptic homeostasis: memories that are rarely used and
have gone stale lose salience over time, while important and constitutional
memories are protected. Decay is *non-destructive* — it only weakens signals
and flags prune candidates. Nothing is ever deleted here.

The optional ``utility_fn`` lets the caller (the memory OS) inject the full
quality score; without it we fall back to raw salience, keeping this module
free of upward dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.memory.salience import IMPORTANT_TAGS
from plasticity_agent.memory.schemas import Memory, utcnow


@dataclass
class DecayStats:
    """Outcome of a decay pass."""

    decayed: int = 0
    prune_candidates: int = 0


def _is_protected(memory: Memory) -> bool:
    if memory.memory_type == "constitutional":
        return True
    return bool({tag.lower() for tag in memory.tags} & IMPORTANT_TAGS)


def decay_memories(
    memories: list[Memory],
    *,
    days_passed: int | None = None,
    config: PlasticityConfig | None = None,
    utility_fn: Callable[[Memory], float] | None = None,
    now: datetime | None = None,
) -> tuple[list[Memory], DecayStats]:
    """Apply the forgetting curve in place and return ``(memories, stats)``.

    ``days_passed`` simulates a fixed elapsed time for every memory (useful for
    tests and "what-if" runs). When ``None``, each memory's real age is used.
    Decay never touches ``updated_at`` — that would falsely refresh recency.
    """

    config = config or PlasticityConfig()
    reference = now or utcnow()
    stats = DecayStats()

    for memory in memories:
        age = float(days_passed) if days_passed is not None else memory.age_days(now=reference)
        low_usage = memory.usage_count <= config.low_usage_threshold
        stale = age >= config.decay_age_days_threshold
        protected = _is_protected(memory)

        changed = False
        if low_usage and stale and not protected:
            growth = config.decay_increment * max(1.0, age / config.decay_age_days_threshold)
            new_decay = min(1.0, memory.decay_rate + growth)
            if new_decay > memory.decay_rate:
                memory.decay_rate = new_decay
                changed = True
            reduced = max(0.0, memory.salience * (1.0 - min(0.9, memory.decay_rate)))
            if reduced < memory.salience:
                memory.salience = reduced
                changed = True
        if changed:
            stats.decayed += 1

        utility = utility_fn(memory) if utility_fn else memory.salience
        is_candidate = (
            utility < config.prune_utility_threshold
            and memory.usage_count <= config.low_usage_threshold
            and not protected
        )
        memory.metadata["prune_candidate"] = is_candidate
        if is_candidate:
            stats.prune_candidates += 1

    return memories, stats

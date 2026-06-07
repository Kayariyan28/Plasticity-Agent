"""The energy report — a thermodynamic-style reliability snapshot.

Assembles entropy, contradiction pressure, estimated wasted compute, repair
burden, and confidence "temperature" into a single :class:`EnergyReport` with a
composite plasticity score. Everything is derived from real memory state and
trace logs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from plasticity_agent.core.events import RUN_FAILED
from plasticity_agent.healing.detector import detect_failures
from plasticity_agent.memory.schemas import Memory
from plasticity_agent.thermodynamics.entropy import salience_entropy
from plasticity_agent.thermodynamics.free_energy import (
    ConfidenceTemperature,
    RepairEnergy,
    confidence_temperature,
    free_energy,
    repair_energy,
)

_HIGH_SEVERITY = {"permission_error", "unknown"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class EnergyReport(BaseModel):
    """A thermodynamic-style reliability report for an agent."""

    memory_entropy: float
    contradiction_pressure: float
    token_waste: float
    repair_energy: RepairEnergy
    confidence_temperature: ConfidenceTemperature
    plasticity_score: float
    summary: str


def estimate_token_waste(trace_records: list[dict[str, Any]]) -> tuple[float, float]:
    """Estimate wasted tokens and the wasted/total ratio from traces.

    Failed runs are counted as wasted work; repeated identical tasks count their
    redundant copies as waste. Characters are converted to tokens at ~4:1.
    """

    total_chars = 0
    wasted_chars = 0
    seen_tasks: dict[str, int] = {}

    for record in trace_records:
        payload = record.get("payload", {}) or {}
        text = json.dumps(payload, default=str)
        total_chars += len(text)
        event_type = record.get("event_type")
        if event_type == RUN_FAILED:
            wasted_chars += len(text)
        elif event_type == "run_completed":
            task = str(payload.get("task") or payload.get("input") or "")
            if task:
                if task in seen_tasks:
                    wasted_chars += len(task)
                seen_tasks[task] = seen_tasks.get(task, 0) + 1

    tokens = round(wasted_chars / 4.0, 1)
    ratio = _clamp01(wasted_chars / total_chars) if total_chars else 0.0
    return tokens, ratio


def build_energy_report(
    memories: Iterable[Memory],
    trace_records: list[dict[str, Any]] | None = None,
) -> EnergyReport:
    """Compute a full :class:`EnergyReport` from memory state and traces."""

    memories = list(memories)
    trace_records = trace_records or []

    entropy = salience_entropy(memories)
    contradiction_pressure = _mean([m.contradiction_score for m in memories])
    wasted_tokens, waste_ratio = estimate_token_waste(trace_records)

    diagnoses = detect_failures(trace_records)
    high_severity = sum(1 for d in diagnoses if d.failure_type in _HIGH_SEVERITY)
    repair = repair_energy(len(diagnoses), high_severity=high_severity)

    temperature = confidence_temperature([m.confidence for m in memories])
    energy = free_energy(
        entropy=entropy,
        contradiction_pressure=contradiction_pressure,
        token_waste_ratio=waste_ratio,
        temperature=temperature,
    )
    plasticity_score = round(100.0 * _clamp01(1.0 - energy), 1)

    summary = (
        f"entropy={entropy:.2f}, contradiction={contradiction_pressure:.2f}, "
        f"wasted≈{wasted_tokens:.0f} tokens, repair={repair}, "
        f"temperature={temperature}. Plasticity {plasticity_score:.0f}/100 "
        f"(free-energy {energy:.2f})."
    )

    return EnergyReport(
        memory_entropy=round(entropy, 4),
        contradiction_pressure=round(contradiction_pressure, 4),
        token_waste=wasted_tokens,
        repair_energy=repair,
        confidence_temperature=temperature,
        plasticity_score=plasticity_score,
        summary=summary,
    )

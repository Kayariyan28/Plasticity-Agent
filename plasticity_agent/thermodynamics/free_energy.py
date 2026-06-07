"""Free-energy-style metaphors.

A loose software analogue of the free-energy principle: an agent is "healthier"
when it minimises contradiction, disorder (entropy), wasted compute, and
confidence instability. These functions quantify those pressures so the energy
report can summarise them. This is a metaphor, not physics.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import pstdev
from typing import Literal

ConfidenceTemperature = Literal["stable", "warm", "unstable"]
RepairEnergy = Literal["low", "medium", "high"]

_INSTABILITY = {"stable": 0.0, "warm": 0.5, "unstable": 1.0}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def confidence_temperature(confidences: Sequence[float]) -> ConfidenceTemperature:
    """Map confidence variance to a 'temperature'.

    Low spread => ``stable``; moderate => ``warm``; high => ``unstable``.
    """

    values = list(confidences)
    if len(values) < 2:
        return "stable"
    spread = pstdev(values)
    if spread < 0.12:
        return "stable"
    if spread < 0.22:
        return "warm"
    return "unstable"


def repair_energy(num_failures: int, *, high_severity: int = 0) -> RepairEnergy:
    """Qualitative repair burden from failure count and severity."""

    score = num_failures + 2 * high_severity
    if score <= 1:
        return "low"
    if score <= 4:
        return "medium"
    return "high"


def free_energy(
    *,
    entropy: float,
    contradiction_pressure: float,
    token_waste_ratio: float,
    temperature: ConfidenceTemperature,
) -> float:
    """Composite 'free energy' in ``[0, 1]`` — lower is healthier."""

    instability = _INSTABILITY[temperature]
    composite = (
        0.30 * _clamp01(entropy)
        + 0.35 * _clamp01(contradiction_pressure)
        + 0.20 * _clamp01(token_waste_ratio)
        + 0.15 * instability
    )
    return _clamp01(composite)

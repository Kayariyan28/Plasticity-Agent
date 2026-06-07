"""Thermodynamic-style reliability: entropy, free energy, and energy reports."""

from __future__ import annotations

from plasticity_agent.thermodynamics.energy_report import (
    EnergyReport,
    build_energy_report,
    estimate_token_waste,
)
from plasticity_agent.thermodynamics.entropy import normalized_entropy, salience_entropy
from plasticity_agent.thermodynamics.free_energy import (
    confidence_temperature,
    free_energy,
    repair_energy,
)

__all__ = [
    "EnergyReport",
    "build_energy_report",
    "estimate_token_waste",
    "salience_entropy",
    "normalized_entropy",
    "confidence_temperature",
    "free_energy",
    "repair_energy",
]

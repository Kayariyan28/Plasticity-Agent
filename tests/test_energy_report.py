"""Tests for the thermodynamic-style energy report."""

from __future__ import annotations

from plasticity_agent.memory.schemas import Memory
from plasticity_agent.thermodynamics.energy_report import (
    build_energy_report,
    estimate_token_waste,
)


def test_energy_report_has_valid_ranges() -> None:
    memories = [
        Memory(content="a", salience=0.8, confidence=0.7, contradiction_score=0.1),
        Memory(content="b", salience=0.3, confidence=0.6, contradiction_score=0.4),
    ]
    traces = [
        {
            "event_type": "run_completed",
            "payload": {"task": "t", "output": "x" * 100, "reward": 0.9},
        },
        {"event_type": "run_failed", "payload": {"error": "TimeoutError: timed out"}},
    ]
    report = build_energy_report(memories, traces)
    assert 0.0 <= report.memory_entropy <= 1.0
    assert 0.0 <= report.contradiction_pressure <= 1.0
    assert report.token_waste >= 0.0
    assert report.repair_energy in {"low", "medium", "high"}
    assert report.confidence_temperature in {"stable", "warm", "unstable"}
    assert 0.0 <= report.plasticity_score <= 100.0
    assert report.summary


def test_empty_inputs_are_safe() -> None:
    report = build_energy_report([], [])
    assert report.memory_entropy == 0.0
    assert report.contradiction_pressure == 0.0
    assert 0.0 <= report.plasticity_score <= 100.0


def test_token_waste_counts_failures() -> None:
    tokens, ratio = estimate_token_waste(
        [{"event_type": "run_failed", "payload": {"error": "x" * 40}}]
    )
    assert tokens > 0.0
    assert 0.0 <= ratio <= 1.0


def test_unstable_temperature_from_confidence_spread() -> None:
    memories = [
        Memory(content="a", confidence=0.05),
        Memory(content="b", confidence=0.95),
        Memory(content="c", confidence=0.1),
        Memory(content="d", confidence=0.9),
    ]
    report = build_energy_report(memories, [])
    assert report.confidence_temperature in {"warm", "unstable"}

"""Tests for deterministic salience scoring."""

from __future__ import annotations

from plasticity_agent.memory.salience import calculate_salience


def test_salience_always_in_unit_range() -> None:
    for kwargs in (
        {},
        {"reward": 1.0, "tags": ["critical"], "recurrence": 9, "confidence": 1.0},
        {"reward": -1.0},
    ):
        assert 0.0 <= calculate_salience("content", **kwargs) <= 1.0


def test_reward_magnitude_increases_salience() -> None:
    assert calculate_salience("x", reward=1.0) > calculate_salience("x", reward=0.0)
    # Negative reward is also salient (costly failures are worth remembering).
    assert calculate_salience("x", reward=-1.0) > calculate_salience("x", reward=0.0)


def test_important_and_failure_tags_boost() -> None:
    base = calculate_salience("note")
    assert calculate_salience("note", tags=["important"]) > base
    assert calculate_salience("note", tags=["user_preference"]) > base
    assert calculate_salience("note", tags=["failure"]) > base


def test_recurrence_and_confidence_increase() -> None:
    assert calculate_salience("x", recurrence=10) > calculate_salience("x", recurrence=1)
    assert calculate_salience("x", confidence=1.0) > calculate_salience("x", confidence=0.0)


def test_emphasis_words_add_weight() -> None:
    assert calculate_salience("always do this") > calculate_salience("do this")

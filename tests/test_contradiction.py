"""Tests for the deterministic contradiction detector."""

from __future__ import annotations

from plasticity_agent.memory.contradiction import contradiction_pair, detect_contradiction


def test_score_in_unit_range() -> None:
    score = detect_contradiction("I love coffee", ["I hate coffee", "the sky is blue"])
    assert 0.0 <= score <= 1.0


def test_opposite_sentiment_is_high() -> None:
    assert contradiction_pair("I love coffee", "I hate coffee") > 0.4


def test_negation_conflict_detected() -> None:
    assert contradiction_pair("The server is reachable", "The server is not reachable") > 0.3


def test_unrelated_is_low() -> None:
    assert contradiction_pair("I love coffee", "The sky is blue today") < 0.2


def test_no_existing_memories_returns_zero() -> None:
    assert detect_contradiction("anything at all", []) == 0.0


def test_returns_maximum_over_candidates() -> None:
    score = detect_contradiction(
        "We should always deploy on Friday",
        ["unrelated note about cats", "We should never deploy on Friday"],
    )
    assert score > 0.3

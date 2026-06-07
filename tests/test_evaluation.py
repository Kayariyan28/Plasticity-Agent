"""Regression guards derived from the critical evaluation.

These lock in the v0.2.1 fixes: contradiction recall (stemmed antonyms +
numeric/temporal conflict), precision (no new false positives), and a
non-gameable improvement metric.
"""

from __future__ import annotations

from plasticity_agent import PlasticAgent
from plasticity_agent.memory.contradiction import contradiction_pair

# Pairs that SHOULD be flagged as contradictions (score >= 0.5).
CONTRA_POSITIVE = [
    ("The server is reachable", "The server is not reachable"),        # negation
    ("I love coffee", "I hate coffee"),                                # antonym/sentiment
    ("The feature is enabled", "The feature is disabled"),             # inflected antonym
    ("Sales increased this quarter", "Sales decreased this quarter"),  # inflected antonym
    ("The build passed", "The build failed"),                          # pass/fail
    ("Revenue grew in Q1", "Revenue fell in Q1"),                      # irregular antonym
    ("The API p95 latency is 80ms", "The API p95 latency is 800ms"),   # numeric
    ("The meeting is at 3pm", "The meeting is at 4pm"),                # temporal
]

# Pairs that should NOT be flagged (score < 0.5).
CONTRA_NEGATIVE = [
    ("The cat is on the mat", "A feline rests on the rug"),
    ("The sky is blue", "Grass is green"),
    ("Python is great for data", "Python is excellent for analytics"),
    ("Cats are mammals", "Cats are mammals"),
    ("We have 3 servers", "We have 3 databases"),
]


def test_contradiction_recall_is_high() -> None:
    detected = sum(contradiction_pair(a, b) >= 0.5 for a, b in CONTRA_POSITIVE)
    assert detected >= 7  # >= 7/8 of the realistic battery


def test_contradiction_precision_no_false_positives() -> None:
    false_positives = sum(contradiction_pair(a, b) >= 0.5 for a, b in CONTRA_NEGATIVE)
    assert false_positives == 0


def test_inflected_antonyms_now_detected() -> None:
    assert contradiction_pair("The feature is enabled", "The feature is disabled") >= 0.5
    assert contradiction_pair("Sales increased", "Sales decreased") >= 0.5


def test_numeric_and_temporal_conflict_detected() -> None:
    assert contradiction_pair("p95 latency is 80ms", "p95 latency is 800ms") >= 0.5
    assert contradiction_pair("The meeting is at 3pm", "The meeting is at 4pm") >= 0.5


def test_numeric_conflict_does_not_false_positive_on_different_nouns() -> None:
    assert contradiction_pair("We have 3 servers", "We have 3 databases") < 0.5


def test_improvement_metric_not_gameable_by_salience_spam(tmp_path) -> None:
    agent = PlasticAgent(name="game", memory=str(tmp_path / "m"), reasoning_market=False)
    try:
        agent.remember("baseline note", "episodic", confidence=0.5)
        agent.checkpoint("before")
        for i in range(5):
            agent.remember(f"shiny fact {i}", "semantic", tags=["important"],
                           reward=1.0, confidence=1.0)
        agent.checkpoint("after")
        # Storing unused high-salience memories must NOT register as improvement.
        assert agent.improvement().improved is False
    finally:
        agent.close()


def test_improvement_metric_detects_genuine_contradiction_resolution(tmp_path) -> None:
    agent = PlasticAgent(name="real", memory=str(tmp_path / "m"), reasoning_market=False)
    try:
        agent.remember("X is true", "semantic", confidence=0.6)
        conflicting = agent.remember("X is not true", "semantic", confidence=0.6)
        agent.checkpoint("before")
        agent.memory.prune(memory_ids=[conflicting.id])
        agent.checkpoint("after")
        assert agent.improvement().improved is True
    finally:
        agent.close()

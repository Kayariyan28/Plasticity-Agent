"""Confidence utilities for the reasoning market.

These translate a set of proposal scores into an overall selection confidence —
notably the *margin* between the winner and the runner-up, which is a more
honest signal than the winner's raw score alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import pstdev


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def aggregate_confidence(values: Sequence[float]) -> float:
    """Mean of confidence values (0.0 for an empty sequence)."""

    values = list(values)
    return clamp01(sum(values) / len(values)) if values else 0.0


def selection_confidence(sorted_scores: Sequence[float]) -> float:
    """Confidence in the *winner*, blending its score with its margin.

    A clear winner (high score, big gap to second place) yields high
    confidence; a near-tie yields low confidence even if the top score is high.
    """

    scores = list(sorted_scores)
    if not scores:
        return 0.0
    top = scores[0]
    if len(scores) == 1:
        return clamp01(0.5 + 0.5 * top)
    margin = max(0.0, top - scores[1])
    return clamp01(0.6 * top + 0.4 * min(1.0, margin * 3.0))


def disagreement(scores: Sequence[float]) -> float:
    """Spread of scores across critics (population stdev), clamped to ``[0, 1]``."""

    scores = list(scores)
    return clamp01(pstdev(scores)) if len(scores) > 1 else 0.0

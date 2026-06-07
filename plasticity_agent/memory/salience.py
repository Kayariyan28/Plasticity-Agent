"""Deterministic salience scoring.

Salience answers "how much should this memory grab attention later?" It is a
bounded blend of reward magnitude, confidence, recurrence, and explicit
importance/failure tags. No model call is involved — this is the v0.1.0
deterministic baseline; an LLM-scored variant is future work.
"""

from __future__ import annotations

from collections.abc import Sequence

IMPORTANT_TAGS = {
    "important",
    "critical",
    "user_preference",
    "preference",
    "key",
    "constitutional",
    "pinned",
}

FAILURE_TAGS = {
    "failure",
    "failed",
    "error",
    "bug",
    "regression",
    "incident",
    "fault",
    "risk",
}

_EMPHASIS = ("always", "never", "must", "do not", "don't", "remember", "important")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_salience(
    content: str,
    reward: float = 0.0,
    tags: Sequence[str] | None = None,
    confidence: float = 0.7,
    recurrence: int = 1,
) -> float:
    """Compute a salience score in ``[0, 1]``.

    Both positive and negative reward raise salience — a costly failure is as
    worth remembering as a big win, which is why we use ``abs(reward)``.
    """

    normalized_tags = [t.lower() for t in (tags or [])]
    lowered = (content or "").lower()

    score = 0.10  # baseline so every memory has a little pull
    score += 0.20 * _clamp01(confidence)
    score += 0.22 * min(abs(reward), 1.0)
    score += 0.16 * min(max(recurrence, 1) / 5.0, 1.0)

    if any(tag in IMPORTANT_TAGS for tag in normalized_tags):
        score += 0.20
    if any(tag in FAILURE_TAGS for tag in normalized_tags):
        score += 0.14
    if any(marker in lowered for marker in _EMPHASIS):
        score += 0.06

    return _clamp01(score)

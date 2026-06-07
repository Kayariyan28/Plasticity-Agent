"""Reward shaping helpers.

Small, deterministic utilities for turning heterogeneous run signals (success
flag, error presence, evaluator score) into a scalar reward in ``[-1, 1]`` and
for normalising rewards for the quality formula.
"""

from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def shape_reward(
    *,
    success: bool | None = None,
    error: bool | str | None = None,
    evaluator_score: float | None = None,
    base: float = 0.0,
) -> float:
    """Combine signals into a reward in ``[-1, 1]``."""

    reward = base
    if success is True:
        reward += 0.7
    elif success is False:
        reward -= 0.7
    if error:
        reward -= 0.5
    if evaluator_score is not None:
        # Map an evaluator score in [0, 1] to a contribution in [-0.5, 0.5].
        reward += (_clamp(evaluator_score, 0.0, 1.0) * 2.0 - 1.0) * 0.5
    return _clamp(reward, -1.0, 1.0)


def normalize_reward(reward: float) -> float:
    """Map a reward in ``[-1, 1]`` to ``[0, 1]``."""

    return _clamp((reward + 1.0) / 2.0, 0.0, 1.0)


def positive_reward(reward: float) -> float:
    """The non-negative part of a reward, clamped to ``[0, 1]``."""

    return _clamp(reward, 0.0, 1.0)

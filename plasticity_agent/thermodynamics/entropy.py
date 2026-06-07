"""Entropy measures over memory.

We use Shannon entropy of the salience distribution as a proxy for "how
disordered / noisy is this memory store?" — binned and normalised to ``[0, 1]``
so it is comparable across runs of different sizes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from plasticity_agent.memory.schemas import Memory


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy of a probability vector, normalised to ``[0, 1]``."""

    probs = [p for p in probabilities if p > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return _clamp01(entropy / math.log(len(probs)))


def salience_entropy(memories: Iterable[Memory], *, bins: int = 10) -> float:
    """Normalised entropy of the salience histogram across ``bins`` buckets."""

    saliences = [memory.salience for memory in memories]
    if len(saliences) < 2:
        return 0.0
    counts = [0] * bins
    for value in saliences:
        index = min(bins - 1, max(0, int(value * bins)))
        counts[index] += 1
    total = sum(counts)
    probs = [count / total for count in counts if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return _clamp01(entropy / math.log(bins))

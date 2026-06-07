"""Lightweight, LLM-free contradiction detection.

This is a *deterministic baseline*, not a truth oracle. A contradiction needs
two things at once: topical relatedness (the memories are about the same thing)
and semantic opposition (negation mismatch, antonyms, or a sentiment flip).
Either signal alone scores ~0. The product of the two is returned in ``[0, 1]``.

A future LLM-backed entailment checker is planned; until then this catches the
common, obvious conflicts and is honest about the rest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from plasticity_agent.llm.client import LLMClient, clamp01, coerce_llm, complete_json
from plasticity_agent.memory.retrieval import lexical_similarity, token_set
from plasticity_agent.memory.schemas import Memory

NEGATIONS = {
    "not", "no", "never", "none", "cannot", "cant", "wont", "dont", "doesnt",
    "isnt", "arent", "shouldnt", "wasnt", "werent", "without", "neither",
    "nor", "fails", "avoid", "stop",
}

_ANTONYM_PAIRS = [
    ("increase", "decrease"), ("increase", "reduce"), ("rise", "fall"),
    ("high", "low"), ("good", "bad"), ("like", "dislike"), ("love", "hate"),
    ("enable", "disable"), ("allow", "deny"), ("allow", "forbid"),
    ("true", "false"), ("yes", "no"), ("success", "failure"),
    ("accept", "reject"), ("start", "stop"), ("open", "close"),
    ("add", "remove"), ("include", "exclude"), ("always", "never"),
    ("fast", "slow"), ("safe", "unsafe"), ("prefer", "avoid"),
    ("more", "less"), ("better", "worse"), ("up", "down"), ("on", "off"),
    ("correct", "incorrect"), ("valid", "invalid"), ("work", "broken"),
]

POSITIVE_WORDS = {
    "good", "great", "best", "love", "like", "prefer", "success", "correct",
    "safe", "fast", "reliable", "improve", "better", "works", "working",
    "pass", "passed", "valid", "enable", "allow",
}

NEGATIVE_WORDS = {
    "bad", "worst", "hate", "dislike", "avoid", "failure", "wrong", "unsafe",
    "slow", "unreliable", "broken", "worse", "fails", "failing", "fail",
    "failed", "error", "invalid", "disable", "deny",
}


def _antonyms() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for left, right in _ANTONYM_PAIRS:
        mapping.setdefault(left, set()).add(right)
        mapping.setdefault(right, set()).add(left)
    return mapping


_ANTONYM_MAP = _antonyms()


def _content_of(item: object) -> str:
    if isinstance(item, Memory):
        return item.content
    return str(item)


def _tags_of(item: object) -> set[str]:
    if isinstance(item, Memory):
        return {tag.lower() for tag in item.tags}
    return set()


def _has_negation(tokens: set[str]) -> bool:
    return bool(tokens & NEGATIONS)


def _sentiment(tokens: set[str]) -> int:
    return len(tokens & POSITIVE_WORDS) - len(tokens & NEGATIVE_WORDS)


def _antonym_crosses(a: set[str], b: set[str]) -> bool:
    for word in a:
        if _ANTONYM_MAP.get(word, set()) & b:
            return True
    return False


def _opposition(a_text: str, b_text: str, a_tags: set[str], b_tags: set[str]) -> float:
    a_tokens, b_tokens = token_set(a_text), token_set(b_text)
    opposition = 0.0

    if _has_negation(a_tokens) != _has_negation(b_tokens):
        opposition = max(opposition, 0.7)
    if _antonym_crosses(a_tokens, b_tokens):
        opposition = max(opposition, 0.85)

    sentiment_product = _sentiment(a_tokens) * _sentiment(b_tokens)
    if sentiment_product < 0:
        opposition = max(opposition, 0.6)

    # Antonymous tags on otherwise-related memories are a strong conflict signal.
    if _antonym_crosses(a_tags, b_tags):
        opposition = max(opposition, 0.8)

    return opposition


def contradiction_pair(a: object, b: object) -> float:
    """Contradiction score between two memories/strings in ``[0, 1]``."""

    a_text, b_text = _content_of(a), _content_of(b)
    overlap = lexical_similarity(a_text, b_text)
    if overlap < 0.12:
        return 0.0  # unrelated: opposition would be coincidental

    opposition = _opposition(a_text, b_text, _tags_of(a), _tags_of(b))
    if opposition <= 0.0:
        return 0.0

    # Require both relatedness and opposition; relatedness modulates confidence.
    return max(0.0, min(1.0, opposition * (0.5 + 0.5 * overlap)))


def detect_contradiction(new_memory: object, existing_memories: Iterable[object]) -> float:
    """Highest pairwise contradiction between ``new_memory`` and any existing one."""

    scores = [contradiction_pair(new_memory, existing) for existing in existing_memories]
    return max(scores, default=0.0)


def semantic_contradiction_pair(a_text: str, b_text: str, llm: LLMClient) -> float | None:
    """LLM entailment-style contradiction score in ``[0, 1]`` (``None`` on failure)."""

    prompt = (
        "Do these two statements contradict each other? Consider negation, opposite "
        "claims, preference reversals, and incompatible facts. Respond with ONLY JSON: "
        '{"contradiction": 0.0-1.0, "reason": "..."} where 1.0 means a direct '
        "contradiction and 0.0 means no contradiction or unrelated.\n"
        f"A: {a_text}\nB: {b_text}\n"
    )
    obj = complete_json(llm, prompt)
    if isinstance(obj, dict) and "contradiction" in obj:
        return clamp01(obj["contradiction"], default=0.0)
    return None


class ContradictionChecker:
    """Pluggable contradiction scoring: deterministic heuristic, optionally
    refined by an LLM entailment check.

    To control cost, the LLM is only consulted for pairs that are at least
    loosely related (a cheap lexical gate); unrelated pairs short-circuit to 0.
    """

    def __init__(
        self,
        llm: LLMClient | Callable[..., Any] | None = None,
        *,
        candidate_gate: float = 0.06,
    ) -> None:
        self.llm = coerce_llm(llm)
        self.candidate_gate = candidate_gate

    def pair(self, a: object, b: object) -> float:
        heuristic = contradiction_pair(a, b)
        if self.llm is None:
            return heuristic
        a_text, b_text = _content_of(a), _content_of(b)
        # Only spend an LLM call when the texts are plausibly about the same thing.
        if lexical_similarity(a_text, b_text) < self.candidate_gate and heuristic == 0.0:
            return heuristic
        score = semantic_contradiction_pair(a_text, b_text, self.llm)
        return score if score is not None else heuristic

    def detect(self, new_memory: object, existing_memories: Iterable[object]) -> float:
        return max(
            (self.pair(new_memory, existing) for existing in existing_memories),
            default=0.0,
        )

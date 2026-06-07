"""Lightweight, LLM-free contradiction detection.

This is a *deterministic baseline*, not a truth oracle. A contradiction needs
two things at once: topical relatedness (the memories are about the same thing)
and semantic opposition. Opposition is detected via:

- negation mismatch (one side negated, the other not);
- antonyms, matched through a light stemmer so inflected forms
  (``enabled``/``disabled``, ``increased``/``decreased``) are caught;
- a sentiment flip;
- **numeric / temporal conflict** — near-identical wording but different numbers
  (``80ms`` vs ``800ms``, ``3pm`` vs ``4pm``).

It is intentionally conservative (high precision, partial recall). For full
semantic entailment, wire an ``llm`` into :class:`ContradictionChecker`. The
numeric heuristic can over-flag version-like or sequential data; treat results
as advisory.
"""

from __future__ import annotations

import re
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
    ("grow", "shrink"), ("grow", "fall"), ("grew", "fell"), ("gain", "lose"),
    ("high", "low"), ("good", "bad"), ("like", "dislike"), ("love", "hate"),
    ("enable", "disable"), ("allow", "deny"), ("allow", "forbid"),
    ("true", "false"), ("yes", "no"), ("success", "failure"), ("pass", "fail"),
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

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _stem(word: str) -> str:
    """A tiny, deterministic suffix stripper so inflected forms collide.

    Not a real lemmatizer — just enough to make ``enabled``≈``enable`` and
    ``increased``≈``increase`` match in the antonym map.
    """

    w = word
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def _antonyms() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for left, right in _ANTONYM_PAIRS:
        left_stem, right_stem = _stem(left), _stem(right)
        mapping.setdefault(left_stem, set()).add(right_stem)
        mapping.setdefault(right_stem, set()).add(left_stem)
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
    a_stems = {_stem(word) for word in a}
    b_stems = {_stem(word) for word in b}
    for word in a_stems:
        if _ANTONYM_MAP.get(word, set()) & b_stems:
            return True
    return False


def _numbers(text: str) -> set[float]:
    return {float(match) for match in _NUM_RE.findall(text)}


def _non_numeric_tokens(text: str) -> set[str]:
    return {token for token in token_set(text) if not token[0].isdigit()}


def _numeric_conflict(a_text: str, b_text: str) -> bool:
    """True when two near-identically worded statements carry different numbers."""

    numbers_a, numbers_b = _numbers(a_text), _numbers(b_text)
    if not numbers_a or not numbers_b or numbers_a == numbers_b:
        return False
    words_a, words_b = _non_numeric_tokens(a_text), _non_numeric_tokens(b_text)
    if not words_a or not words_b:
        return False
    union = len(words_a | words_b)
    overlap = len(words_a & words_b) / union if union else 0.0
    return overlap >= 0.5


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

    if _antonym_crosses(a_tags, b_tags):
        opposition = max(opposition, 0.8)

    return opposition


def contradiction_pair(a: object, b: object) -> float:
    """Contradiction score between two memories/strings in ``[0, 1]``."""

    a_text, b_text = _content_of(a), _content_of(b)
    overlap = lexical_similarity(a_text, b_text)
    numeric = _numeric_conflict(a_text, b_text)

    # Need relatedness; the numeric heuristic carries its own (stricter) gate.
    if overlap < 0.12 and not numeric:
        return 0.0

    opposition = _opposition(a_text, b_text, _tags_of(a), _tags_of(b))
    score = opposition * (0.5 + 0.5 * overlap) if opposition > 0.0 else 0.0
    if numeric:
        score = max(score, 0.6)
    return max(0.0, min(1.0, score))


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
        if lexical_similarity(a_text, b_text) < self.candidate_gate and heuristic == 0.0:
            return heuristic
        score = semantic_contradiction_pair(a_text, b_text, self.llm)
        return score if score is not None else heuristic

    def detect(self, new_memory: object, existing_memories: Iterable[object]) -> float:
        return max(
            (self.pair(new_memory, existing) for existing in existing_memories),
            default=0.0,
        )

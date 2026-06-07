"""Provider-agnostic LLM access.

Plasticity never imports a model SDK. Instead every LLM-powered feature accepts
an ``llm`` that is either an :class:`LLMClient` or a plain callable
``fn(prompt) -> str`` (the same callback you pass to ``PlasticAgent``). This
module normalises those into a small, safe interface and provides robust JSON
extraction so model output can drive structured decisions — always with a
deterministic fallback if the call or parse fails.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Anything with a ``complete(prompt) -> str`` method."""

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str: ...


class CallableLLM:
    """Adapts a plain callable into an :class:`LLMClient`.

    Tolerates a range of callable signatures: ``fn(prompt)``,
    ``fn(prompt, system=...)``, and ``fn(prompt, **kwargs)``.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
            ((prompt,), {"system": system, **kwargs} if system is not None else dict(kwargs)),
            ((prompt,), {}),
        )
        last_error: Exception | None = None
        for args, kw in attempts:
            try:
                return str(self._fn(*args, **kw))
            except TypeError as exc:  # signature mismatch: try a simpler call
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return str(self._fn(prompt))


def coerce_llm(llm: LLMClient | Callable[..., Any] | None) -> LLMClient | None:
    """Normalise ``llm`` to an :class:`LLMClient` (or ``None``)."""

    if llm is None:
        return None
    if hasattr(llm, "complete") and callable(llm.complete):
        return llm  # already an LLMClient
    if callable(llm):
        return CallableLLM(llm)
    raise TypeError(f"llm must be an LLMClient or callable, got {type(llm)!r}")


def safe_complete(
    llm: LLMClient | None, prompt: str, *, system: str | None = None
) -> str | None:
    """Call ``llm.complete`` and return ``None`` on any failure (never raises)."""

    if llm is None:
        return None
    try:
        result = llm.complete(prompt, system=system)
    except Exception:  # noqa: BLE001 - LLM calls are best-effort; fall back deterministically
        return None
    return result if isinstance(result, str) else (None if result is None else str(result))


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str | None) -> Any | None:
    """Extract the first JSON object/array from possibly-noisy model output."""

    if not text:
        return None
    text = text.strip()
    # Strip Markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1 :]
    for opener, closer in (("{", "}"), ("[", "]")):
        candidate = _find_balanced(text, opener, closer)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def complete_json(
    llm: LLMClient | None, prompt: str, *, system: str | None = None
) -> Any | None:
    """Complete and parse JSON in one step; ``None`` if anything fails."""

    return extract_json(safe_complete(llm, prompt, system=system))


def clamp01(value: Any, default: float = 0.5) -> float:
    """Best-effort coerce a model-provided number into ``[0, 1]``."""

    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default

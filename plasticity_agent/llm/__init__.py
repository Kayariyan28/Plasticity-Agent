"""Provider-agnostic LLM layer (callback-based; no model SDK dependency)."""

from __future__ import annotations

from plasticity_agent.llm.client import (
    CallableLLM,
    LLMClient,
    clamp01,
    coerce_llm,
    complete_json,
    extract_json,
    safe_complete,
)

__all__ = [
    "LLMClient",
    "CallableLLM",
    "coerce_llm",
    "safe_complete",
    "complete_json",
    "extract_json",
    "clamp01",
]

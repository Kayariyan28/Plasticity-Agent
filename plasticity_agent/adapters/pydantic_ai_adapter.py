"""Pydantic AI adapter (experimental).

Bridges a Pydantic AI ``Agent`` to a :class:`PlasticAgent`. ``pydantic-ai`` is
an optional dependency; a clear error is raised if it is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.adapters.generic import GenericAdapter
from plasticity_agent.core.runtime import RunResult


class PydanticAIAdapter:
    """Experimental bridge between Pydantic AI and Plasticity."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._generic = GenericAdapter(agent)

    @staticmethod
    def _require() -> Any:
        try:
            import pydantic_ai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "PydanticAIAdapter requires the optional 'pydantic-ai' package "
                "(experimental). Install it with `uv add pydantic-ai`."
            ) from exc
        return pydantic_ai

    def wrap(self, pydantic_agent: Any, method: str = "run_sync") -> Callable[..., RunResult]:
        """Wrap a Pydantic AI agent's ``run_sync`` (or another method)."""

        self._require()
        return self._generic.wrap_agent(pydantic_agent, method=method)

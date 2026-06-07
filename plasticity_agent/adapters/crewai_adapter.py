"""CrewAI adapter (experimental).

Bridges a CrewAI crew to a :class:`PlasticAgent`. ``crewai`` is an optional
dependency; a clear error is raised if it is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.adapters.generic import GenericAdapter
from plasticity_agent.core.runtime import RunResult


class CrewAIAdapter:
    """Experimental bridge between CrewAI and Plasticity."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._generic = GenericAdapter(agent)

    @staticmethod
    def _require() -> Any:
        try:
            import crewai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "CrewAIAdapter requires the optional 'crewai' package "
                "(experimental). Install it with `uv add crewai`."
            ) from exc
        return crewai

    def wrap(self, crew: Any, method: str = "kickoff") -> Callable[..., RunResult]:
        """Wrap ``crew.kickoff`` (or another method) through the agent runtime."""

        self._require()
        return self._generic.wrap_agent(crew, method=method)

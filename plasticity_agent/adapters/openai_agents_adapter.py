"""OpenAI Agents SDK adapter (experimental).

Bridges an OpenAI Agents SDK ``Agent``/``Runner`` to a :class:`PlasticAgent`.
``openai-agents`` is an optional dependency; a clear error is raised if missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.adapters.generic import GenericAdapter
from plasticity_agent.core.runtime import RunResult


class OpenAIAgentsAdapter:
    """Experimental bridge between the OpenAI Agents SDK and Plasticity."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._generic = GenericAdapter(agent)

    @staticmethod
    def _require() -> Any:
        try:
            import agents  # noqa: F401  (the `openai-agents` package imports as `agents`)
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "OpenAIAgentsAdapter requires the optional 'openai-agents' package "
                "(experimental). Install it with `uv add openai-agents`."
            ) from exc
        return agents

    def wrap(self, runner: Any, method: str = "run") -> Callable[..., RunResult]:
        """Wrap a runner's ``run`` (or another method) through the agent runtime."""

        self._require()
        return self._generic.wrap_agent(runner, method=method)

"""LangGraph adapter (experimental).

Bridges a compiled LangGraph graph to a :class:`PlasticAgent` so each graph
invocation is traced and remembered. ``langgraph`` is an optional dependency;
this adapter raises a clear, actionable error if it is not installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.adapters.generic import GenericAdapter
from plasticity_agent.core.runtime import RunResult


class LangGraphAdapter:
    """Experimental bridge between LangGraph and Plasticity."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._generic = GenericAdapter(agent)

    @staticmethod
    def _require() -> Any:
        try:
            import langgraph  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "LangGraphAdapter requires the optional 'langgraph' package "
                "(experimental). Install it with `uv add langgraph`."
            ) from exc
        return langgraph

    def wrap(self, graph: Any, method: str = "invoke") -> Callable[..., RunResult]:
        """Wrap ``graph.invoke`` (or another method) through the agent runtime."""

        self._require()
        return self._generic.wrap_agent(graph, method=method)

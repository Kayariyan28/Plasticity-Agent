"""The generic adapter.

Framework-agnostic glue: wrap any Python callable or any object exposing a
``run``-like method so it gains Plasticity tracing, an episodic memory of each
invocation, and advisory healing on failure. The framework-specific adapters
build on this.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.core.runtime import RunResult


class GenericAdapter:
    """Wrap callables/agents so their runs flow through a :class:`PlasticAgent`."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def wrap_callable(
        self, fn: Callable[..., Any], *, reflect_on_failure: bool = False
    ) -> Callable[..., RunResult]:
        """Return a wrapper that runs ``fn`` through the agent runtime."""

        def wrapped(*args: Any, **kwargs: Any) -> RunResult:
            result = self.agent.run(lambda: fn(*args, **kwargs))
            if result.status == "failed" and reflect_on_failure:
                self.agent.reflect(task=result.task, error=result.error, reward=-1.0)
            return result

        wrapped.__name__ = getattr(fn, "__name__", "wrapped_callable")
        return wrapped

    def wrap_agent(self, external_agent: Any, method: str = "run") -> Callable[..., RunResult]:
        """Wrap an external agent object's ``method`` (default ``run``)."""

        target = getattr(external_agent, method, None)
        if not callable(target):
            raise AttributeError(
                f"external agent has no callable '{method}' method to wrap"
            )
        return self.wrap_callable(target)

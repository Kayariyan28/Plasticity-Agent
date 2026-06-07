"""Core runtime: agent, execution runtime, config, events, and tracing."""

from __future__ import annotations

from plasticity_agent.core.agent import PlasticAgent
from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.core.events import TraceEvent
from plasticity_agent.core.runtime import RunResult, Runtime
from plasticity_agent.core.trace import Tracer

__all__ = [
    "PlasticAgent",
    "PlasticityConfig",
    "TraceEvent",
    "RunResult",
    "Runtime",
    "Tracer",
]

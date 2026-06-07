"""Framework adapters (the framework-specific ones are experimental)."""

from __future__ import annotations

from plasticity_agent.adapters.crewai_adapter import CrewAIAdapter
from plasticity_agent.adapters.generic import GenericAdapter
from plasticity_agent.adapters.langgraph_adapter import LangGraphAdapter
from plasticity_agent.adapters.openai_agents_adapter import OpenAIAgentsAdapter
from plasticity_agent.adapters.pydantic_ai_adapter import PydanticAIAdapter

__all__ = [
    "GenericAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
    "PydanticAIAdapter",
]

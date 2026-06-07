"""Tests for the generic and framework adapters.

Framework adapters are exercised against faithful fakes (and a fake module in
``sys.modules``) so the wrapping, tracing, and import-guard behaviour is covered
without installing LangGraph/CrewAI/etc.
"""

from __future__ import annotations

import sys
import types

import pytest

from plasticity_agent import PlasticAgent
from plasticity_agent.adapters.crewai_adapter import CrewAIAdapter
from plasticity_agent.adapters.generic import GenericAdapter
from plasticity_agent.adapters.langgraph_adapter import LangGraphAdapter
from plasticity_agent.adapters.openai_agents_adapter import OpenAIAgentsAdapter
from plasticity_agent.adapters.pydantic_ai_adapter import PydanticAIAdapter

# (AdapterClass, fake import-name, wrapped method name)
ADAPTERS = [
    (LangGraphAdapter, "langgraph", "invoke"),
    (CrewAIAdapter, "crewai", "kickoff"),
    (OpenAIAgentsAdapter, "agents", "run"),
    (PydanticAIAdapter, "pydantic_ai", "run_sync"),
]


def test_generic_wrap_callable(agent: PlasticAgent) -> None:
    wrapped = GenericAdapter(agent).wrap_callable(lambda x: x * 2)
    result = wrapped(21)
    assert result.status == "completed"
    assert result.output == 42


def test_generic_wrap_callable_reflects_on_failure(agent: PlasticAgent) -> None:
    def boom() -> None:
        raise ValueError("nope")

    wrapped = GenericAdapter(agent).wrap_callable(boom, reflect_on_failure=True)
    result = wrapped()
    assert result.status == "failed"
    assert agent.memory.list_memories(memory_type="reflective")


def test_generic_wrap_agent(agent: PlasticAgent) -> None:
    class External:
        def run(self, query: str) -> str:
            return f"ran {query}"

    wrapped = GenericAdapter(agent).wrap_agent(External())
    assert wrapped("hello").output == "ran hello"


def test_generic_wrap_agent_missing_method(agent: PlasticAgent) -> None:
    with pytest.raises(AttributeError):
        GenericAdapter(agent).wrap_agent(object(), method="does_not_exist")


@pytest.mark.parametrize(("adapter_cls", "module_name", "method"), ADAPTERS)
def test_framework_adapter_with_fake_dependency(
    monkeypatch, agent: PlasticAgent, adapter_cls, module_name, method
) -> None:
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    class External:
        pass

    setattr(External, method, lambda self, payload: f"{method}:{payload}")
    wrapped = adapter_cls(agent).wrap(External())
    result = wrapped("payload")
    assert result.status == "completed"
    assert result.output == f"{method}:payload"


@pytest.mark.parametrize(("adapter_cls", "module_name", "method"), ADAPTERS)
def test_framework_adapter_missing_dependency_raises(
    monkeypatch, agent: PlasticAgent, adapter_cls, module_name, method
) -> None:
    monkeypatch.setitem(sys.modules, module_name, None)  # force ImportError on import

    class External:
        pass

    setattr(External, method, lambda self, payload: payload)
    with pytest.raises(ImportError):
        adapter_cls(agent).wrap(External())

"""Shared pytest fixtures: isolated, temp-dir-backed memory and agents."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from plasticity_agent import PlasticAgent
from plasticity_agent.memory.memory_os import MemoryOS


@pytest.fixture
def memory_dir(tmp_path) -> str:
    return str(tmp_path / "memory")


@pytest.fixture
def memory(memory_dir: str) -> Iterator[MemoryOS]:
    store = MemoryOS(memory_dir=memory_dir)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def agent(memory_dir: str) -> Iterator[PlasticAgent]:
    instance = PlasticAgent(name="test-agent", memory=memory_dir)
    try:
        yield instance
    finally:
        instance.close()

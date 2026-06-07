"""FastAPI server exposing the Plasticity runtime over HTTP.

FastAPI/uvicorn are imported lazily inside :func:`build_app`/:func:`serve` so
importing the package never requires the web stack. Each request uses its own
short-lived :class:`MemoryOS` (own SQLite connection), which keeps the server
safe under uvicorn's threaded sync execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from plasticity_agent.healing.repair import heal
from plasticity_agent.memory.memory_os import MemoryOS
from plasticity_agent.memory.schemas import MemoryType
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.thermodynamics.energy_report import build_energy_report


class RecordRequest(BaseModel):
    content: str
    memory_type: MemoryType = "episodic"
    tags: list[str] = []
    reward: float = 0.0
    confidence: float = 0.7


class RecallRequest(BaseModel):
    query: str
    limit: int = 5


class HealRequest(BaseModel):
    error: str


class MarketRequest(BaseModel):
    task: str
    context: dict[str, Any] = {}


@contextmanager
def _memory_os(memory_dir: str):
    memory = MemoryOS(memory_dir=memory_dir)
    try:
        yield memory
    finally:
        memory.close()


def build_app(memory_dir: str = "./memory"):
    """Construct and return the FastAPI application."""

    from fastapi import FastAPI

    from plasticity_agent import __version__

    app = FastAPI(title="Plasticity Agent Runtime", version=__version__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "memory_dir": memory_dir}

    @app.get("/memories")
    def list_memories(limit: int = 50) -> list[dict[str, Any]]:
        with _memory_os(memory_dir) as memory:
            return [m.model_dump(mode="json") for m in memory.list_memories(limit=limit)]

    @app.post("/memories")
    def create_memory(request: RecordRequest) -> dict[str, Any]:
        with _memory_os(memory_dir) as memory:
            record = memory.record(
                request.content,
                request.memory_type,
                tags=request.tags,
                reward=request.reward,
                confidence=request.confidence,
            )
            return record.model_dump(mode="json")

    @app.post("/recall")
    def recall(request: RecallRequest) -> list[dict[str, Any]]:
        with _memory_os(memory_dir) as memory:
            return [
                r.model_dump(mode="json")
                for r in memory.recall(request.query, limit=request.limit)
            ]

    @app.post("/sleep")
    def sleep() -> dict[str, Any]:
        with _memory_os(memory_dir) as memory:
            return memory.sleep().model_dump(mode="json")

    @app.post("/heal")
    def heal_error(request: HealRequest) -> dict[str, Any]:
        return heal(request.error).model_dump(mode="json")

    @app.post("/market")
    def market(request: MarketRequest) -> dict[str, Any]:
        result = ReasoningMarket().deliberate(request.task, request.context)
        return result.model_dump(mode="json")

    @app.get("/report")
    def report() -> dict[str, Any]:
        with _memory_os(memory_dir) as memory:
            energy = build_energy_report(memory.list_memories(), memory.load_traces())
            return {
                "memories": memory.count(),
                "skills": memory.skills.count(),
                "energy": energy.model_dump(mode="json"),
            }

    @app.get("/skills")
    def skills() -> list[dict[str, Any]]:
        with _memory_os(memory_dir) as memory:
            return [s.model_dump(mode="json") for s in memory.skills.list_skills()]

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, memory_dir: str = "./memory") -> None:
    """Run the API server with uvicorn (blocking)."""

    import uvicorn

    uvicorn.run(build_app(memory_dir), host=host, port=port)

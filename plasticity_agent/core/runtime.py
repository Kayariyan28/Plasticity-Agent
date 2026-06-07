"""The execution runtime.

Wraps a single run with start/complete/fail tracing and resolves the three
v0.1.0 execution modes: a Python callable, a string task with an LLM callback,
or a string task with no callback (a structured advisory response). The runtime
is model-agnostic — it never imports an LLM SDK.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from plasticity_agent.core.events import RUN_COMPLETED, RUN_FAILED, RUN_STARTED
from plasticity_agent.core.trace import Tracer

RunStatus = Literal["completed", "failed", "advisory"]


def _preview(value: object, limit: int = 500) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class RunResult(BaseModel):
    """The structured outcome of a single run."""

    run_id: str
    task: str
    status: RunStatus
    output: Any | None = None
    error: str | None = None
    advisory: bool = False
    metadata: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

    def __str__(self) -> str:
        if self.status == "failed":
            return f"<RunResult failed: {self.error}>"
        return f"<RunResult {self.status}: {_preview(self.output, 120)}>"


class Runtime:
    """Executes tasks and records run lifecycle traces."""

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer

    def execute(
        self,
        task_or_callable: str | Callable[..., Any],
        *,
        llm_callback: Callable[..., Any] | None = None,
        advisory_fn: Callable[[str], Any] | None = None,
        **kwargs: Any,
    ) -> RunResult:
        run_id = self.tracer.run_id
        is_callable = callable(task_or_callable)
        task = (
            getattr(task_or_callable, "__name__", "callable")
            if is_callable
            else str(task_or_callable)
        )
        self.tracer.emit(
            RUN_STARTED, {"task": task, "kind": "callable" if is_callable else "string"}
        )

        try:
            if callable(task_or_callable):
                output = task_or_callable(**kwargs) if kwargs else task_or_callable()
                result = RunResult(run_id=run_id, task=task, status="completed", output=output)
            elif llm_callback is not None:
                output = llm_callback(task, **kwargs)
                result = RunResult(run_id=run_id, task=task, status="completed", output=output)
            else:
                output = advisory_fn(task) if advisory_fn else _default_advisory(task)
                result = RunResult(
                    run_id=run_id, task=task, status="advisory", output=output, advisory=True
                )
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed RunResult, not swallowed
            self.tracer.emit(RUN_FAILED, {"task": task, "error": f"{type(exc).__name__}: {exc}"})
            return RunResult(
                run_id=run_id,
                task=task,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

        self.tracer.emit(
            RUN_COMPLETED,
            {"task": task, "status": result.status, "output": _preview(result.output)},
        )
        return result


def _default_advisory(task: str) -> dict[str, Any]:
    return {
        "advisory": True,
        "task": task,
        "message": (
            "Plasticity runtime is active, but no LLM callback is configured. "
            "Returning an advisory plan instead of a model completion. "
            "Pass llm_callback=... to PlasticAgent to enable generation."
        ),
    }

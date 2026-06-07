"""PlasticAgent — the full-stack agentic runtime façade.

One object wires together neuroplastic memory, reflection, advisory self-healing,
the reasoning market, the skill library, and thermodynamic reporting. It is
framework-agnostic: drive it with a Python callable, an LLM callback, or neither
(advisory mode). Every run is traced to JSONL.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.core.events import (
    ENERGY_REPORT_CREATED,
    HEALING_DIAGNOSED,
    REASONING_AUCTION,
    REFLECTION_CREATED,
)
from plasticity_agent.core.runtime import RunResult, Runtime
from plasticity_agent.core.trace import Tracer
from plasticity_agent.healing.diagnosis import diagnose
from plasticity_agent.healing.repair import RepairPlan, plan_repair
from plasticity_agent.healing.sandbox import RepairConsent, Sandbox, SandboxResult
from plasticity_agent.learning.skill_library import Skill
from plasticity_agent.memory.consolidation import SleepReport
from plasticity_agent.memory.embeddings import EmbeddingBackend
from plasticity_agent.memory.memory_os import MemoryOS
from plasticity_agent.memory.schemas import Memory, MemorySearchResult, MemoryType
from plasticity_agent.metrics.tracker import ImprovementReport, ImprovementTracker, MetricSnapshot
from plasticity_agent.reasoning.auction import AuctionResult
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.reflection.lessons import Lesson, ReflectionInput
from plasticity_agent.reflection.reflector import Reflector
from plasticity_agent.reflection.self_refine import SelfRefine, SelfRefineResult
from plasticity_agent.thermodynamics.energy_report import EnergyReport, build_energy_report


class PlasticAgent:
    """A local-first agent with memory, reflection, healing, and reasoning."""

    def __init__(
        self,
        name: str,
        model: str | None = None,
        memory: str = "./memory",
        *,
        self_heal: bool = True,
        reasoning_market: bool = True,
        sleep_cycle: bool = True,
        llm_callback: Callable[..., Any] | None = None,
        embeddings: EmbeddingBackend | Callable[..., Any] | str | None = None,
        otel: bool | object = False,
        metrics: bool = True,
        config: PlasticityConfig | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.self_heal = self_heal
        self.sleep_cycle = sleep_cycle
        self.llm_callback = llm_callback

        self.config = config or PlasticityConfig.from_memory_dir(memory)
        exporter = self._build_exporter(otel, name)
        self.tracer = Tracer(self.config.traces_dir, exporter=exporter)
        self.memory = MemoryOS(
            config=self.config, tracer=self.tracer, llm=llm_callback, embedder=embeddings
        )
        self.skills = self.memory.skills
        self.reflector = Reflector(llm=llm_callback)
        self.refiner = SelfRefine(llm_callback=self._refine_callback())
        self.market: ReasoningMarket | None = (
            ReasoningMarket(llm=llm_callback) if reasoning_market else None
        )
        self.runtime = Runtime(self.tracer)
        self.sandbox = Sandbox()
        self.metrics: ImprovementTracker | None = (
            ImprovementTracker(self.config.metrics_path) if metrics else None
        )
        self.last_run: RunResult | None = None

    @staticmethod
    def _build_exporter(otel: bool | object, name: str) -> Any | None:
        if not otel:
            return None
        if hasattr(otel, "export"):
            return otel
        from plasticity_agent.observability.otel import OTelExporter

        return OTelExporter(service_name=f"plasticity-agent:{name}")

    def _refine_callback(self) -> Callable[[str, str], str] | None:
        if self.llm_callback is None:
            return None
        callback = self.llm_callback

        def refine(prompt: str, _rubric: str) -> str:
            return str(callback(prompt))

        return refine

    # -- execution ------------------------------------------------------------

    def run(self, task_or_callable: str | Callable[..., Any], **kwargs: Any) -> RunResult:
        """Execute a task (callable / LLM callback / advisory) with full tracing."""

        result = self.runtime.execute(
            task_or_callable,
            llm_callback=self.llm_callback,
            advisory_fn=self._advisory_response,
            **kwargs,
        )

        if result.status == "failed" and self.self_heal and result.error:
            result.metadata["repair_plan"] = self.heal(result.error).model_dump()

        # Keep a lightweight episodic trace of every run in memory.
        self.memory.record(
            f"Run '{result.task}' -> {result.status}",
            "episodic",
            tags=["run", result.status],
            confidence=0.6,
            source_trace=result.run_id,
            check_contradictions=False,
        )
        self.last_run = result
        return result

    def _advisory_response(self, task: str) -> dict[str, Any]:
        recalled = self.memory.search(task, limit=3)
        response: dict[str, Any] = {
            "advisory": True,
            "task": task,
            "message": (
                "Plasticity runtime is active, but no LLM callback is configured. "
                "Returning an advisory plan instead of a model completion."
            ),
            "recalled_memories": [
                {"content": r.memory.content, "score": round(r.score, 3)} for r in recalled
            ],
            "next_steps": [
                "Pass llm_callback=... to PlasticAgent to enable generation.",
                "Or call agent with a Python callable to execute deterministic work.",
            ],
        }
        if self.market is not None:
            auction = self.market.deliberate(task, {"mode": "advisory"})
            response["suggested_action"] = auction.winner.action
            response["selection_score"] = round(auction.selection_score, 3)
        return response

    # -- memory ---------------------------------------------------------------

    def remember(
        self,
        content: str,
        memory_type: MemoryType = "episodic",
        *,
        tags: list[str] | None = None,
        reward: float = 0.0,
        confidence: float = 0.7,
        **kwargs: Any,
    ) -> Memory:
        return self.memory.record(
            content,
            memory_type,
            tags=tags,
            reward=reward,
            confidence=confidence,
            **kwargs,
        )

    def recall(self, query: str, *, limit: int = 5) -> list[MemorySearchResult]:
        return self.memory.recall(query, limit=limit)

    # -- reflection & refinement ---------------------------------------------

    def reflect(
        self,
        task: str | None = None,
        output: str | None = None,
        error: object | None = None,
        reward: float = 0.0,
        evaluator_feedback: str | None = None,
    ) -> Lesson:
        """Create a lesson from a run and store it as a reflective memory."""

        if task is None and self.last_run is not None:
            task = self.last_run.task
            output = output if output is not None else _as_text(self.last_run.output)
            error = error if error is not None else self.last_run.error
        task = task or "(unspecified task)"
        error_text = _as_error_text(error)

        lesson = self.reflector.create_lesson(
            ReflectionInput(
                task=task,
                output=output,
                error=error_text,
                reward=reward,
                evaluator_feedback=evaluator_feedback,
            )
        )
        self.memory.record(
            lesson.content,
            "reflective",
            tags=lesson.tags,
            reward=lesson.reward,
            confidence=lesson.confidence,
            check_contradictions=False,
        )
        self.tracer.emit(
            REFLECTION_CREATED,
            {"lesson_type": lesson.lesson_type, "confidence": lesson.confidence},
        )
        return lesson

    def refine(
        self, output: str, rubric: str = "accuracy, safety, completeness"
    ) -> SelfRefineResult:
        return self.refiner.refine(output, rubric)

    # -- healing --------------------------------------------------------------

    def heal(self, error: object) -> RepairPlan:
        """Return an advisory repair plan for ``error`` (never auto-applied)."""

        diagnosis = diagnose(error)
        plan = plan_repair(diagnosis)
        self.tracer.emit(
            HEALING_DIAGNOSED,
            {
                "failure_type": diagnosis.failure_type,
                "confidence": diagnosis.confidence,
                "risk_level": plan.risk_level,
                "advisory_only": plan.advisory_only,
            },
        )
        return plan

    def apply_repair(
        self, error_or_plan: object | RepairPlan, consent: RepairConsent | None = None
    ) -> SandboxResult:
        """Attempt a sandboxed repair. Default consent applies nothing (dry/off).

        Pass a :class:`RepairConsent` with ``allow_apply=True`` (and the relevant
        capability, e.g. ``allow_install=True``, ``dry_run=False``) to actually
        execute a safe, whitelisted repair.
        """

        plan = error_or_plan if isinstance(error_or_plan, RepairPlan) else self.heal(error_or_plan)
        result = self.sandbox.apply(plan, consent)
        self.tracer.emit(
            "healing_applied",
            {
                "failure_type": plan.diagnosis.failure_type,
                "applied": result.applied,
                "dry_run": result.dry_run,
            },
        )
        return result

    # -- reasoning ------------------------------------------------------------

    def deliberate(
        self, task: str, context: dict[str, Any] | None = None
    ) -> AuctionResult:
        """Run the reasoning market on ``task`` (requires reasoning_market=True)."""

        if self.market is None:
            raise RuntimeError("reasoning_market is disabled for this agent")
        auction = self.market.deliberate(task, context or {})
        self.tracer.emit(
            REASONING_AUCTION,
            {"task": task, "winner": auction.winner.critic_name, "score": auction.selection_score},
        )
        return auction

    # -- consolidation & reporting -------------------------------------------

    def sleep(self) -> SleepReport:
        """Run a sleep/consolidation cycle over memory and traces."""

        return self.memory.sleep()

    def energy_report(self) -> EnergyReport:
        """Produce a thermodynamic-style reliability report."""

        report = build_energy_report(self.memory.list_memories(), self.memory.load_traces())
        self.tracer.emit(ENERGY_REPORT_CREATED, report.model_dump(mode="json"))
        return report

    def report(self) -> dict[str, Any]:
        """A high-level status snapshot for dashboards/CLI."""

        quality = self.memory.evaluate_all()
        recommendation_counts: dict[str, int] = {}
        for item in quality:
            recommendation_counts[item.recommendation] = (
                recommendation_counts.get(item.recommendation, 0) + 1
            )
        return {
            "name": self.name,
            "model": self.model,
            "memories": self.memory.count(),
            "skills": self.skills.count(),
            "recommendations": recommendation_counts,
            "energy": self.energy_report().model_dump(),
        }

    def export(self, path: str | Path | None = None) -> Path:
        return self.memory.export_jsonl(path)

    def list_skills(self) -> list[Skill]:
        return self.skills.list_skills()

    # -- cross-run improvement metrics ---------------------------------------

    def checkpoint(self, label: str = "checkpoint") -> MetricSnapshot:
        """Record a metric snapshot so improvement can be measured over time."""

        energy = build_energy_report(self.memory.list_memories(), self.memory.load_traces())
        quality = self.memory.evaluate_all()
        avg_utility = mean(q.utility_score for q in quality) if quality else 0.0
        # Grounded utility weights each memory by how much it has actually been
        # used (recalled), so storing unused but flattering memories doesn't help.
        grounded = (
            mean(q.utility_score * min(q.usage_count, 5) / 5.0 for q in quality)
            if quality
            else 0.0
        )
        snapshot = MetricSnapshot(
            label=label,
            memories=self.memory.count(),
            skills=self.skills.count(),
            plasticity_score=energy.plasticity_score,
            avg_utility=round(avg_utility, 4),
            grounded_utility=round(grounded, 4),
            contradiction_pressure=energy.contradiction_pressure,
            memory_entropy=energy.memory_entropy,
        )
        if self.metrics is not None:
            self.metrics.record(snapshot)
        return snapshot

    def improvement(self) -> ImprovementReport:
        """Compare recorded checkpoints — did the agent actually get better?"""

        if self.metrics is None:
            raise RuntimeError("metrics tracking is disabled for this agent")
        return self.metrics.report()

    def close(self) -> None:
        self.memory.close()
        if self.metrics is not None:
            self.metrics.close()


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _as_error_text(error: object | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)

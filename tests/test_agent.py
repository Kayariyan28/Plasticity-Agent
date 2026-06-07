"""Tests for the PlasticAgent integration layer."""

from __future__ import annotations

from plasticity_agent import PlasticAgent
from plasticity_agent.memory.memory_os import MemoryOS


def test_advisory_run_without_callback(agent: PlasticAgent) -> None:
    result = agent.run("Summarize this paper and critique its methodology")
    assert result.status == "advisory"
    assert result.advisory is True
    assert "suggested_action" in result.output  # reasoning market is enabled by default


def test_callable_run_executes(agent: PlasticAgent) -> None:
    result = agent.run(lambda: 6 * 7)
    assert result.status == "completed"
    assert result.output == 42


def test_llm_callback_run(memory_dir: str) -> None:
    instance = PlasticAgent(
        name="cb",
        memory=memory_dir,
        llm_callback=lambda task, **_kwargs: f"answer to {task}",
    )
    try:
        result = instance.run("a question")
        assert result.status == "completed"
        assert "answer to" in result.output
    finally:
        instance.close()


def test_remember_and_recall(agent: PlasticAgent) -> None:
    agent.remember("Cache warmup reduced p95 latency", "semantic", tags=["important"])
    assert agent.recall("latency cache")


def test_reflect_stores_reflective_memory(agent: PlasticAgent) -> None:
    lesson = agent.reflect(task="extract claims", error="boom happened", reward=-0.7)
    assert lesson.lesson_type == "failure"
    reflective = agent.memory.list_memories(memory_type="reflective")
    assert any(m.content == lesson.content for m in reflective)


def test_heal_returns_plan(agent: PlasticAgent) -> None:
    plan = agent.heal(ModuleNotFoundError("No module named 'foo'"))
    assert plan.diagnosis.failure_type == "missing_dependency"


def test_apply_repair_default_is_safe(agent: PlasticAgent) -> None:
    # Without explicit consent, apply_repair must never execute anything.
    result = agent.apply_repair(ModuleNotFoundError("No module named 'foo'"))
    assert result.applied is False
    assert result.advisory_only is True


def test_sleep_and_energy_report(agent: PlasticAgent) -> None:
    agent.remember("a durable fact", "semantic")
    sleep_report = agent.sleep()
    assert 0.0 <= sleep_report.plasticity_score <= 100.0
    energy = agent.energy_report()
    assert 0.0 <= energy.plasticity_score <= 100.0


def test_failed_run_attaches_repair_plan(agent: PlasticAgent) -> None:
    def boom() -> None:
        raise ValueError("bad value")

    result = agent.run(boom)
    assert result.status == "failed"
    assert "repair_plan" in result.metadata


def test_report_snapshot(agent: PlasticAgent) -> None:
    agent.remember("something")
    report = agent.report()
    assert report["name"] == "test-agent"
    assert report["memories"] >= 1
    assert "energy" in report


def test_agent_skills_shared_with_memory(agent: PlasticAgent) -> None:
    assert isinstance(agent.memory, MemoryOS)
    assert agent.skills is agent.memory.skills

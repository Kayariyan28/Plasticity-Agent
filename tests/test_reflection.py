"""Tests for Reflexion-style lesson extraction."""

from __future__ import annotations

from plasticity_agent.reflection.lessons import ReflectionInput
from plasticity_agent.reflection.reflector import Reflector


def test_failure_lesson_from_error_and_negative_reward() -> None:
    lesson = Reflector().create_lesson(
        ReflectionInput(task="do thing", error="something broke badly", reward=-0.8)
    )
    assert lesson.lesson_type == "failure"


def test_success_lesson_from_positive_reward() -> None:
    lesson = Reflector().create_lesson(ReflectionInput(task="do thing", reward=0.9))
    assert lesson.lesson_type == "success"


def test_tool_use_lesson_from_schema_error() -> None:
    lesson = Reflector().create_lesson(
        ReflectionInput(
            task="call tool",
            error="TypeError: missing 1 required positional argument: 'schema'",
            reward=-0.5,
        )
    )
    assert lesson.lesson_type == "tool_use"


def test_reasoning_lesson_from_accuracy_feedback() -> None:
    lesson = Reflector().create_lesson(
        ReflectionInput(task="answer", evaluator_feedback="the accuracy was poor; facts were wrong")
    )
    assert lesson.lesson_type == "reasoning"


def test_preference_lesson_from_feedback() -> None:
    lesson = Reflector().create_lesson(
        ReflectionInput(task="answer", evaluator_feedback="I prefer concise answers")
    )
    assert lesson.lesson_type == "preference"


def test_lesson_has_valid_fields() -> None:
    lesson = Reflector().reflect("a task", reward=0.5)
    assert 0.0 <= lesson.confidence <= 1.0
    assert lesson.lesson_type in {
        "success",
        "failure",
        "risk",
        "preference",
        "tool_use",
        "reasoning",
    }
    assert lesson.tags
    assert lesson.content

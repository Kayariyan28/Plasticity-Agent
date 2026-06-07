"""Schemas for reflection: the input an agent reflects on, and the lesson out.

Reflexion-style: an agent that stores explicit lessons from feedback improves
across runs. Lessons are later persisted as ``reflective`` memories.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

LessonType = Literal["success", "failure", "risk", "preference", "tool_use", "reasoning"]


class ReflectionInput(BaseModel):
    """What happened on a run, framed for reflection."""

    task: str
    output: str | None = None
    error: str | None = None
    reward: float = 0.0
    evaluator_feedback: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Lesson(BaseModel):
    """A single, reusable takeaway extracted from a run."""

    content: str
    lesson_type: LessonType
    confidence: float
    reward: float
    tags: list[str] = Field(default_factory=list)

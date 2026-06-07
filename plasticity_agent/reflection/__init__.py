"""Reflection: lesson extraction (Reflexion) and self-refine loops."""

from __future__ import annotations

from plasticity_agent.reflection.lessons import Lesson, LessonType, ReflectionInput
from plasticity_agent.reflection.reflector import Reflector
from plasticity_agent.reflection.self_refine import SelfRefine, SelfRefineResult

__all__ = [
    "Lesson",
    "LessonType",
    "ReflectionInput",
    "Reflector",
    "SelfRefine",
    "SelfRefineResult",
]

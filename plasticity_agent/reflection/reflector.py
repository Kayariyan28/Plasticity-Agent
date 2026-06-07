"""The Reflector — deterministic lesson extraction from run outcomes.

This is the v0.1.0 baseline: it classifies an outcome using transparent
keyword/heuristic rules. An LLM-authored reflection is planned for v0.2.0; the
:meth:`Reflector.create_lesson` contract stays the same.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.llm.client import LLMClient, clamp01, coerce_llm, complete_json
from plasticity_agent.reflection.lessons import Lesson, LessonType, ReflectionInput

_LESSON_TYPES = {"success", "failure", "risk", "preference", "tool_use", "reasoning"}

_TOOL_MARKERS = (
    "argument",
    "schema",
    "missing required",
    "typeerror",
    "unexpected keyword",
    "positional",
    "validationerror",
    "keyerror",
    "attributeerror",
    "signature",
)

_PREFERENCE_MARKERS = (
    "prefer",
    "preference",
    "rather",
    "instead",
    "i like",
    "i want",
    "please use",
    "more concise",
    "tone",
)

_REASONING_MARKERS = (
    "accuracy",
    "inaccurate",
    "incorrect",
    "wrong",
    "reasoning",
    "logic",
    "hallucinat",
    "evidence",
    "fact",
    "citation",
    "unsupported",
)


def _shorten(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _contains(haystack: str | None, markers: tuple[str, ...]) -> bool:
    if not haystack:
        return False
    lowered = haystack.lower()
    return any(marker in lowered for marker in markers)


class Reflector:
    """Turns a :class:`ReflectionInput` into a typed :class:`Lesson`.

    With an ``llm`` configured, reflection is model-authored (richer content and
    classification); without one it uses the deterministic rule baseline. The
    LLM path always falls back to rules if the call or parse fails.
    """

    def __init__(self, llm: LLMClient | Callable[..., Any] | None = None) -> None:
        self.llm = coerce_llm(llm)

    def create_lesson(self, data: ReflectionInput) -> Lesson:
        if self.llm is not None:
            lesson = self._llm_lesson(data)
            if lesson is not None:
                return lesson
        lesson_type = self._classify(data)
        content = self._compose(lesson_type, data)
        confidence = self._confidence(lesson_type, data)
        tags = self._tags(lesson_type, data)
        return Lesson(
            content=content,
            lesson_type=lesson_type,
            confidence=confidence,
            reward=data.reward,
            tags=tags,
        )

    def _llm_lesson(self, data: ReflectionInput) -> Lesson | None:
        prompt = (
            "You are an agent's reflection module (Reflexion). Analyse this run and "
            "extract one reusable lesson. Respond with ONLY a JSON object:\n"
            '{"lesson_type": one of '
            '["success","failure","risk","preference","tool_use","reasoning"], '
            '"content": "a concise, actionable lesson", '
            '"confidence": 0.0-1.0, "tags": ["..."]}\n\n'
            f"task: {data.task}\n"
            f"output: {_shorten(data.output, 400)}\n"
            f"error: {_shorten(data.error, 300)}\n"
            f"reward: {data.reward}\n"
            f"evaluator_feedback: {_shorten(data.evaluator_feedback, 300)}\n"
        )
        obj = complete_json(self.llm, prompt)
        if not isinstance(obj, dict):
            return None
        lesson_type = str(obj.get("lesson_type", "")).lower().strip()
        if lesson_type not in _LESSON_TYPES:
            return None
        content = str(obj.get("content") or "").strip()
        if not content:
            return None
        raw_tags = obj.get("tags") or []
        if not isinstance(raw_tags, list):
            raw_tags = [str(raw_tags)]
        tags = sorted({str(tag).lower() for tag in raw_tags} | {lesson_type, "reflection"})
        return Lesson(
            content=content,
            lesson_type=lesson_type,  # type: ignore[arg-type]
            confidence=clamp01(obj.get("confidence", 0.7)),
            reward=data.reward,
            tags=tags,
        )

    def reflect(
        self,
        task: str,
        *,
        output: str | None = None,
        error: str | None = None,
        reward: float = 0.0,
        evaluator_feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Lesson:
        """Convenience wrapper that builds the :class:`ReflectionInput` for you."""

        return self.create_lesson(
            ReflectionInput(
                task=task,
                output=output,
                error=error,
                reward=reward,
                evaluator_feedback=evaluator_feedback,
                metadata=metadata or {},
            )
        )

    # -- classification rules -------------------------------------------------

    def _classify(self, data: ReflectionInput) -> LessonType:
        if _contains(data.error, _TOOL_MARKERS):
            return "tool_use"
        if _contains(data.evaluator_feedback, _PREFERENCE_MARKERS):
            return "preference"
        if _contains(data.evaluator_feedback, _REASONING_MARKERS):
            return "reasoning"
        if data.error and data.reward < 0:
            return "failure"
        if data.reward > 0:
            return "success"
        if data.error:
            return "failure"
        if data.reward < 0:
            return "failure"
        return "risk"

    def _compose(self, lesson_type: LessonType, data: ReflectionInput) -> str:
        task = _shorten(data.task, 120)
        error = _shorten(data.error)
        feedback = _shorten(data.evaluator_feedback)
        if lesson_type == "tool_use":
            return (
                f"Tool/schema error while doing '{task}': {error}. "
                "Validate argument names and types against the tool schema before calling."
            )
        if lesson_type == "preference":
            return (
                f"User preference observed on '{task}': {feedback}. "
                "Honor this preference on similar tasks."
            )
        if lesson_type == "reasoning":
            return (
                f"Feedback on '{task}' flagged accuracy/reasoning: {feedback}. "
                "Add supporting evidence and re-check claims before finalizing."
            )
        if lesson_type == "success":
            return (
                f"Approach for '{task}' worked (reward={data.reward:.2f}). "
                "Reuse this pattern on similar tasks."
            )
        if lesson_type == "failure":
            tail = f": {error}" if error else "."
            return (
                f"Attempt at '{task}' failed (reward={data.reward:.2f}){tail} "
                "Add guards and verify inputs next time."
            )
        return (
            f"Uncertain outcome on '{task}' (reward={data.reward:.2f}). "
            "Monitor for risk and gather more signal."
        )

    def _confidence(self, lesson_type: LessonType, data: ReflectionInput) -> float:
        base = 0.5 + 0.4 * min(abs(data.reward), 1.0)
        if lesson_type in {"tool_use", "preference"}:
            base = max(base, 0.7)
        return max(0.0, min(1.0, base))

    def _tags(self, lesson_type: LessonType, data: ReflectionInput) -> list[str]:
        tags = [lesson_type, "reflection"]
        if lesson_type in {"failure", "tool_use"}:
            tags.append("failure")
        if data.reward > 0:
            tags.append("success")
        if lesson_type == "preference":
            tags.append("user_preference")
        return sorted(set(tags))

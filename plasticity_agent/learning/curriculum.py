"""Skill-evolution curriculum.

Proposes what the agent should practice next, derived from low-confidence skills
and recurring failures. This is a deterministic, advisory planner — it suggests
focus areas; it does not run anything.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from plasticity_agent.learning.skill_library import Skill
from plasticity_agent.memory.salience import FAILURE_TAGS
from plasticity_agent.memory.schemas import Memory


class CurriculumItem(BaseModel):
    """A single recommended focus area."""

    focus: str
    reason: str
    priority: float = Field(ge=0.0, le=1.0)


class Curriculum:
    """Builds a prioritised practice list from skills and memories."""

    def propose(
        self,
        skills: list[Skill] | None = None,
        memories: list[Memory] | None = None,
        *,
        limit: int = 10,
    ) -> list[CurriculumItem]:
        items: list[CurriculumItem] = []

        for skill in skills or []:
            if skill.confidence < 0.6:
                items.append(
                    CurriculumItem(
                        focus=f"Reinforce skill '{skill.name}'",
                        reason=f"low confidence ({skill.confidence:.2f})",
                        priority=min(1.0, 0.6 - skill.confidence + 0.5),
                    )
                )

        for memory in memories or []:
            tags = {tag.lower() for tag in memory.tags}
            if (tags & FAILURE_TAGS) or memory.reward < 0:
                snippet = memory.content[:80]
                items.append(
                    CurriculumItem(
                        focus=f"Address recurring failure: {snippet}",
                        reason="failure-tagged or negative-reward memory",
                        priority=min(1.0, 0.5 + memory.salience * 0.4),
                    )
                )

        items.sort(key=lambda item: item.priority, reverse=True)
        return items[:limit]

"""Learning: reward shaping, the skill library, and the curriculum."""

from __future__ import annotations

from plasticity_agent.learning.curriculum import Curriculum, CurriculumItem
from plasticity_agent.learning.reward import (
    normalize_reward,
    positive_reward,
    shape_reward,
)
from plasticity_agent.learning.skill_library import Skill, SkillLibrary

__all__ = [
    "Curriculum",
    "CurriculumItem",
    "shape_reward",
    "normalize_reward",
    "positive_reward",
    "Skill",
    "SkillLibrary",
]

"""The proposal schema and the base critic interface.

A critic is an opinionated reasoner: given a task and context it returns a
:class:`Proposal` — a candidate action scored along several axes (truthiness,
expected reward, cost, risk, novelty, reversibility, confidence). The reasoning
market then runs an auction over competing proposals.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class Proposal(BaseModel):
    """A scored candidate action put forward by a critic."""

    critic_name: str
    action: str
    rationale: str
    truth_value: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    reversibility: float = Field(ge=0.0, le=1.0)
    expected_reward: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class Critic:
    """Base class for built-in critics. Subclasses implement :meth:`propose`."""

    name: str = "critic"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        raise NotImplementedError

    def _proposal(self, action: str, rationale: str, **scores: float) -> Proposal:
        return Proposal(
            critic_name=self.name,
            action=action,
            rationale=rationale,
            truth_value=clamp01(scores.get("truth_value", 0.5)),
            cost=clamp01(scores.get("cost", 0.5)),
            risk=clamp01(scores.get("risk", 0.5)),
            novelty=clamp01(scores.get("novelty", 0.5)),
            reversibility=clamp01(scores.get("reversibility", 0.5)),
            expected_reward=clamp01(scores.get("expected_reward", 0.5)),
            confidence=clamp01(scores.get("confidence", 0.5)),
        )

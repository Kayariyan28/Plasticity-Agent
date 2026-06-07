"""The auction: turn competing proposals into a ranked decision.

Each proposal is scored by the documented weighting and the highest scorer
wins. The full ranking and a human-readable audit trail are returned so the
decision is inspectable.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from plasticity_agent.reasoning.confidence import selection_confidence
from plasticity_agent.reasoning.critic import Proposal, clamp01


class AuctionResult(BaseModel):
    """The outcome of an auction over proposals."""

    winner: Proposal
    ranked: list[Proposal] = Field(default_factory=list)
    audit_trail: list[str] = Field(default_factory=list)
    selection_score: float = 0.0


def score_proposal(proposal: Proposal) -> float:
    """Documented selection score, clamped to ``[0, 1]``.

    ``0.25*truth + 0.20*reward + 0.15*confidence + 0.10*novelty
    + 0.10*reversibility - 0.10*risk - 0.10*cost``
    """

    score = (
        0.25 * proposal.truth_value
        + 0.20 * proposal.expected_reward
        + 0.15 * proposal.confidence
        + 0.10 * proposal.novelty
        + 0.10 * proposal.reversibility
        - 0.10 * proposal.risk
        - 0.10 * proposal.cost
    )
    return clamp01(score)


def run_auction(proposals: Sequence[Proposal]) -> AuctionResult:
    """Rank ``proposals`` by :func:`score_proposal` and return the winner."""

    if not proposals:
        raise ValueError("run_auction requires at least one proposal")

    scored = sorted(
        ((score_proposal(proposal), proposal) for proposal in proposals),
        key=lambda pair: pair[0],
        reverse=True,
    )
    ranked = [proposal for _, proposal in scored]
    audit_trail = [
        (
            f"{proposal.critic_name}: score={score:.3f} "
            f"(truth={proposal.truth_value:.2f}, reward={proposal.expected_reward:.2f}, "
            f"risk={proposal.risk:.2f}, cost={proposal.cost:.2f}) -> {proposal.action}"
        )
        for score, proposal in scored
    ]
    confidence = selection_confidence([score for score, _ in scored])
    audit_trail.append(f"selection confidence (score+margin): {confidence:.3f}")

    return AuctionResult(
        winner=ranked[0],
        ranked=ranked,
        audit_trail=audit_trail,
        selection_score=scored[0][0],
    )

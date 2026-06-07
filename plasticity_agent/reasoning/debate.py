"""A lightweight, deterministic debate over proposals.

Before the final auction, critics challenge one another: the Skeptic discounts
low-truth proposals, the Risk Analyst penalises high-risk ones, and the
Evidence Auditor rewards well-supported ones. Every adjustment is recorded in a
transcript so the deliberation is fully auditable. This is the deterministic
baseline; an LLM-driven debate is future work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from plasticity_agent.llm.client import LLMClient
from plasticity_agent.reasoning.auction import AuctionResult
from plasticity_agent.reasoning.critic import Critic, Proposal, clamp01
from plasticity_agent.reasoning.market import ReasoningMarket


class DebateResult(BaseModel):
    """Transcript plus final auction outcome of a debate."""

    task: str
    rounds: int
    transcript: list[str] = Field(default_factory=list)
    result: AuctionResult


class Debate:
    """Runs challenge rounds over critic proposals, then auctions them."""

    def __init__(
        self,
        critics: list[Critic] | None = None,
        *,
        llm: LLMClient | Callable[..., Any] | None = None,
    ) -> None:
        self.market = ReasoningMarket(critics, llm=llm)

    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        *,
        rounds: int = 1,
    ) -> DebateResult:
        context = context or {}
        proposals = self.market.proposals_for(task, context)
        transcript: list[str] = []
        total_rounds = max(1, rounds)

        for index in range(total_rounds):
            transcript.append(f"-- round {index + 1} --")
            proposals = self._challenge(proposals, transcript)

        result = self.market.auction(proposals)
        transcript.append(f"winner: {result.winner.critic_name} -> {result.winner.action}")
        return DebateResult(
            task=task, rounds=total_rounds, transcript=transcript, result=result
        )

    def _challenge(self, proposals: list[Proposal], transcript: list[str]) -> list[Proposal]:
        adjusted: list[Proposal] = []
        for proposal in proposals:
            updated = proposal.model_copy(deep=True)
            if proposal.truth_value < 0.5:
                updated.confidence = clamp01(updated.confidence - 0.08)
                transcript.append(
                    f"Skeptic challenges {proposal.critic_name}: "
                    f"low truth_value ({proposal.truth_value:.2f}), confidence -0.08"
                )
            if proposal.risk >= 0.6:
                updated.expected_reward = clamp01(updated.expected_reward - 0.10)
                updated.risk = clamp01(updated.risk + 0.05)
                transcript.append(
                    f"Risk Analyst challenges {proposal.critic_name}: "
                    f"high risk ({proposal.risk:.2f}), expected_reward -0.10"
                )
            if proposal.truth_value >= 0.85:
                updated.confidence = clamp01(updated.confidence + 0.05)
                transcript.append(
                    f"Evidence Auditor backs {proposal.critic_name}: "
                    "strong truth_value, confidence +0.05"
                )
            adjusted.append(updated)
        return adjusted

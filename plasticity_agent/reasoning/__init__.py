"""Critical reasoning: critics, proposals, auctions, market, and debate."""

from __future__ import annotations

from plasticity_agent.reasoning.auction import AuctionResult, run_auction, score_proposal
from plasticity_agent.reasoning.critic import Critic, Proposal
from plasticity_agent.reasoning.critics import (
    Builder,
    CompressionCritic,
    EvidenceAuditor,
    GameTheorist,
    LLMCritic,
    RiskAnalyst,
    Skeptic,
    default_critics,
)
from plasticity_agent.reasoning.debate import Debate, DebateResult
from plasticity_agent.reasoning.market import ReasoningMarket

__all__ = [
    "AuctionResult",
    "run_auction",
    "score_proposal",
    "Critic",
    "Proposal",
    "Skeptic",
    "Builder",
    "RiskAnalyst",
    "EvidenceAuditor",
    "GameTheorist",
    "CompressionCritic",
    "LLMCritic",
    "default_critics",
    "Debate",
    "DebateResult",
    "ReasoningMarket",
]

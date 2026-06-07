"""The reasoning market.

Brings the critic panel and the auction together: :meth:`deliberate` asks every
critic for a proposal and auctions them; :meth:`auction` scores an arbitrary
set of proposals you supply yourself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.llm.client import LLMClient, coerce_llm
from plasticity_agent.reasoning.auction import AuctionResult, run_auction
from plasticity_agent.reasoning.critic import Critic, Proposal
from plasticity_agent.reasoning.critics import LLMCritic, default_critics


class ReasoningMarket:
    """A market of critics that bid on the best next action.

    When an ``llm`` is supplied (and no explicit ``critics`` list is given), an
    LLM-authored critic joins the standard six-critic panel.
    """

    def __init__(
        self,
        critics: list[Critic] | None = None,
        *,
        llm: LLMClient | Callable[..., Any] | None = None,
    ) -> None:
        client = coerce_llm(llm)
        if critics is not None:
            self.critics = critics
        else:
            self.critics = default_critics()
            if client is not None:
                self.critics = [*self.critics, LLMCritic(client)]

    def proposals_for(
        self, task: str, context: dict[str, Any] | None = None
    ) -> list[Proposal]:
        return [critic.propose(task, context or {}) for critic in self.critics]

    def auction(self, proposals: list[Proposal]) -> AuctionResult:
        """Run an auction over a supplied list of proposals."""

        return run_auction(proposals)

    def deliberate(
        self, task: str, context: dict[str, Any] | None = None
    ) -> AuctionResult:
        """Ask every critic to propose, then auction the proposals."""

        return self.auction(self.proposals_for(task, context or {}))

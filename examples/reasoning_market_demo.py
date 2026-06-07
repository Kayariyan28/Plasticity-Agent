"""Reasoning market: six critics bid; an auction selects the winner.

Run:  uv run python examples/reasoning_market_demo.py
"""

from __future__ import annotations

from plasticity_agent import ReasoningMarket
from plasticity_agent.reasoning.debate import Debate


def main() -> None:
    market = ReasoningMarket()
    result = market.deliberate(
        task="Choose best repair strategy for schema error",
        context={"error": "missing required argument"},
    )

    print("== Ranked proposals ==")
    for rank, proposal in enumerate(result.ranked, start=1):
        print(f"  {rank}. {proposal.critic_name:18s} {proposal.action}")

    print(f"\nWinner: {result.winner.critic_name} (selection score {result.selection_score:.3f})")
    print("\n== Audit trail ==")
    for line in result.audit_trail:
        print(f"  {line}")

    print("\n== Debate (2 rounds) ==")
    debate = Debate().run("Should we deploy the migration on Friday?", rounds=2)
    for line in debate.transcript[-6:]:
        print(f"  {line}")


if __name__ == "__main__":
    main()

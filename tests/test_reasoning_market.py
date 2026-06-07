"""Tests for the critic-driven reasoning market and auction."""

from __future__ import annotations

import pytest

from plasticity_agent.reasoning.auction import run_auction, score_proposal
from plasticity_agent.reasoning.critic import Proposal
from plasticity_agent.reasoning.critics import default_critics
from plasticity_agent.reasoning.debate import Debate
from plasticity_agent.reasoning.market import ReasoningMarket


def test_deliberate_returns_winner_and_full_ranking() -> None:
    result = ReasoningMarket().deliberate(
        "Choose best repair strategy for schema error",
        {"error": "missing required argument"},
    )
    assert result.winner is not None
    assert 0.0 <= result.selection_score <= 1.0
    assert len(result.ranked) == 6
    assert result.audit_trail


def test_all_proposal_scores_in_unit_range() -> None:
    for critic in default_critics():
        proposal = critic.propose("do something useful", {})
        assert 0.0 <= score_proposal(proposal) <= 1.0


def test_ranking_is_sorted_descending() -> None:
    result = ReasoningMarket().deliberate("a representative task")
    scores = [score_proposal(p) for p in result.ranked]
    assert scores == sorted(scores, reverse=True)


def test_auction_requires_proposals() -> None:
    with pytest.raises(ValueError):
        run_auction([])


def test_strong_proposal_beats_weak_one() -> None:
    strong = Proposal(
        critic_name="Strong",
        action="x",
        rationale="r",
        truth_value=0.9,
        cost=0.1,
        risk=0.1,
        novelty=0.6,
        reversibility=0.9,
        expected_reward=0.9,
        confidence=0.9,
    )
    weak = Proposal(
        critic_name="Weak",
        action="y",
        rationale="r",
        truth_value=0.1,
        cost=0.9,
        risk=0.9,
        novelty=0.1,
        reversibility=0.1,
        expected_reward=0.1,
        confidence=0.1,
    )
    assert run_auction([weak, strong]).winner.critic_name == "Strong"


def test_debate_runs_and_selects() -> None:
    result = Debate().run("decide whether to deploy on Friday", rounds=2)
    assert result.rounds == 2
    assert result.transcript
    assert result.result.winner is not None

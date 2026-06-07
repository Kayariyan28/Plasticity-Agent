# Reasoning Market

The reasoning market is an internal auction: a panel of **critics** each submit a scored
`Proposal` for the best next action, and an auction ranks them. This gives deliberation genuine
diversity of perspective and a fully auditable decision.

## The six critics

| Critic | Bias |
| --- | --- |
| **Skeptic** | Verify assumptions first; high truth, low risk |
| **Builder** | Ship a minimal version; high reward, higher risk |
| **Risk Analyst** | Mitigate downside; very low risk, high reversibility |
| **Evidence Auditor** | Gather and cite evidence; very high truth |
| **Game Theorist** | Pick the move robust to likely responses |
| **Compression Critic** | Simplest sufficient approach; lowest cost |

Each adjusts its scores to signals in the task/context (risky/irreversible language, a present
error, a need for evidence, strategic cues).

## `Proposal` and scoring

```python
Proposal(critic_name, action, rationale,
         truth_value, cost, risk, novelty, reversibility, expected_reward, confidence)
```

The auction scores each proposal (clamped to `[0, 1]`):

```text
score =  0.25*truth_value + 0.20*expected_reward + 0.15*confidence
       + 0.10*novelty + 0.10*reversibility
       - 0.10*risk - 0.10*cost
```

## Deliberate

```python
from plasticity_agent import ReasoningMarket

market = ReasoningMarket()
result = market.deliberate(
    task="Choose best repair strategy for schema error",
    context={"error": "missing required argument"},
)
print(result.winner.critic_name, "->", result.winner.action)
for line in result.audit_trail:
    print(line)
```

`AuctionResult` gives you the `winner`, the full `ranked` list, an `audit_trail`, and a
`selection_score`. The selection confidence blends the winner's score with its *margin* over the
runner-up, so a near-tie is reported as low confidence.

## Auction your own proposals

```python
from plasticity_agent.reasoning.auction import run_auction
result = run_auction([proposal_a, proposal_b, proposal_c])
```

## Debate

```python
from plasticity_agent.reasoning.debate import Debate

debate = Debate().run("Should we deploy the migration on Friday?", rounds=2)
print(debate.result.winner.critic_name)
for line in debate.transcript:
    print(line)
```

Before the final auction, critics challenge each other (the Skeptic discounts low-truth proposals,
the Risk Analyst penalises high-risk ones, the Evidence Auditor rewards well-supported ones), and
every adjustment is recorded in the transcript.

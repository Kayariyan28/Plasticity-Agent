"""The six built-in critics.

Each critic is a deterministic, opinionated reasoner with its own scoring
bias, lightly modulated by signals detected in the task/context (e.g. risky or
irreversible language, a present error, a need for evidence). Together they
give the auction genuine diversity of perspective.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plasticity_agent.llm.client import LLMClient, clamp01, coerce_llm, complete_json
from plasticity_agent.reasoning.critic import Critic, Proposal

_RISKY = (
    "delete", "drop", "irreversible", "production", "payment", "money",
    "destructive", "overwrite", "rm -rf", "migrate", "wipe", "deploy",
)
_UNCERTAIN = ("unknown", "unclear", "ambiguous", "uncertain", "maybe", "not sure")
_EVIDENCE = ("claim", "paper", "fact", "cite", "citation", "evidence", "research", "data")
_STRATEGIC = ("competitor", "opponent", "negotiat", "bid", "auction", "adversar", "rival")


def _text(task: str, context: dict[str, Any] | None) -> str:
    parts = [task]
    if context:
        parts.extend(str(value) for value in context.values())
    return " ".join(parts).lower()


def _signals(task: str, context: dict[str, Any] | None) -> dict[str, bool]:
    text = _text(task, context)
    has_error = bool(context and ("error" in context)) or "error" in text
    return {
        "risky": any(token in text for token in _RISKY),
        "uncertain": any(token in text for token in _UNCERTAIN),
        "evidence": any(token in text for token in _EVIDENCE),
        "strategic": any(token in text for token in _STRATEGIC),
        "error": has_error,
    }


class Skeptic(Critic):
    name = "Skeptic"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        s = _signals(task, context)
        reward = 0.5 + (0.1 if s["uncertain"] else 0.0) + (0.1 if s["error"] else 0.0)
        return self._proposal(
            action=f"Verify the key assumptions behind '{task}' before committing.",
            rationale=(
                "Unverified assumptions are the dominant failure mode; "
                "reduce uncertainty first."
            ),
            truth_value=0.82,
            cost=0.45,
            risk=0.2,
            novelty=0.3,
            reversibility=0.85,
            expected_reward=reward,
            confidence=0.72,
        )


class Builder(Critic):
    name = "Builder"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        s = _signals(task, context)
        risk = 0.55 + (0.2 if s["risky"] else 0.0)
        reversibility = 0.55 - (0.2 if s["risky"] else 0.0)
        reward = 0.85 - (0.1 if s["uncertain"] else 0.0)
        confidence = 0.7 - (0.1 if s["uncertain"] else 0.0)
        return self._proposal(
            action=f"Ship a minimal working version of '{task}', then iterate.",
            rationale="A working draft generates feedback faster than further analysis.",
            truth_value=0.55,
            cost=0.5,
            risk=risk,
            novelty=0.7,
            reversibility=reversibility,
            expected_reward=reward,
            confidence=confidence,
        )


class RiskAnalyst(Critic):
    name = "Risk Analyst"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        s = _signals(task, context)
        reward = 0.5 + (0.12 if s["risky"] else 0.0) + (0.05 if s["error"] else 0.0)
        confidence = 0.75 + (0.1 if s["risky"] else 0.0)
        return self._proposal(
            action=f"Mitigate the top failure modes of '{task}' (guards, backups, rollback).",
            rationale="Protect the downside and keep the action reversible before proceeding.",
            truth_value=0.65,
            cost=0.5,
            risk=0.15,
            novelty=0.25,
            reversibility=0.9,
            expected_reward=reward,
            confidence=confidence,
        )


class EvidenceAuditor(Critic):
    name = "Evidence Auditor"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        s = _signals(task, context)
        truth = 0.9 + (0.05 if s["evidence"] else 0.0)
        reward = 0.55 + (0.12 if s["evidence"] else 0.0)
        return self._proposal(
            action=f"Collect and cite evidence supporting '{task}'; flag unsupported claims.",
            rationale="Decisions grounded in verifiable sources beat confident assertion.",
            truth_value=truth,
            cost=0.55,
            risk=0.2,
            novelty=0.25,
            reversibility=0.85,
            expected_reward=reward,
            confidence=0.7,
        )


class GameTheorist(Critic):
    name = "Game Theorist"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        s = _signals(task, context)
        reward = 0.7 + (0.1 if s["strategic"] else 0.0)
        novelty = 0.65 + (0.1 if s["strategic"] else 0.0)
        risk = 0.45 + (0.1 if s["risky"] else 0.0)
        return self._proposal(
            action=f"Pick the move for '{task}' that is robust to the most likely responses.",
            rationale="Optimize for the equilibrium outcome, not just the immediate payoff.",
            truth_value=0.6,
            cost=0.45,
            risk=risk,
            novelty=novelty,
            reversibility=0.5,
            expected_reward=reward,
            confidence=0.66,
        )


class CompressionCritic(Critic):
    name = "Compression Critic"

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        complex_task = len(task) > 140
        cost = 0.15 + (0.05 if complex_task else 0.0)
        return self._proposal(
            action=f"Solve '{task}' with the simplest sufficient approach; cut wasted steps.",
            rationale="Minimize free energy: less compute, fewer moving parts, lower entropy.",
            truth_value=0.6,
            cost=cost,
            risk=0.25,
            novelty=0.3,
            reversibility=0.85,
            expected_reward=0.6,
            confidence=0.7,
        )


class LLMCritic(Critic):
    """A critic whose proposal is authored by an LLM.

    Falls back to a neutral proposal if the model call or parse fails, so the
    auction always has a complete, valid set of proposals to score.
    """

    name = "LLM Strategist"

    def __init__(
        self,
        llm: LLMClient | Callable[..., Any],
        *,
        name: str | None = None,
        persona: str = "a pragmatic strategist who balances reward against risk",
    ) -> None:
        self.llm = coerce_llm(llm)
        if name:
            self.name = name
        self.persona = persona

    def propose(self, task: str, context: dict[str, Any] | None = None) -> Proposal:
        prompt = (
            f"You are {self.persona}. Propose the single best next action for the task "
            "below and score it. Respond with ONLY JSON:\n"
            '{"action": "...", "rationale": "...", "truth_value": 0-1, "cost": 0-1, '
            '"risk": 0-1, "novelty": 0-1, "reversibility": 0-1, "expected_reward": 0-1, '
            '"confidence": 0-1}\n\n'
            f"task: {task}\ncontext: {context or {}}\n"
        )
        obj = complete_json(self.llm, prompt)
        if isinstance(obj, dict) and obj.get("action"):
            return Proposal(
                critic_name=self.name,
                action=str(obj["action"]),
                rationale=str(obj.get("rationale") or ""),
                truth_value=clamp01(obj.get("truth_value", 0.6)),
                cost=clamp01(obj.get("cost", 0.5)),
                risk=clamp01(obj.get("risk", 0.4)),
                novelty=clamp01(obj.get("novelty", 0.5)),
                reversibility=clamp01(obj.get("reversibility", 0.6)),
                expected_reward=clamp01(obj.get("expected_reward", 0.6)),
                confidence=clamp01(obj.get("confidence", 0.6)),
            )
        return self._proposal(
            action=f"Address '{task}' with a balanced, reversible approach.",
            rationale="LLM unavailable; neutral fallback proposal.",
            truth_value=0.5,
            cost=0.5,
            risk=0.5,
            novelty=0.5,
            reversibility=0.5,
            expected_reward=0.5,
            confidence=0.4,
        )


def default_critics() -> list[Critic]:
    """The standard six-critic panel used by :class:`ReasoningMarket`."""

    return [
        Skeptic(),
        Builder(),
        RiskAnalyst(),
        EvidenceAuditor(),
        GameTheorist(),
        CompressionCritic(),
    ]

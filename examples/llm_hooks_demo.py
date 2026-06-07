"""LLM-backed reflection, contradiction, and critic proposals.

Uses a fake model here for determinism; swap ``fake_llm`` for a real callback
(``fn(prompt) -> str``) to drive these with an actual LLM.

Run:  uv run python examples/llm_hooks_demo.py
"""

from __future__ import annotations

from plasticity_agent.memory.contradiction import ContradictionChecker
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.reflection.lessons import ReflectionInput
from plasticity_agent.reflection.reflector import Reflector


def fake_llm(prompt: str, **_kwargs: object) -> str:
    lowered = prompt.lower()
    if "contradict" in lowered:
        return '{"contradiction": 0.9, "reason": "opposite stance"}'
    if "reflection module" in lowered:
        return (
            '{"lesson_type": "reasoning", '
            '"content": "Verify figures against the source before reporting.", '
            '"confidence": 0.82, "tags": ["accuracy"]}'
        )
    return (
        '{"action": "Run a cheap experiment to de-risk the unknown", '
        '"rationale": "high information gain at low cost", "truth_value": 0.8, '
        '"cost": 0.2, "risk": 0.2, "novelty": 0.7, "reversibility": 0.9, '
        '"expected_reward": 0.85, "confidence": 0.8}'
    )


def main() -> None:
    lesson = Reflector(llm=fake_llm).create_lesson(
        ReflectionInput(task="estimate quarterly revenue", reward=0.6)
    )
    print(f"LLM lesson [{lesson.lesson_type}]: {lesson.content}")

    score = ContradictionChecker(llm=fake_llm).pair("ship it today", "do not ship it today")
    print(f"LLM contradiction score: {score}")

    result = ReasoningMarket(llm=fake_llm).deliberate("Choose the next step on an ambiguous task")
    print(f"market winner: {result.winner.critic_name} -> {result.winner.action}")


if __name__ == "__main__":
    main()

"""Tests for the LLM layer and LLM-backed features (using fake callbacks)."""

from __future__ import annotations

from plasticity_agent import PlasticAgent
from plasticity_agent.llm.client import coerce_llm, complete_json, extract_json
from plasticity_agent.memory.contradiction import ContradictionChecker
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.reflection.lessons import ReflectionInput
from plasticity_agent.reflection.reflector import Reflector


def test_extract_json_from_noise() -> None:
    assert extract_json('blah blah {"a": 1, "b": [2, 3]} trailing') == {"a": 1, "b": [2, 3]}
    assert extract_json("```json\n{\"x\": 2}\n```") == {"x": 2}
    assert extract_json("no json here at all") is None


def test_coerce_callable_llm_signatures() -> None:
    assert coerce_llm(None) is None
    assert coerce_llm(lambda prompt: "hi").complete("x") == "hi"
    with_system = coerce_llm(lambda prompt, system=None: f"{system}|{prompt}")
    assert with_system.complete("q", system="sys") == "sys|q"


def test_complete_json_returns_none_on_garbage() -> None:
    assert complete_json(coerce_llm(lambda p: "not json"), "x") is None


def test_llm_reflection_overrides_rules() -> None:
    def fake(prompt: str, **_kw: object) -> str:
        return (
            '{"lesson_type": "reasoning", "content": "Cite sources next time.", '
            '"confidence": 0.8, "tags": ["accuracy"]}'
        )

    # Rules alone would classify positive reward as "success"; the LLM overrides.
    lesson = Reflector(llm=fake).create_lesson(ReflectionInput(task="t", reward=0.9))
    assert lesson.lesson_type == "reasoning"
    assert "Cite sources" in lesson.content


def test_llm_reflection_falls_back_on_bad_json() -> None:
    lesson = Reflector(llm=lambda p, **k: "totally not json").create_lesson(
        ReflectionInput(task="t", reward=0.9)
    )
    assert lesson.lesson_type == "success"  # deterministic fallback


def test_llm_contradiction_checker_uses_model() -> None:
    checker = ContradictionChecker(llm=lambda p, **k: '{"contradiction": 0.95}')
    # Texts are related enough to pass the lexical gate, so the LLM is consulted.
    score = checker.pair("the cache is enabled", "the cache is enabled fully")
    assert score == 0.95


def test_llm_critic_joins_and_can_win_market() -> None:
    def fake(prompt: str, **_kw: object) -> str:
        return (
            '{"action": "do the LLM-recommended thing", "rationale": "r", '
            '"truth_value": 0.99, "cost": 0.0, "risk": 0.0, "novelty": 0.9, '
            '"reversibility": 0.9, "expected_reward": 0.99, "confidence": 0.99}'
        )

    market = ReasoningMarket(llm=fake)
    assert len(market.critics) == 7  # six built-ins + the LLM critic
    result = market.deliberate("pick the best action")
    assert result.winner.critic_name == "LLM Strategist"


def test_agent_uses_llm_for_reflection(tmp_path) -> None:
    def fake(task: str, **_kw: object) -> str:
        return '{"lesson_type": "risk", "content": "watch the edge case", "confidence": 0.7}'

    agent = PlasticAgent(
        name="llm", memory=str(tmp_path / "m"), llm_callback=fake, reasoning_market=False
    )
    try:
        lesson = agent.reflect(task="t", reward=0.9)
        assert lesson.lesson_type == "risk"
    finally:
        agent.close()

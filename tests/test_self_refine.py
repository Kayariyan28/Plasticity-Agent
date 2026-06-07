"""Tests for the Self-Refine loop (deterministic + LLM-backed)."""

from __future__ import annotations

from plasticity_agent.reflection.self_refine import SelfRefine


def test_deterministic_refine_returns_all_fields() -> None:
    result = SelfRefine().refine(
        "It will always work, probably.", rubric="accuracy, safety, completeness"
    )
    assert result.original_output
    assert result.critique
    assert result.refined_output
    assert result.rubric == "accuracy, safety, completeness"
    assert 0.0 <= result.improvement_score <= 1.0


def test_refine_flags_vague_and_unsupported_claims() -> None:
    result = SelfRefine().refine("This is probably fine and will always succeed without issue.")
    lowered = result.critique.lower()
    assert "vague" in lowered or "claim" in lowered
    assert result.improvement_score > 0.0
    assert "[Self-Refine notes]" in result.refined_output


def test_clean_output_has_low_improvement() -> None:
    clean = (
        "The migration ran in 12 minutes because the indexes were prebuilt; "
        "rows verified against the source checksum with zero mismatches."
    )
    result = SelfRefine().refine(clean, rubric="accuracy")
    assert result.improvement_score <= 0.2


def test_llm_callback_path_is_used() -> None:
    def callback(prompt: str, rubric: str) -> str:
        return "A precise, well-supported rewrite."

    result = SelfRefine(llm_callback=callback).refine("vague maybe stuff", rubric="accuracy")
    assert result.refined_output == "A precise, well-supported rewrite."
    assert "LLM" in result.critique

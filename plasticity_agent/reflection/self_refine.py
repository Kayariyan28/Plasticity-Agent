"""Self-Refine: iteratively critique and improve an output.

By default this is fully deterministic (no LLM): it runs a small battery of
checks against a rubric and appends actionable refinement notes. Pass an
``llm_callback`` to delegate the critique+rewrite to a model instead.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from plasticity_agent.memory.retrieval import lexical_similarity

_VAGUE_PHRASES = (
    "maybe",
    "probably",
    "i think",
    "kind of",
    "sort of",
    "a lot",
    "as needed",
    "and so on",
    "etc",
    "stuff",
    "things",
)

_UNSUPPORTED_MARKERS = ("always", "never", "guaranteed", "proven", "everyone", "no one")

RefineCallback = Callable[[str, str], str]


class SelfRefineResult(BaseModel):
    """The outcome of one refine pass."""

    original_output: str
    critique: str
    refined_output: str
    rubric: str
    improvement_score: float


class SelfRefine:
    """Deterministic-by-default self-refinement with an optional LLM hook."""

    def __init__(self, llm_callback: RefineCallback | None = None) -> None:
        self.llm_callback = llm_callback

    def refine(
        self,
        output: str,
        rubric: str = "accuracy, safety, completeness",
        *,
        min_length: int = 40,
    ) -> SelfRefineResult:
        if self.llm_callback is not None:
            return self._refine_with_llm(output, rubric)
        return self._refine_deterministic(output, rubric, min_length)

    # -- LLM-backed path ------------------------------------------------------

    def _refine_with_llm(self, output: str, rubric: str) -> SelfRefineResult:
        prompt = (
            "Critique the following output against the rubric, then rewrite it to "
            f"address every weakness.\nRubric: {rubric}\n\nOutput:\n{output}"
        )
        refined = str(self.llm_callback(prompt, rubric))  # type: ignore[misc]
        improvement = max(0.05, min(0.95, 1.0 - lexical_similarity(output, refined)))
        return SelfRefineResult(
            original_output=output,
            critique="LLM critique applied against rubric.",
            refined_output=refined,
            rubric=rubric,
            improvement_score=improvement,
        )

    # -- deterministic path ---------------------------------------------------

    def _refine_deterministic(
        self, output: str, rubric: str, min_length: int
    ) -> SelfRefineResult:
        issues, fixes = self._inspect(output, rubric, min_length)

        if issues:
            critique = "\n".join(f"- {issue}" for issue in issues)
            refined = output.rstrip() + "\n\n[Self-Refine notes]\n" + "\n".join(
                f"- {fix}" for fix in fixes
            )
            improvement = min(1.0, 0.12 * len(issues) + 0.1)
        else:
            critique = "No major issues detected by deterministic checks."
            refined = output
            improvement = 0.05

        return SelfRefineResult(
            original_output=output,
            critique=critique,
            refined_output=refined,
            rubric=rubric,
            improvement_score=round(improvement, 3),
        )

    def _inspect(
        self, output: str, rubric: str, min_length: int
    ) -> tuple[list[str], list[str]]:
        issues: list[str] = []
        fixes: list[str] = []
        lowered = output.lower()
        rubric_lower = rubric.lower()

        if len(output.strip()) < min_length:
            issues.append("output is very short for the rubric")
            fixes.append("Expand with concrete detail and an example.")

        if "completeness" in rubric_lower and output.count("\n") == 0 and len(output) > 200:
            issues.append("long output has no structure (completeness)")
            fixes.append("Break the answer into labelled sections or bullet points.")

        found_vague = sorted({p for p in _VAGUE_PHRASES if p in lowered})
        if found_vague:
            issues.append(f"vague phrasing: {', '.join(found_vague)}")
            fixes.append("Replace vague phrasing with specific, verifiable statements.")

        found_claims = sorted({m for m in _UNSUPPORTED_MARKERS if m in lowered})
        if found_claims and "because" not in lowered and "source" not in lowered:
            issues.append(f"strong claims without support: {', '.join(found_claims)}")
            fixes.append("Qualify absolute claims or cite supporting evidence.")

        if "safety" in rubric_lower and any(
            risky in lowered for risky in ("delete", "rm -rf", "drop table", "secret", "password")
        ):
            issues.append("possible safety-sensitive content")
            fixes.append("Confirm safety implications and avoid destructive instructions.")

        return issues, fixes

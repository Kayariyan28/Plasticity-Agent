"""Self-Refine: deterministic critique + an optional LLM-backed rewrite.

Run:  uv run python examples/self_refine_demo.py
"""

from __future__ import annotations

from plasticity_agent.reflection.self_refine import SelfRefine


def main() -> None:
    draft = "It will always work, probably. We deleted the old data to keep things simple."

    print("== Deterministic refine ==")
    result = SelfRefine().refine(draft, rubric="accuracy, safety, completeness")
    print("critique:")
    print(result.critique)
    print(f"\nimprovement_score: {result.improvement_score}")
    print("refined_output:")
    print(result.refined_output)

    print("\n== LLM-backed refine (callback) ==")

    def fake_llm(prompt: str, rubric: str) -> str:
        return "The approach succeeds under the tested conditions; no data was deleted."

    refined = SelfRefine(llm_callback=fake_llm).refine(draft, rubric="accuracy, safety")
    print(f"critique: {refined.critique}")
    print(f"refined_output: {refined.refined_output}")
    print(f"improvement_score: {refined.improvement_score:.3f}")


if __name__ == "__main__":
    main()

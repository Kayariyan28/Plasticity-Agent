"""Reflexion-style loop: store lessons from feedback across several runs.

Run:  uv run python examples/reflection_loop.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent import PlasticAgent


def main() -> None:
    agent = PlasticAgent(name="reflector", memory=tempfile.mkdtemp(prefix="plasticity_reflect_"))

    episodes = [
        {"task": "call the pricing tool", "error": "TypeError: missing required argument: 'sku'",
         "reward": -0.6},
        {"task": "summarize the contract", "reward": 0.85},
        {"task": "answer the user", "evaluator_feedback": "I prefer concise, bulleted answers"},
        {"task": "estimate the figure",
         "evaluator_feedback": "the accuracy was poor; cite sources"},
    ]

    for episode in episodes:
        lesson = agent.reflect(**episode)
        print(f"[{lesson.lesson_type:10s}] {lesson.content}")

    print("\n== Stored reflective memories ==")
    for memory in agent.memory.list_memories(memory_type="reflective"):
        print(f"  ({', '.join(memory.tags)}) {memory.content[:80]}")

    agent.close()


if __name__ == "__main__":
    main()

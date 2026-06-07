"""Cross-run improvement metrics: did the agent actually get better?

This shows a real, measurable improvement: the agent starts with a contradiction
in memory, then resolves it (and adds a consistent fact), and the tracker reports
falling contradiction pressure and a rising plasticity score.

Run:  uv run python examples/improvement_metrics_demo.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent import PlasticAgent


def main() -> None:
    agent = PlasticAgent(name="improver", memory=tempfile.mkdtemp(prefix="plasticity_metrics_"))

    # Round 1: memory contains a contradiction.
    agent.remember("Deploys on Friday are safe", "semantic", confidence=0.6)
    conflicting = agent.remember("Deploys on Friday are not safe", "semantic", confidence=0.6)
    first = agent.checkpoint("before")
    print(f"before: plasticity={first.plasticity_score:.0f}, "
          f"contradiction={first.contradiction_pressure:.2f}")

    # Round 2: resolve the contradiction, add a consistent fact.
    agent.memory.prune(memory_ids=[conflicting.id])
    agent.remember("Post-deploy smoke tests pass consistently", "semantic", confidence=0.6)
    second = agent.checkpoint("after")
    print(f"after:  plasticity={second.plasticity_score:.0f}, "
          f"contradiction={second.contradiction_pressure:.2f}")

    report = agent.improvement()
    print(f"\nimproved: {report.improved}")
    print(report.summary)

    agent.close()


if __name__ == "__main__":
    main()

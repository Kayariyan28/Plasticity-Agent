"""Sleep cycle: decay, consolidate, detect contradictions, mine skills.

Seeds a few similar memories and successful run traces, then sleeps and prints
the report in the CLI's checkmark style.

Run:  uv run python examples/sleep_cycle_demo.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent.core.events import RUN_COMPLETED
from plasticity_agent.memory.memory_os import MemoryOS


def main() -> None:
    memory = MemoryOS(memory_dir=tempfile.mkdtemp(prefix="plasticity_sleep_"))

    # Repeated reflective insights -> consolidate into semantic gist.
    memory.record("The payment API call timed out", "reflective", tags=["failure"])
    memory.record("The payment API call timed out again under load", "reflective", tags=["failure"])
    memory.record("The payment API call timed out during the spike", "reflective", tags=["failure"])
    # A contradiction.
    memory.record("Deploys on Friday are safe", "semantic")
    memory.record("Deploys on Friday are not safe", "semantic")
    # Repeated successful runs -> mine a skill + procedural memory.
    for _ in range(3):
        memory.tracer.emit(
            RUN_COMPLETED, {"task": "generate the weekly status report", "reward": 0.9}
        )

    report = memory.sleep()
    print(f"✓ {report.traces_analyzed} traces analyzed")
    print(f"✓ {report.weak_memories_decayed} weak memories decayed")
    print(f"✓ {report.memories_consolidated} memories consolidated")
    print(f"✓ {report.contradictions_detected} contradictions detected")
    print(f"✓ {report.skills_created} reusable skills created")
    print(f"✓ {report.policies_improved} prompt policies improved")
    print(f"✓ agent plasticity score: {report.plasticity_score:.0f}/100")

    print("\nLearned skills:")
    for skill in memory.skills.list_skills():
        print(f"  - {skill.name}: {skill.description}")

    memory.close()


if __name__ == "__main__":
    main()

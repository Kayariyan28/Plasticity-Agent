"""Basic memory: record, recall, and evaluate memory quality.

Run:  uv run python examples/basic_memory.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent.memory.memory_os import MemoryOS


def main() -> None:
    memory_dir = tempfile.mkdtemp(prefix="plasticity_basic_")
    memory = MemoryOS(memory_dir=memory_dir)
    print(f"Memory directory: {memory_dir}\n")

    memory.record("User prefers concise, well-cited answers.", "constitutional",
                  tags=["user_preference"])
    memory.record("Cache warmup reduced p95 latency by 40%.", "semantic",
                  tags=["important"], reward=0.9)
    memory.record("Tried retrying the flaky upload; it eventually succeeded.", "episodic",
                  reward=0.4)
    memory.record("The build is green.", "episodic")
    memory.record("The build is not green.", "episodic")  # contradicts the previous one

    print("== Recall: 'latency cache' ==")
    for hit in memory.recall("latency cache"):
        print(f"  {hit.score:.3f}  {hit.memory.content}  ({hit.match_reason})")

    print("\n== Memory quality ==")
    for report in sorted(memory.evaluate_all(), key=lambda r: r.utility_score, reverse=True):
        print(f"  utility={report.utility_score:.3f}  {report.recommendation:11s}  "
              f"{'; '.join(report.reasons)}")

    memory.close()


if __name__ == "__main__":
    main()

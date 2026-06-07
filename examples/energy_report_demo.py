"""Energy report: a thermodynamic-style reliability snapshot.

Run:  uv run python examples/energy_report_demo.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent import PlasticAgent
from plasticity_agent.core.events import RUN_COMPLETED, RUN_FAILED


def main() -> None:
    agent = PlasticAgent(name="energy_demo", memory=tempfile.mkdtemp(prefix="plasticity_energy_"))

    agent.remember("Stable, well-cited fact about the system.", "semantic",
                   tags=["important"], confidence=0.9, reward=0.8)
    agent.remember("A shaky, low-confidence guess.", "episodic", confidence=0.2)
    agent.remember("Contradictory note: the cache is enabled.", "semantic")
    agent.remember("Contradictory note: the cache is not enabled.", "semantic")

    # Simulate some run traces, including a failure (wasted compute).
    agent.tracer.emit(RUN_COMPLETED, {"task": "build summary", "output": "ok" * 80, "reward": 0.9})
    agent.tracer.emit(RUN_FAILED, {"task": "fetch data", "error": "TimeoutError: timed out"})

    energy = agent.energy_report()
    print(f"memory_entropy:         {energy.memory_entropy}")
    print(f"contradiction_pressure: {energy.contradiction_pressure}")
    print(f"token_waste:            {energy.token_waste} tokens")
    print(f"repair_energy:          {energy.repair_energy}")
    print(f"confidence_temperature: {energy.confidence_temperature}")
    print(f"plasticity_score:       {energy.plasticity_score}/100")
    print(f"\n{energy.summary}")

    agent.close()


if __name__ == "__main__":
    main()

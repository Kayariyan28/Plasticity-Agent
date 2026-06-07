"""The README quickstart, end to end (advisory mode, no LLM required).

Run:  uv run python examples/plastic_agent_quickstart.py
"""

from __future__ import annotations

import tempfile

from plasticity_agent import PlasticAgent


def main() -> None:
    agent = PlasticAgent(
        name="research_copilot",
        model="openai:gpt-5.5",
        memory=tempfile.mkdtemp(prefix="plasticity_quickstart_"),
        self_heal=True,
        reasoning_market=True,
        sleep_cycle=True,
    )

    result = agent.run(
        "Read this paper, extract claims, critique methodology, and produce a reproducible summary."
    )
    print(f"Run status: {result.status}")
    print(f"Suggested next action: {result.output.get('suggested_action')}\n")

    agent.reflect(reward=0.8)
    sleep_report = agent.sleep()
    print(sleep_report.summary)

    energy = agent.energy_report()
    print(energy.summary)

    agent.close()


if __name__ == "__main__":
    main()

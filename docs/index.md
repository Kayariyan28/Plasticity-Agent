# Plasticity Agent Runtime

> Neuroplastic memory, self-healing, and critical reasoning for AI agents.

`plasticity-agent` is a **local-first, framework-agnostic** Python runtime that gives agentic
AI systems the machinery to *measurably improve across runs*. It ships deterministic baselines
for everything, so it works with no LLM, with a Python callable, or with any model via a single
`llm_callback`.

## Install

```bash
pip install plasticity-agent      # or: uv add plasticity-agent
```

## Quickstart

```python
from plasticity_agent import PlasticAgent

agent = PlasticAgent(
    name="research_copilot",
    model="openai:gpt-5.5",
    memory="./memory",
    self_heal=True,
    reasoning_market=True,
    sleep_cycle=True,
)

result = agent.run("Read this paper, extract claims, and produce a reproducible summary.")
agent.reflect(reward=0.8)
agent.sleep()
print(agent.energy_report())
```

## What you get

- **[Memory](memory.md)** — episodic / semantic / procedural / reflective / constitutional
  memories with plasticity signals and deterministic quality scoring.
- **[Retrieval](retrieval.md)** — lexical by default; hybrid lexical+vector with one
  `embeddings=` argument (zero-dep hashing, sentence-transformers, or any embed callable).
- **[LLM features](llm.md)** — opt-in model-authored reflection, semantic contradiction, an LLM
  critic, and LLM self-refine — each with a deterministic fallback.
- **[Sleep cycle](sleep_cycle.md)** — decay, de-duplicate, consolidate into gist, mine reusable
  skills, suggest policies, and govern the constitution.
- **[Self-healing](self_healing.md)** — diagnose errors; advisory by default, with **opt-in**
  sandboxed repair.
- **[Reasoning market](reasoning_market.md)** — six critics (plus an LLM critic) bid; an auction
  selects the winner with a full audit trail.
- **Energy report** — entropy, contradiction pressure, wasted compute, and confidence
  "temperature" rolled into a plasticity score.
- **[Improvement metrics](metrics.md)** — checkpoint health and prove the agent is getting better.
- **[Observability](observability.md)** — stream every trace event to OpenTelemetry.

## CLI

```bash
plasticity init ./memory
plasticity remember "User prefers concise answers" --type constitutional --tag user_preference
plasticity recall "preferences"
plasticity sleep ./agent_runs
plasticity report
plasticity market "Choose a rollout strategy"
plasticity serve        # FastAPI
plasticity dashboard    # Streamlit
```

See **[Concepts](concepts.md)** for the research foundations and the safety positioning.

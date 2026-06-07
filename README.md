<div align="center">

# 🧠 Plasticity Agent Runtime

**Neuroplastic memory, self-healing, and critical reasoning for AI agents.**

[![PyPI version](https://img.shields.io/pypi/v/plasticity-agent.svg)](https://pypi.org/project/plasticity-agent/)
[![Python](https://img.shields.io/pypi/pyversions/plasticity-agent.svg)](https://pypi.org/project/plasticity-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-122%20passing-brightgreen.svg)](#testing)
[![Typed](https://img.shields.io/badge/mypy-clean-blue.svg)](#)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

*A local-first, framework-agnostic Python runtime that gives agents an evolving memory,
sleep-like consolidation, reflection, advisory (opt-in executable) self-healing, a
critic-driven reasoning market, a skill library, and thermodynamic-style reliability reporting.*

</div>

---

## Table of contents

- [Why Plasticity Agent?](#why-plasticity-agent)
- [Feature matrix](#feature-matrix)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Use cases & examples](#use-cases--examples)
  - [1. Neuroplastic memory](#1-neuroplastic-memory)
  - [2. Memory quality scoring](#2-memory-quality-scoring)
  - [3. Hybrid lexical + vector retrieval](#3-hybrid-lexical--vector-retrieval)
  - [4. LLM-powered upgrades](#4-llm-powered-upgrades)
  - [5. Reflection (Reflexion)](#5-reflection-reflexion)
  - [6. Self-Refine](#6-self-refine)
  - [7. Sleep / consolidation](#7-sleep--consolidation)
  - [8. Advisory & opt-in self-healing](#8-advisory--opt-in-self-healing)
  - [9. Critical reasoning market](#9-critical-reasoning-market)
  - [10. Skill library](#10-skill-library)
  - [11. Energy report](#11-energy-report)
  - [12. Cross-run improvement metrics](#12-cross-run-improvement-metrics)
  - [13. OpenTelemetry export](#13-opentelemetry-export)
  - [14. Framework adapters](#14-framework-adapters)
- [Command-line interface](#command-line-interface)
- [FastAPI server](#fastapi-server)
- [Streamlit dashboard](#streamlit-dashboard)
- [Configuration & storage](#configuration--storage)
- [Research foundation](#research-foundation)
- [Safety](#safety)
- [Testing & development](#testing--development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Why Plasticity Agent?

Most "agent memory" is a vector store: embed text, do nearest-neighbour search, done. That
captures *similarity* but not *plasticity* — it doesn't model how a memory's value changes as it
is used, contradicted, consolidated, or forgotten.

Plasticity Agent treats memory as a living system with explicit signals — **salience,
confidence, usage, contradiction, decay, reward** — and a set of processes that act on them:
quality scoring, decay/forgetting, sleep-like consolidation, contradiction detection, and skill
mining. A vector backend is *optional* (and pluggable); the plasticity logic is the point.

Everything ships with a **deterministic baseline** so it runs fully offline with zero model
calls — then upgrades to LLM-powered behaviour the moment you pass a single `llm_callback`, and
to hybrid semantic retrieval the moment you pass `embeddings=`. Every LLM path falls back to the
deterministic one if a call or parse fails, so nothing ever hard-depends on a model.

## Feature matrix

| Capability | What it does | Deterministic | LLM/vector upgrade |
| --- | --- | :---: | :---: |
| **Neuroplastic memory** | 5 memory types with plasticity signals | ✅ | — |
| **Memory quality** | Utility scoring + keep/decay/consolidate/review/prune | ✅ | — |
| **Retrieval** | Lexical; hybrid+vector with a backend (semantic needs `sentence-transformers`) | ✅ | ✅ embeddings |
| **Contradiction detection** | Negation/antonym(stemmed)/sentiment/numeric heuristic — precision-first | ✅ | ✅ LLM |
| **Reflection (Reflexion)** | Store lessons from feedback | ✅ | ✅ LLM |
| **Self-Refine** | Critique + notes (deterministic) → full rewrite (LLM) | ✅ | ✅ LLM |
| **Sleep / consolidation** | Decay, dedup, gist, skill mining, constitution governance | ✅ | ✅ embeddings |
| **Self-healing** | Diagnose → advisory plan → **opt-in** sandboxed apply | ✅ | — |
| **Reasoning market** | 6 critics bid; auction selects winner | ✅ | ✅ LLM critic |
| **Skill library** | Successful traces → reusable skills | ✅ | — |
| **Energy report** | Entropy, contradiction, waste, temperature → plasticity score | ✅ | — |
| **Improvement metrics** | Checkpoint health; prove the agent got better | ✅ | — |
| **Observability** | Stream trace events to OpenTelemetry | ✅ | — |

---

## Installation

Requires **Python 3.11+**.

### With pip

```bash
pip install plasticity-agent
```

### With uv

```bash
uv add plasticity-agent
```

### Optional extras

The core install has everything you need for offline, deterministic use plus the FastAPI server
and Streamlit dashboard. Heavyweight/semantic backends are opt-in:

```bash
pip install "plasticity-agent[otel]"        # OpenTelemetry trace export
pip install "plasticity-agent[docs]"        # MkDocs site
pip install "plasticity-agent[dev]"         # pytest, ruff, mypy, numpy, otel

# Optional embedding/vector backends (installed on demand, not via an extra):
pip install sentence-transformers           # true semantic embeddings
pip install faiss-cpu                        # large-scale ANN index
pip install numpy                            # accelerates the built-in vector index
```

| Extra | Adds | For |
| --- | --- | --- |
| `otel` | `opentelemetry-api`, `opentelemetry-sdk` | Tracing/observability |
| `docs` | `mkdocs`, `mkdocs-material` | Building the docs site |
| `dev` | `pytest`, `ruff`, `mypy`, `numpy`, `opentelemetry-sdk` | Contributing |

### From source

```bash
git clone https://github.com/Kayariyan28/Plasticity-Agent.git
cd Plasticity-Agent
uv sync --all-extras       # or: pip install -e ".[dev]"
uv run pytest              # 122 tests
```

### Verify

```bash
python -c "from plasticity_agent import PlasticAgent; print('ok')"
plasticity --help
```

---

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

result = agent.run(
    "Read this paper, extract claims, critique methodology, and produce a reproducible summary."
)

agent.reflect(reward=0.8)
agent.sleep()
print(agent.energy_report())
```

With no `llm_callback` configured the runtime returns a **structured advisory response** (it still
recalls relevant memories and asks the reasoning market for a suggested action) and traces the
run. Wire a model in one line:

```python
def my_llm(prompt: str, **kwargs) -> str:
    ...  # call your LLM of choice; return text
    return "the model's answer"

agent = PlasticAgent(name="copilot", llm_callback=my_llm)   # reflection, contradiction,
                                                            # critics, self-refine all upgrade
```

`agent.run(...)` accepts three things:

- a **Python callable** → it's executed and traced;
- a **string + `llm_callback`** → the model is called;
- a **string + no callback** → a structured advisory plan is returned.

---

## Use cases & examples

> Every snippet below is runnable. The repo's [`examples/`](examples) folder has 12 complete,
> self-contained scripts (`uv run python examples/<name>.py`).

### 1. Neuroplastic memory

```python
from plasticity_agent import PlasticAgent

agent = PlasticAgent(name="demo", memory="./memory")
agent.remember("User prefers concise, well-cited answers.", "constitutional", tags=["user_preference"])
agent.remember("Cache warmup reduced p95 latency by 40%.", "semantic", tags=["important"], reward=0.9)
agent.remember("Tried retrying the flaky upload; it worked.", "episodic", reward=0.4)

for hit in agent.recall("latency cache"):
    print(hit.score, hit.memory.content, "·", hit.match_reason)
```

Memory types: `episodic`, `semantic`, `procedural`, `reflective`, `constitutional`. Each memory
carries `salience`, `confidence`, `usage_count`, `contradiction_score`, `decay_rate`, and
`reward`.

### 2. Memory quality scoring

```python
m = agent.remember("Cache warmup reduced p95 latency.", "semantic", tags=["important"], reward=0.9)
report = agent.memory.evaluate_memory(m)
print(report.utility_score, report.recommendation, report.reasons)
# 0.71 'consolidate' ['high recurrence/salience (...)']
```

```text
utility = 0.25*salience + 0.20*confidence + 0.15*usage + 0.15*reward
        + 0.10*recency - 0.20*contradiction - 0.10*decay          # clamped to [0, 1]
```

Recommendations: **keep · consolidate · review · decay · prune** (pruning is the only destructive
op and is always explicit; constitutional/important memories are protected).

### 3. Hybrid lexical + vector retrieval

Recall is lexical by default and becomes hybrid the moment you set an embedding backend — no other
code changes.

```python
# Zero-dependency hashing vectors, semantic transformers, or any embed callable:
agent = PlasticAgent(name="copilot", embeddings="hashing")
# agent = PlasticAgent(name="copilot", embeddings="st:all-MiniLM-L6-v2")   # pip install sentence-transformers
# agent = PlasticAgent(name="copilot", embeddings=lambda texts: my_embed(texts))

agent.remember("Cache warmup reduced p95 latency by 40%", "semantic")
hit = agent.recall("latency cache performance")[0]
print(hit.match_reason)   # "lexical overlap on: cache, latency · semantic cos=0.41"
```

Embeddings are persisted in SQLite; a NumPy-accelerated `VectorIndex` (with an optional FAISS
variant) narrows candidates for large corpora, and relevance is
`(1-α)·lexical + α·cosine`.

> **Honest note on backends.** The default `embeddings="hashing"` is a *structural* embedding
> (token feature-hashing): fast, deterministic, zero-dependency — but it only matches on shared
> tokens, so it will **not** retrieve pure synonyms/paraphrases (e.g. "make responses faster"
> won't match a memory about "latency"). For genuine semantic recall use
> `embeddings="st:all-MiniLM-L6-v2"` (`pip install sentence-transformers`) or your own embeddings
> callable. The hybrid scoring, persistence, and index are identical across backends.

### 4. LLM-powered upgrades

One callback upgrades reflection, contradiction detection, the reasoning market, and self-refine —
each with a deterministic fallback:

```python
agent = PlasticAgent(name="copilot", llm_callback=my_llm)

# or use the components directly:
from plasticity_agent import Reflector, ReasoningMarket, ContradictionChecker
Reflector(llm=my_llm).create_lesson(...)
ReasoningMarket(llm=my_llm).deliberate("...")          # adds an LLM critic to the panel
ContradictionChecker(llm=my_llm).pair("ship it", "do not ship it")   # semantic entailment
```

### 5. Reflection (Reflexion)

```python
lesson = agent.reflect(
    task="call the pricing tool",
    error="TypeError: missing 1 required positional argument: 'sku'",
    reward=-0.6,
)
print(lesson.lesson_type)   # 'tool_use'
# stored automatically as a reflective memory for next time
```

Lesson types: `success`, `failure`, `risk`, `preference`, `tool_use`, `reasoning`.

### 6. Self-Refine

```python
result = agent.refine("It will always work, probably.", rubric="accuracy, safety, completeness")
print(result.critique)            # flags vague phrasing, unsupported claims, ...
print(result.improvement_score)   # 0.46
print(result.refined_output)
```

> The **deterministic** refiner *critiques and appends actionable notes* — it does **not** rewrite
> the text. For an actual critique-and-rewrite, pass a model:
> `SelfRefine(llm_callback=lambda prompt, rubric: my_llm(prompt))` (or give the agent an
> `llm_callback`).

### 7. Sleep / consolidation

```python
report = agent.sleep()
print(report.summary)
# Analyzed N traces; decayed ..., merged ... duplicates, consolidated ...,
# found ... contradictions, ... constitution conflicts, created ... skills,
# suggested ... policies. Plasticity score XX/100.
```

Sleep decays weak memories, **de-duplicates**, consolidates reflective clusters into semantic
gist, mines procedural memories + skills from repeated successes, detects contradictions, governs
the **constitution**, and suggests advisory prompt policies. All counts are real.

### 8. Advisory & opt-in self-healing

Advisory by default — it diagnoses and recommends, never editing your files:

```python
try:
    import nonexistent_pkg
except Exception as error:
    plan = agent.heal(error)
    print(plan.diagnosis.failure_type)   # 'missing_dependency'
    for step in plan.steps:
        print("-", step)
```

Execution is **strictly opt-in**, allowlisted, no-shell, and timed:

```python
from plasticity_agent import RepairConsent

agent.apply_repair(error)                                                       # advisory: runs nothing
agent.apply_repair(error, RepairConsent(allow_apply=True, allow_install=True, dry_run=True))   # shows the command
agent.apply_repair(error, RepairConsent(allow_apply=True, allow_install=True, dry_run=False))  # installs the pkg
```

### 9. Critical reasoning market

```python
from plasticity_agent import ReasoningMarket

result = ReasoningMarket().deliberate(
    task="Choose best repair strategy for schema error",
    context={"error": "missing required argument"},
)
print(result.winner.critic_name, "->", result.winner.action)
for line in result.audit_trail:
    print(line)
```

Six critics — **Skeptic, Builder, Risk Analyst, Evidence Auditor, Game Theorist, Compression
Critic** (plus an LLM critic when configured) — each submit a scored `Proposal`; the auction ranks
by `0.25·truth + 0.20·reward + 0.15·confidence + 0.10·novelty + 0.10·reversibility − 0.10·risk − 0.10·cost`.

### 10. Skill library

```python
agent.skills.save("debug_import_error", {"steps": ["read traceback", "check pkg name", "uv add"]})
print(agent.skills.find("import error")[0].name)   # 'debug_import_error'
```

Skills are also mined automatically from repeated successful traces during `agent.sleep()`.

### 11. Energy report

```python
energy = agent.energy_report()
print(energy.memory_entropy, energy.contradiction_pressure, energy.token_waste)
print(energy.confidence_temperature)   # 'stable' | 'warm' | 'unstable'
print(energy.plasticity_score)         # 0..100
```

### 12. Cross-run improvement metrics

```python
agent.checkpoint("before")
# ... the agent learns, consolidates, resolves contradictions ...
agent.checkpoint("after")

report = agent.improvement()
print(report.improved, report.summary)
# True 'Over 2 checkpoints the agent improved (score +0.135): contradiction -0.328, grounded-utility +..., skills +0 ...'
```

> The verdict rewards **reduced contradiction**, **grounded utility** (utility weighted by how much
> each memory is actually *recalled*), and **learned skills** — it deliberately ignores raw
> salience/reward, so you **can't** fake improvement by storing high-reward memories that are never
> used.

### 13. OpenTelemetry export

```python
agent = PlasticAgent(name="copilot", otel=True)   # every trace event -> an OTel span
```

Requires `pip install "plasticity-agent[otel]"`. Local JSONL tracing keeps working regardless.

### 14. Framework adapters

Wrap any callable or framework agent so it gains tracing, memory, and advisory healing:

```python
from plasticity_agent.adapters import GenericAdapter, LangGraphAdapter

wrapped = GenericAdapter(agent).wrap_callable(my_function)
graph_runner = LangGraphAdapter(agent).wrap(my_compiled_graph)   # also CrewAI / OpenAI-Agents / Pydantic-AI
```

---

## Command-line interface

A beautiful Typer + Rich console:

```bash
plasticity init ./memory
plasticity remember "User prefers concise answers" --type constitutional --tag user_preference
plasticity recall "preferences"
plasticity evaluate
plasticity sleep ./agent_runs
plasticity report
plasticity heal "ModuleNotFoundError: No module named 'pandas'"
plasticity apply "ModuleNotFoundError: No module named 'rich'" --install            # dry run
plasticity apply "ModuleNotFoundError: No module named 'rich'" --install --execute  # run it
plasticity market "Choose a rollout strategy"
plasticity skills
plasticity metrics --checkpoint
plasticity export ./backup.jsonl
plasticity serve
plasticity dashboard
```

```text
$ plasticity sleep ./agent_runs
✓ 128 traces analyzed
✓ 41 weak memories decayed
✓ 17 memories consolidated
✓ 6 contradictions detected
✓ 4 reusable skills created
✓ 2 prompt policies improved
✓ agent plasticity score: 78/100
```

## FastAPI server

```bash
plasticity serve            # http://127.0.0.1:8000
```

Endpoints: `GET /health`, `GET /memories`, `POST /memories`, `POST /recall`, `POST /sleep`,
`POST /heal`, `POST /market`, `GET /report`, `GET /skills`.

```python
from plasticity_agent.server.api import build_app
app = build_app(memory_dir="./memory")   # mount in your own ASGI stack
```

## Streamlit dashboard

```bash
plasticity dashboard        # http://localhost:8501
```

Pages: Overview · Memories · Memory Quality · Sleep Reports · Failure Diagnostics ·
Reasoning Market · Skills · Energy Report.

## Configuration & storage

Local-first. State lives under one directory (default `./memory`):

```
memory/
├── plasticity.sqlite     # memories + embedding vectors + skills
├── metrics.sqlite        # improvement-metric snapshots
└── traces/
    └── YYYY-MM-DD.jsonl   # append-only trace events
```

```python
from plasticity_agent import PlasticityConfig, PlasticAgent

config = PlasticityConfig.from_memory_dir("./agent_state")
agent = PlasticAgent(name="copilot", config=config)
```

---

## Research foundation

These theories are used as **software metaphors and design principles**, not as claims of
biological equivalence or consciousness:

- **Complementary learning systems** — fast episodic memory + slower semantic consolidation.
- **Synaptic homeostasis** — sleep-like compression, forgetting, and strengthening.
- **Reflexion** (Shinn et al., 2023) — agents improve by storing lessons from feedback.
- **Self-Refine** (Madaan et al., 2023) — iterative self-critique and improvement.
- **Voyager** (Wang et al., 2023) — successful traces become a reusable skill library.
- **Game / auction theory** — critics bid; an auction selects the best next action.
- **Free-energy principle / thermodynamics** — reduce uncertainty, contradiction, and waste.

## Safety

- **No consciousness or biological claims.** This is engineering inspired by these ideas.
- **Self-healing is advisory by default; execution is strictly opt-in.** Only narrow, reversible
  repairs (currently `pip install <pkg>`) are eligible, and only with an explicit `RepairConsent`
  (`allow_apply=True`, the capability flag, and `dry_run=False`). The sandbox uses a hard
  allowlist, validated package names, no shell, and a timeout — and **never** edits, moves, or
  deletes your source files.
- **Local-first.** Nothing leaves your machine unless *you* wire an `llm_callback`, an embedding
  backend, or an OTel exporter that does.
- **Pruning is the only destructive memory op** and is always explicit; constitutional and
  important memories are protected.

## Testing & development

```bash
git clone https://github.com/Kayariyan28/Plasticity-Agent.git
cd Plasticity-Agent
uv sync --all-extras

uv run pytest            # 122 tests
uv run ruff check .      # lint
uv run mypy plasticity_agent   # types (clean)
uv build                 # sdist + wheel
```

## Roadmap

**Shipped in v0.2.0:** LLM-powered reflection / contradiction / critics / self-refine, pluggable
hybrid vector retrieval, opt-in sandboxed healing, cross-run improvement metrics, OpenTelemetry
export, concurrency-safe storage, and tested framework adapters.

**Next (v0.3.0):** native OTLP push exporter and metrics export, an inverted-index lexical
candidate stage for million-scale hybrid recall, async LLM batching, automatic embedding
backfill/migration, and a richer constitution policy engine (severity tiers + remediation).

## Contributing

Issues and PRs welcome. Please run `ruff`, `mypy`, and `pytest` before submitting:

```bash
uv run ruff check . && uv run mypy plasticity_agent && uv run pytest
```

## Citation

```bibtex
@software{plasticity_agent_2026,
  title   = {Plasticity Agent Runtime: Neuroplastic memory, self-healing, and critical reasoning for AI agents},
  author  = {Kayariyan28},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/Kayariyan28/Plasticity-Agent}
}
```

## License

[MIT](LICENSE) © 2026 Kayariyan28

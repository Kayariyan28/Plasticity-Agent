# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-06-07

### Added

- **LLM-powered upgrades** behind a single `llm_callback` (provider-agnostic, no SDK
  dependency): model-authored reflection, semantic contradiction/entailment scoring
  (`ContradictionChecker`), an `LLMCritic` in the reasoning market, and LLM self-refine. Every
  path falls back to the deterministic baseline on failure.
- **Hybrid lexical + vector retrieval**: pluggable `EmbeddingBackend`
  (`HashingEmbeddingBackend` with zero dependencies, optional `SentenceTransformerBackend`, and
  `CallableEmbeddingBackend`), embeddings persisted in SQLite, a NumPy-accelerated `VectorIndex`
  with an optional `FaissVectorIndex`, and `(1-α)·lexical + α·cosine` recall.
- **Opt-in sandboxed self-healing**: `Sandbox.apply(plan, RepairConsent(...))` can execute narrow,
  reversible repairs (allowlisted `pip install` of a validated package, no shell, timeout, dry-run
  default). `agent.apply_repair(...)` and `plasticity apply`.
- **Cross-run improvement metrics**: `ImprovementTracker`, `agent.checkpoint()` and
  `agent.improvement()`, persisted to `metrics.sqlite`, plus `plasticity metrics`.
- **OpenTelemetry export**: `OTelExporter` and `PlasticAgent(otel=True)` stream trace events as
  spans.
- **Richer consolidation**: embedding-aware clustering, near-duplicate merge (`duplicates_merged`),
  and constitutional-memory governance (`constitution_conflicts`).
- First-class, tested framework adapters (LangGraph / CrewAI / OpenAI-Agents / Pydantic-AI).
- `otel` optional-dependency extra.

### Changed

- Self-healing is no longer advisory-only: narrow, reversible repairs are now
  `auto_apply_allowed` but still require explicit consent to execute.
- SQLite stores are now concurrency-safe (WAL journaling, a guarding lock, retry-on-busy,
  `check_same_thread=False`) and safe to share across threads.
- `SleepReport` gained `duplicates_merged` and `constitution_conflicts` fields.

### Fixed

- Full `mypy` clean across all source files.

## [0.1.0] - 2026-06-06

### Added

- Initial release: neuroplastic memory (`MemoryOS`, 5 memory types, plasticity signals),
  deterministic memory-quality scoring, salience, heuristic contradiction detection, decay /
  forgetting, sleep-like consolidation (`SleepReport`), Reflexion-style reflection, Self-Refine,
  advisory self-healing (diagnosis + repair plans), a six-critic reasoning market with an auction,
  a Voyager-style skill library, thermodynamic-style energy reporting, the `PlasticAgent` runtime,
  JSONL tracing, a Typer + Rich CLI, a FastAPI server, a Streamlit dashboard, and framework
  adapters.

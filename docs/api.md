# API Reference

Everything below is importable from the top-level `plasticity_agent` package.

## `PlasticAgent`

```python
PlasticAgent(
    name: str,
    model: str | None = None,
    memory: str = "./memory",
    *,
    self_heal: bool = True,
    reasoning_market: bool = True,
    sleep_cycle: bool = True,
    llm_callback: Callable | None = None,
    embeddings: EmbeddingBackend | Callable | str | None = None,   # e.g. "hashing", "st:<model>"
    otel: bool | object = False,                                   # True or a custom exporter
    metrics: bool = True,
)
```

| Method | Returns | Description |
| --- | --- | --- |
| `run(task_or_callable, **kwargs)` | `RunResult` | Execute a callable, an LLM task, or an advisory plan |
| `remember(content, memory_type="episodic", ...)` | `Memory` | Record a memory |
| `recall(query, limit=5)` | `list[MemorySearchResult]` | Recall (hybrid if embeddings on; counts as use) |
| `reflect(task=None, output=None, error=None, reward=0.0, evaluator_feedback=None)` | `Lesson` | Create + store a lesson (LLM-authored if configured) |
| `refine(output, rubric="accuracy, safety, completeness")` | `SelfRefineResult` | Self-refine an output |
| `heal(error)` | `RepairPlan` | Advisory repair plan |
| `apply_repair(error_or_plan, consent=None)` | `SandboxResult` | Opt-in sandboxed repair (advisory unless consent given) |
| `deliberate(task, context=None)` | `AuctionResult` | Run the reasoning market |
| `sleep()` | `SleepReport` | Run a consolidation cycle |
| `energy_report()` | `EnergyReport` | Thermodynamic-style report |
| `checkpoint(label="checkpoint")` | `MetricSnapshot` | Record a metrics snapshot |
| `improvement()` | `ImprovementReport` | Did the agent get better across checkpoints? |
| `report()` | `dict` | High-level status snapshot |
| `export(path=None)` | `Path` | Export memories to JSONL |

## `MemoryOS`

`record`, `recall`, `search`, `list_memories`, `evaluate_memory`, `evaluate_all`, `decay`,
`consolidate`, `sleep`, `export_jsonl`, `import_jsonl`, `prune`.

## Memory schemas

- `Memory(id, content, memory_type, tags, salience, confidence, usage_count, contradiction_score, decay_rate, reward, source_trace, created_at, updated_at, metadata)`
- `MemoryQualityReport(memory_id, utility_score, salience, confidence, contradiction_score, decay_rate, usage_count, recommendation, reasons)`
- `MemorySearchResult(memory, score, match_reason)`

## Scoring functions

- `calculate_salience(content, reward=0.0, tags=None, confidence=0.7, recurrence=1) -> float`
- `detect_contradiction(new_memory, existing_memories) -> float`
- `compute_utility_score(memory) -> float`
- `score_memory_quality(memory) -> MemoryQualityReport`

## Reflection

- `Reflector().create_lesson(ReflectionInput(...)) -> Lesson`
- `Lesson(content, lesson_type, confidence, reward, tags)`
- `SelfRefine(llm_callback=None).refine(output, rubric) -> SelfRefineResult`

## Healing

- `diagnose(error) -> FailureDiagnosis`
- `plan_repair(diagnosis) -> RepairPlan`
- `heal(error) -> RepairPlan`
- `Sandbox().evaluate(plan) -> SandboxResult` — non-executing review
- `Sandbox().apply(plan, consent) -> SandboxResult` — opt-in, gated execution
- `RepairConsent(allow_apply=False, allow_install=False, allowed_types=[...], dry_run=True, timeout=120)`
- `detect_failures(trace_records) -> list[FailureDiagnosis]`

## Retrieval & embeddings

- `get_embedder(spec) -> EmbeddingBackend | None` — `"hashing"`, `"hashing:512"`, `"st:<model>"`, callable, or backend
- `HashingEmbeddingBackend(dim=256, ngram=2)`, `SentenceTransformerBackend(model)`, `CallableEmbeddingBackend(fn)`
- `VectorIndex(store, embedder)`: `search(query_vector, k) -> list[(id, cosine)]`, `embed_text(text)`
- `cosine_similarity(a, b) -> float`
- `search_memories(query, memories, *, query_vector=None, vector_of=None, alpha=0.5, backend=None)`

## LLM layer

- `coerce_llm(llm) -> LLMClient | None` (accepts a callable or an `LLMClient`)
- `Reflector(llm=...)`, `ReasoningMarket(llm=...)`, `Debate(llm=...)`, `ContradictionChecker(llm=...)`, `SelfRefine(llm_callback=...)`

## Metrics & observability

- `ImprovementTracker(db_path)`: `record(snapshot)`, `history()`, `report() -> ImprovementReport`, `to_jsonl(path)`
- `MetricSnapshot(...)`, `ImprovementReport(...)`
- `OTelExporter(service_name="plasticity-agent", provider=None)` — attach to a `Tracer` or `PlasticAgent(otel=...)`

## Reasoning

- `ReasoningMarket(critics=None).deliberate(task, context=None) -> AuctionResult`
- `ReasoningMarket().auction(proposals) -> AuctionResult`
- `run_auction(proposals) -> AuctionResult`
- `score_proposal(proposal) -> float`
- `Debate(critics=None).run(task, context=None, rounds=1) -> DebateResult`

## Learning

- `SkillLibrary(db_path)`: `save`, `find`, `list_skills`, `use`, `promote_from_trace`
- `Skill(id, name, description, trigger_patterns, successful_trace, usage_count, confidence, reward, created_at, updated_at)`
- `shape_reward(...)`, `normalize_reward(reward)`
- `Curriculum().propose(skills, memories) -> list[CurriculumItem]`

## Thermodynamics

- `build_energy_report(memories, trace_records=None) -> EnergyReport`
- `EnergyReport(memory_entropy, contradiction_pressure, token_waste, repair_energy, confidence_temperature, plasticity_score, summary)`

## Server

- `from plasticity_agent.server.api import build_app, serve`

## CLI

`plasticity {init, remember, recall, evaluate, sleep, report, heal, market, skills, export, serve, dashboard, version}`

# Concepts

## Scientific positioning

Plasticity Agent borrows ideas from neuroscience and learning theory as **software metaphors and
design principles**. It does **not** claim biological equivalence or consciousness. The value is
engineering: explicit signals and processes that make an agent's behaviour improve over time.

!!! warning "Safety"
    - Self-healing is **advisory only** in v0.1.0. `auto_apply_allowed` is always `False`, and the
      sandbox never executes commands, installs packages, or edits files.
    - Memory is **local-first**: a single SQLite file plus JSONL traces. Nothing leaves your
      machine unless you wire an `llm_callback` that sends it somewhere.
    - The only destructive memory operation is `prune()`, and it is explicit by design;
      constitutional and important memories are protected.

## The theories → components map

| Theory | Used as | Component |
| --- | --- | --- |
| Complementary learning systems | Fast episodic memory + slow semantic consolidation | `MemoryOS`, sleep cycle |
| Synaptic homeostasis | Sleep-like compression, forgetting, strengthening | `forgetting`, `consolidation` |
| Reflexion | Store explicit lessons from feedback | `Reflector`, reflective memories |
| Self-Refine | Iterative critique → improved output | `SelfRefine` |
| Voyager skill library | Successful traces become reusable skills | `SkillLibrary` |
| Game / auction theory | Critics bid; an auction selects the action | `ReasoningMarket`, `run_auction` |
| Free-energy principle | Reduce uncertainty, contradiction, waste | `energy_report`, `free_energy` |

## Why memory is more than vector search

A vector store answers *"what is similar to this?"*. Plasticity treats memory as a system with
signals — salience, confidence, usage, contradiction, decay, reward — and processes that act on
them: quality scoring, decay, consolidation, contradiction detection, and skill mining. Vector
retrieval is an *optional future backend* (the `RetrievalBackend` protocol); the plasticity logic
is the point.

## Deterministic by default, LLM-powered on demand

Everything has a deterministic baseline so the framework is testable and runs fully offline:

- Salience, contradiction, and quality scoring are formula-based.
- Reflection classifies outcomes with transparent rules.
- Self-Refine and the debate run rule-based critiques.
- Critics produce scored proposals from heuristics.
- Retrieval is lexical.

As of v0.2.0 each of these has a **working LLM/vector upgrade** that you opt into with a single
`llm_callback` or `embeddings=` argument — and which always falls back to the deterministic path
if a call fails. See [LLM Features](llm.md) and [Retrieval](retrieval.md).

## Safety, upgraded but still conservative

Self-healing can now *apply* repairs, but only the narrow, reversible kind (e.g. installing a
missing package), only with an explicit `RepairConsent`, behind a hard allowlist with no shell and
a timeout — and it never edits your source files. See [Self-Healing](self_healing.md).

# Memory

## The `Memory` model

Every memory carries content plus *plasticity signals*:

| Field | Meaning |
| --- | --- |
| `memory_type` | `episodic`, `semantic`, `procedural`, `reflective`, `constitutional` |
| `salience` | How much the memory should grab attention later (0–1) |
| `confidence` | How much it is trusted (0–1) |
| `usage_count` | How often it has been recalled |
| `contradiction_score` | How much it conflicts with other memories (0–1) |
| `decay_rate` | Forgetting pressure accumulated over time (0–1) |
| `reward` | Outcome signal associated with the memory |

`salience`, `confidence`, `contradiction_score`, and `decay_rate` are clamped to `[0, 1]`.

## Recording and recall

```python
from plasticity_agent.memory.memory_os import MemoryOS

memory = MemoryOS(memory_dir="./memory")
memory.record("Cache warmup cut p95 latency by 40%", "semantic", tags=["important"], reward=0.9)

for hit in memory.recall("latency cache"):
    print(hit.score, hit.memory.content, hit.match_reason)
```

- `record(...)` auto-scores salience (unless given) and contradiction against existing memories.
- `recall(...)` ranks by lexical similarity + tag match + a salience prior, and counts as *use*
  (it increments `usage_count` and emits a trace).
- `search(...)` is the same ranking with **no** side effects.

Retrieval is deterministic lexical similarity (Jaccard + overlap coefficient). No vector DB is
required in v0.1.0; `RetrievalBackend` is the hook for adding one later.

## Quality scoring

`evaluate_memory` / `evaluate_all` return a `MemoryQualityReport`. The utility score is:

```text
utility =  0.25*salience + 0.20*confidence + 0.15*normalized_usage
         + 0.15*positive_reward + 0.10*recency
         - 0.20*contradiction_score - 0.10*decay_rate         # clamped to [0, 1]
```

Recommendations:

- **keep** — healthy memory.
- **consolidate** — high recurrence (`usage_count ≥ 4`) or high salience (`≥ 0.8`).
- **review** — high contradiction (`≥ 0.6`).
- **decay** — low utility.
- **prune** — very low utility (and only ever deleted on an explicit `prune()` call).

## Salience

```python
from plasticity_agent.memory.salience import calculate_salience
calculate_salience("Always validate inputs", reward=0.9, tags=["critical"], recurrence=3)
```

Salience blends reward magnitude (success *and* failure are salient), confidence, recurrence, and
explicit importance/failure tags.

## Contradiction detection

```python
from plasticity_agent.memory.contradiction import detect_contradiction
detect_contradiction("I love coffee", ["I hate coffee"])   # ~0.6
```

A contradiction needs both topical relatedness **and** semantic opposition (negation mismatch,
antonyms, or a sentiment flip). It is a deterministic baseline, documented as such.

## Decay and forgetting

```python
memory.decay(days_passed=30)
```

Stale, rarely-used, unprotected memories lose salience and gain decay pressure. Constitutional and
important-tagged memories are protected. Decay never deletes — it only weakens and flags prune
candidates.

## Import / export

```python
path = memory.export_jsonl("backup.jsonl")
memory.import_jsonl(path)
```

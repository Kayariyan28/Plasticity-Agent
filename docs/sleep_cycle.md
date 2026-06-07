# Sleep Cycle

Sleep is a software analogue of complementary learning systems plus synaptic homeostasis. During
a cycle the agent replays recent experience and:

1. **Decays** stale, rarely-used memories (forgetting curve).
2. **Consolidates** clusters of related reflective memories into durable **semantic** gist.
3. **Mines procedural memories and skills** from repeated successful run traces.
4. **Detects contradictions** and updates each memory's contradiction score.
5. **Suggests advisory prompt policies** from recurring failures.

```python
report = agent.sleep()
print(report.summary)
```

## `SleepReport`

```python
SleepReport(
    traces_analyzed: int,
    weak_memories_decayed: int,
    memories_consolidated: int,
    contradictions_detected: int,
    skills_created: int,
    policies_improved: int,
    plasticity_score: float,   # 0–100
    summary: str,
)
```

All counts are **real** — they reflect work actually performed on your data. Small datasets
produce small numbers.

## CLI

```bash
plasticity sleep ./agent_runs
```

```text
✓ 128 traces analyzed
✓ 41 weak memories decayed
✓ 17 memories consolidated
✓ 6 contradictions detected
✓ 4 reusable skills created
✓ 2 prompt policies improved
✓ agent plasticity score: 78/100
```

Pass a directory of JSONL run logs and sleep will analyse those traces (it tolerates foreign
schemas — malformed lines are skipped). With no path, it analyses the agent's own
`./memory/traces/`.

## `consolidate()` vs `sleep()`

- `memory.consolidate()` runs only the consolidation step (reflective → semantic, successful
  traces → procedural/skills).
- `memory.sleep()` runs the full cycle (decay + consolidation + contradiction + policies) and
  emits a `sleep_completed` trace.

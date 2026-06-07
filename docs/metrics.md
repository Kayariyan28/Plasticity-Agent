# Cross-Run Improvement Metrics

"Is the agent actually getting better?" Plasticity answers it with persisted **metric
snapshots** and a signed **improvement score**.

```python
agent.checkpoint("before")
# ... the agent runs, learns, consolidates, resolves contradictions ...
agent.checkpoint("after")

report = agent.improvement()
print(report.improved)        # True / False
print(report.summary)
```

## What a snapshot captures

`MetricSnapshot`: `plasticity_score`, `avg_utility`, `contradiction_pressure`, `memory_entropy`,
plus memory and skill counts, timestamped.

## The verdict

`ImprovementReport` compares the first and latest snapshots:

```text
improvement_score = 0.4 * (Δplasticity / 100)
                  + 0.3 * Δavg_utility
                  + 0.3 * (-Δcontradiction_pressure)      # less contradiction is better
improved = improvement_score > 0
```

It also reports each delta (plasticity, utility, contradiction, entropy, skills) and a one-line
summary. Snapshots persist to `./memory/metrics.sqlite`, so improvement is measured across
process restarts.

## CLI

```bash
plasticity metrics --checkpoint     # record a checkpoint, then report
plasticity metrics                  # report only
```

## Direct use

```python
from plasticity_agent import ImprovementTracker, MetricSnapshot

tracker = ImprovementTracker("./memory/metrics.sqlite")
tracker.record(MetricSnapshot(plasticity_score=70, avg_utility=0.6, contradiction_pressure=0.2))
report = tracker.report()
tracker.to_jsonl("metrics_history.jsonl")
```

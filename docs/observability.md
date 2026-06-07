# Observability (OpenTelemetry)

Every meaningful action is already written to local JSONL traces. To also stream them to your
observability backend, attach an OpenTelemetry exporter — each `TraceEvent` becomes a span with
`plasticity.*` attributes.

```python
agent = PlasticAgent(name="copilot", otel=True)   # uses a default TracerProvider
```

`opentelemetry-sdk` is an optional dependency (`pip install "plasticity-agent[otel]"`). Local
JSONL tracing keeps working regardless, and exporter failures never break tracing.

## Wiring your own provider

```python
from opentelemetry.sdk.trace import TracerProvider
from plasticity_agent.observability.otel import OTelExporter
from plasticity_agent.core.trace import Tracer

provider = TracerProvider()           # add your OTLP span processor/exporter here
tracer = Tracer("./memory/traces", exporter=OTelExporter(provider=provider))
agent = PlasticAgent(name="copilot", otel=OTelExporter(provider=provider))
```

You can pass any object with an `export(event)` method as `otel=...` to integrate a custom sink.

## What gets exported

Span name = the event type (`run_started`, `memory_recorded`, `reasoning_auction`,
`sleep_completed`, `healing_diagnosed`, `energy_report_created`, …). Attributes include
`plasticity.run_id`, `plasticity.event_id`, `plasticity.timestamp`, and the scalar fields of the
event payload (e.g. `plasticity.task`, `plasticity.failure_type`).

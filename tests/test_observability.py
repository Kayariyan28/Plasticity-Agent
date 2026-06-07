"""Tests for the OpenTelemetry trace exporter (skipped if otel not installed)."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from plasticity_agent.core.events import RUN_STARTED  # noqa: E402
from plasticity_agent.core.trace import Tracer  # noqa: E402
from plasticity_agent.observability.otel import OTelExporter  # noqa: E402


def _provider_with_memory() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    return provider, memory_exporter


def test_otel_exporter_emits_spans_with_attributes(tmp_path) -> None:
    provider, memory_exporter = _provider_with_memory()
    tracer = Tracer(str(tmp_path / "traces"), exporter=OTelExporter(provider=provider))

    tracer.emit(RUN_STARTED, {"task": "do the thing"})

    spans = memory_exporter.get_finished_spans()
    assert any(span.name == RUN_STARTED for span in spans)
    span = next(s for s in spans if s.name == RUN_STARTED)
    assert span.attributes.get("plasticity.task") == "do the thing"
    assert span.attributes.get("plasticity.run_id")


def test_tracer_still_writes_jsonl_when_exporter_present(tmp_path) -> None:
    provider, _ = _provider_with_memory()
    traces_dir = tmp_path / "traces"
    tracer = Tracer(str(traces_dir), exporter=OTelExporter(provider=provider))
    tracer.emit(RUN_STARTED, {"task": "x"})
    assert list(traces_dir.glob("*.jsonl"))

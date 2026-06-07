"""OpenTelemetry export of Plasticity trace events.

Converts each :class:`~plasticity_agent.core.events.TraceEvent` into an
OpenTelemetry span with ``plasticity.*`` attributes. ``opentelemetry-sdk`` is an
optional dependency; :class:`OTelExporter` raises a clear error if it is absent.

Attach an exporter to a :class:`~plasticity_agent.core.trace.Tracer` (or pass
``otel=True`` to :class:`~plasticity_agent.core.agent.PlasticAgent`) to stream
agent activity into any OTLP-compatible backend.
"""

from __future__ import annotations

from typing import Any

from plasticity_agent.core.events import TraceEvent


class TraceExporter:
    """Structural type: anything with ``export(event) -> None``."""

    def export(self, event: TraceEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class OTelExporter:
    """Exports trace events as OpenTelemetry spans."""

    def __init__(self, service_name: str = "plasticity-agent", provider: Any | None = None) -> None:
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OTelExporter requires the optional 'opentelemetry-sdk' package. "
                "Install it with `pip install opentelemetry-sdk` (or the 'otel' extra)."
            ) from exc
        if provider is None:
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        self.provider = provider
        self._tracer = provider.get_tracer("plasticity_agent")

    def export(self, event: TraceEvent) -> None:
        span = self._tracer.start_span(event.event_type)
        try:
            span.set_attribute("plasticity.run_id", event.run_id)
            span.set_attribute("plasticity.event_id", event.id)
            span.set_attribute("plasticity.timestamp", event.timestamp.isoformat())
            for key, value in (event.payload or {}).items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"plasticity.{key}", value)
        finally:
            span.end()

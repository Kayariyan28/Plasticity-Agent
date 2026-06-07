"""JSONL tracing.

Traces are append-only newline-delimited JSON, one file per UTC day under
``<memory>/traces/YYYY-MM-DD.jsonl``. The :class:`Tracer` writes them; the
module-level readers load them back (tolerantly, so externally produced run
logs can also be analysed by the sleep cycle).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from plasticity_agent.core.events import TraceEvent, new_run_id


@runtime_checkable
class TraceExporter(Protocol):
    """Anything that can forward a trace event (e.g. to OpenTelemetry)."""

    def export(self, event: TraceEvent) -> None: ...


class Tracer:
    """Writes :class:`TraceEvent` records to daily JSONL files.

    An optional ``exporter`` (e.g. ``OTelExporter``) additionally forwards each
    event to an external backend. Exporter failures never break local tracing.
    """

    def __init__(
        self,
        traces_dir: str | Path,
        run_id: str | None = None,
        *,
        exporter: TraceExporter | None = None,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self.exporter = exporter

    def _today_path(self) -> Path:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.traces_dir / f"{day}.jsonl"

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> TraceEvent:
        """Create, persist, and return a trace event."""

        event = TraceEvent(
            run_id=run_id or self.run_id,
            event_type=event_type,
            payload=payload or {},
        )
        line = event.model_dump_json()
        with self._today_path().open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.exporter is not None:
            try:
                self.exporter.export(event)
            except Exception:  # noqa: BLE001 - external export must not break tracing
                pass
        return event

    def read_events(self) -> list[TraceEvent]:
        """Read every event this tracer's directory contains."""

        return read_trace_events(self.traces_dir)


def _iter_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    return []


def load_trace_records(path: str | Path) -> list[dict[str, Any]]:
    """Load raw trace dicts from a file or directory of JSONL logs.

    Tolerant by design: malformed lines are skipped rather than raising, so a
    folder of mixed agent run logs can still be analysed.
    """

    records: list[dict[str, Any]] = []
    for file in _iter_jsonl_files(Path(path)):
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def read_trace_events(path: str | Path) -> list[TraceEvent]:
    """Load trace records and coerce the ones that look like events."""

    events: list[TraceEvent] = []
    for record in load_trace_records(path):
        if "event_type" not in record:
            continue
        record.setdefault("run_id", "unknown")
        try:
            events.append(TraceEvent.model_validate(record))
        except Exception:  # noqa: BLE001 - tolerate foreign schemas
            continue
    return events

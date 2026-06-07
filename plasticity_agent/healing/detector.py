"""Failure detection.

Surfaces failures from live exceptions or from recorded traces (``run_failed``
events), each turned into a :class:`FailureDiagnosis`. Detection is read-only.
"""

from __future__ import annotations

from typing import Any

from plasticity_agent.core.events import RUN_FAILED
from plasticity_agent.healing.diagnosis import FailureDiagnosis, diagnose


class FailureDetector:
    """Detects and diagnoses failures from exceptions or trace records."""

    def from_exception(self, error: object) -> FailureDiagnosis:
        return diagnose(error)

    def scan_traces(self, trace_records: list[dict[str, Any]]) -> list[FailureDiagnosis]:
        """Diagnose every ``run_failed`` event found in ``trace_records``."""

        diagnoses: list[FailureDiagnosis] = []
        for record in trace_records:
            if record.get("event_type") != RUN_FAILED:
                continue
            payload = record.get("payload", {}) or {}
            error_text = payload.get("error") or payload.get("message") or ""
            if error_text:
                diagnoses.append(diagnose(error_text))
        return diagnoses


def detect_failures(trace_records: list[dict[str, Any]]) -> list[FailureDiagnosis]:
    """Module-level convenience over :meth:`FailureDetector.scan_traces`."""

    return FailureDetector().scan_traces(trace_records)

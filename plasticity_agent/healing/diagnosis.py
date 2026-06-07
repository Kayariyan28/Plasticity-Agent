"""Failure diagnosis.

Classifies an error (an ``Exception`` or a raw message string) into a typed
:class:`FailureDiagnosis` using transparent rules. No code is executed and no
files are touched — diagnosis only inspects text and exception types.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

FailureType = Literal[
    "tool_schema_error",
    "missing_dependency",
    "timeout",
    "import_error",
    "type_error",
    "value_error",
    "permission_error",
    "file_not_found",
    "unknown",
]


class FailureDiagnosis(BaseModel):
    """A typed, evidence-backed read on what went wrong."""

    failure_type: FailureType
    root_cause: str
    repair_strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def _normalize(error: object) -> tuple[str, str]:
    if isinstance(error, BaseException):
        return type(error).__name__, str(error)
    return "", str(error)


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _extract(pattern: str, message: str, default: str) -> str:
    match = re.search(pattern, message)
    return match.group(1) if match else default


def diagnose(error: object) -> FailureDiagnosis:
    """Classify ``error`` into a :class:`FailureDiagnosis`."""

    type_name, message = _normalize(error)
    text = f"{type_name} {message}".lower()
    evidence = [f"type={type_name or 'unknown'}", f"message={_short(message)}"]

    if type_name == "ModuleNotFoundError" or "no module named" in text:
        pkg = _extract(r"no module named ['\"]?([\w\.\-]+)", message.lower(), "the package")
        return FailureDiagnosis(
            failure_type="missing_dependency",
            root_cause=f"Required module '{pkg}' is not installed in this environment.",
            repair_strategy=f"Install it: `uv add {pkg}` (or `pip install {pkg}`), then re-run.",
            confidence=0.9,
            evidence=evidence,
            details={"package": pkg},
        )

    if type_name == "ImportError" or "cannot import name" in text or "importerror" in text:
        name = _extract(r"cannot import name ['\"]?([\w]+)", message, "the symbol")
        return FailureDiagnosis(
            failure_type="import_error",
            root_cause=f"Import failed for {name} — likely a version or API mismatch.",
            repair_strategy=(
                "Check the installed version and the symbol's import path; "
                "pin a compatible version."
            ),
            confidence=0.8,
            evidence=evidence,
        )

    if type_name == "TypeError" or "typeerror" in text:
        if any(token in text for token in ("argument", "positional", "keyword", "missing")):
            arg = _extract(r"argument:? ['\"]?([\w]+)", message, "a required argument")
            return FailureDiagnosis(
                failure_type="tool_schema_error",
                root_cause=f"Call did not match the expected signature ({arg}).",
                repair_strategy=(
                    "Align the call to the tool/function schema: "
                    "fix argument names, order, and types."
                ),
                confidence=0.85,
                evidence=evidence,
                details={"argument": arg},
            )
        return FailureDiagnosis(
            failure_type="type_error",
            root_cause="An operation received a value of the wrong type.",
            repair_strategy=(
                "Add a type check/coercion at the boundary before the failing operation."
            ),
            confidence=0.7,
            evidence=evidence,
        )

    if type_name == "TimeoutError" or "timeout" in text or "timed out" in text:
        return FailureDiagnosis(
            failure_type="timeout",
            root_cause="An operation exceeded its time budget.",
            repair_strategy="Add retry with exponential backoff and a sensible timeout ceiling.",
            confidence=0.8,
            evidence=evidence,
        )

    if type_name == "FileNotFoundError" or "no such file" in text:
        path = _extract(r"['\"]([^'\"]+)['\"]", message, "the path")
        return FailureDiagnosis(
            failure_type="file_not_found",
            root_cause=f"A required file or directory was not found ({path}).",
            repair_strategy=(
                "Verify the path exists and is created/downloaded before use; "
                "check the working directory."
            ),
            confidence=0.85,
            evidence=evidence,
            details={"path": path},
        )

    if type_name == "PermissionError" or "permission denied" in text:
        return FailureDiagnosis(
            failure_type="permission_error",
            root_cause="The process lacks permission for the requested resource.",
            repair_strategy=(
                "Review file/socket permissions and run with appropriate (least) privileges."
            ),
            confidence=0.8,
            evidence=evidence,
        )

    if type_name == "ValueError" or "valueerror" in text:
        return FailureDiagnosis(
            failure_type="value_error",
            root_cause="A value was out of range or malformed for the operation.",
            repair_strategy="Validate and normalize inputs (range, format) before the operation.",
            confidence=0.7,
            evidence=evidence,
        )

    return FailureDiagnosis(
        failure_type="unknown",
        root_cause="Could not classify the failure from the available evidence.",
        repair_strategy="Capture a fuller stack trace and reproduce with verbose logging.",
        confidence=0.35,
        evidence=evidence,
    )

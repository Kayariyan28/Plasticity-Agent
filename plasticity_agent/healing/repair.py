"""Repair planning.

Given a :class:`FailureDiagnosis`, produce a :class:`RepairPlan`: concrete steps
a human — or, with **explicit opt-in consent**, the sandbox executor — can take.

Most repair types stay advisory (``auto_apply_allowed=False``). Only narrowly
safe, reversible repairs (currently ``missing_dependency`` → ``pip install``)
are marked ``auto_apply_allowed=True``. Even then nothing runs unless the caller
passes an explicit :class:`~plasticity_agent.healing.sandbox.RepairConsent` to
the sandbox. The framework never edits user source files.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from plasticity_agent.healing.diagnosis import FailureDiagnosis, FailureType, diagnose

RiskLevel = Literal["low", "medium", "high"]

_STEPS: dict[FailureType, list[str]] = {
    "missing_dependency": [
        "Confirm the exact distribution name (import name may differ).",
        "Add the dependency: `uv add <package>` or `pip install <package>`.",
        "Record it in pyproject.toml so the environment is reproducible.",
        "Re-run the failing step to confirm resolution.",
    ],
    "import_error": [
        "Check the installed version of the package.",
        "Verify the symbol's import path against the current API.",
        "Pin a known-compatible version and reinstall.",
    ],
    "tool_schema_error": [
        "Inspect the tool/function signature and required arguments.",
        "Fix argument names, ordering, and types at the call site.",
        "Add schema validation before invoking the tool.",
    ],
    "timeout": [
        "Wrap the operation in retry-with-backoff (e.g., 3 attempts).",
        "Set an explicit, generous timeout ceiling.",
        "Consider chunking the work or adding a progress checkpoint.",
    ],
    "file_not_found": [
        "Print the resolved absolute path and current working directory.",
        "Ensure the file is created/downloaded before it is read.",
        "Guard the read with an existence check and a clear error.",
    ],
    "permission_error": [
        "Inspect the resource's ownership and permission bits.",
        "Run with least-privilege credentials that include the needed access.",
        "Avoid writing to protected locations; use a user-writable directory.",
    ],
    "type_error": [
        "Log the actual type received at the failing boundary.",
        "Add a coercion/validation step before the operation.",
    ],
    "value_error": [
        "Validate the input's range/format before use.",
        "Add a defensive default or a clear, early error message.",
    ],
    "unknown": [
        "Capture a full stack trace with verbose logging.",
        "Reproduce in isolation with a minimal input.",
        "File the trace for human review.",
    ],
}

_RISK: dict[FailureType, RiskLevel] = {
    "missing_dependency": "medium",
    "import_error": "medium",
    "tool_schema_error": "low",
    "timeout": "low",
    "file_not_found": "low",
    "permission_error": "high",
    "type_error": "low",
    "value_error": "low",
    "unknown": "high",
}


# Repair types that are narrow + reversible enough to be applied with consent.
_AUTO_APPLICABLE: set[FailureType] = {"missing_dependency"}


class RepairPlan(BaseModel):
    """A repair plan derived from a diagnosis.

    ``auto_apply_allowed`` indicates the repair is *eligible* for sandboxed
    execution; it never executes without an explicit consent object. Everything
    else is ``advisory_only``.
    """

    diagnosis: FailureDiagnosis
    steps: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "medium"
    auto_apply_allowed: bool = False
    advisory_only: bool = True


def plan_repair(diagnosis: FailureDiagnosis) -> RepairPlan:
    """Build a :class:`RepairPlan` from a diagnosis."""

    failure_type = diagnosis.failure_type
    auto_applicable = failure_type in _AUTO_APPLICABLE
    return RepairPlan(
        diagnosis=diagnosis,
        steps=list(_STEPS.get(failure_type, _STEPS["unknown"])),
        risk_level=_RISK.get(failure_type, "medium"),
        auto_apply_allowed=auto_applicable,
        advisory_only=not auto_applicable,
    )


def heal(error: object) -> RepairPlan:
    """Diagnose ``error`` and return an advisory repair plan."""

    return plan_repair(diagnose(error))

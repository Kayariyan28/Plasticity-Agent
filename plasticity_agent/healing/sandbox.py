"""Sandboxed repair execution — opt-in, default-off, tightly constrained.

The sandbox can *apply* a repair, but only through several explicit gates:

1. The plan must be ``auto_apply_allowed`` (only narrow, reversible repairs are).
2. The caller must pass a :class:`RepairConsent` with ``allow_apply=True`` and
   the relevant capability flag (e.g. ``allow_install=True``).
3. The failure type must be in ``consent.allowed_types``.
4. ``dry_run`` defaults to ``True`` — you must explicitly turn it off to execute.

Execution is limited to a hard allowlist (currently: ``pip install`` of a
**validated** package name) run via ``subprocess`` with no shell, a timeout, and
captured output. The sandbox never edits, deletes, or moves user files.
"""

from __future__ import annotations

import re
import subprocess
import sys

from pydantic import BaseModel, Field

from plasticity_agent.healing.repair import RepairPlan

# Conservative PEP 508-ish distribution name (no spaces, flags, or shell metachars).
_SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_PLACEHOLDER_PACKAGES = {"the package", ""}
_MAX_CAPTURE = 4000


class RepairConsent(BaseModel):
    """Explicit, opt-in authorisation to apply repairs. All gates default off."""

    allow_apply: bool = False
    allow_install: bool = False
    allowed_types: list[str] = Field(default_factory=lambda: ["missing_dependency"])
    dry_run: bool = True
    timeout: int = 120


class SandboxResult(BaseModel):
    """The result of a sandbox review or (gated) execution."""

    applied: bool = False
    advisory_only: bool = True
    dry_run: bool = True
    risk_level: str = "medium"
    command: list[str] | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    notes: list[str] = Field(default_factory=list)


class Sandbox:
    """Reviews repair plans and, only under explicit consent, applies safe ones."""

    def evaluate(self, plan: RepairPlan) -> SandboxResult:
        """Non-executing review: report what would be suggested."""

        notes = [
            "Sandbox review (no commands executed).",
            f"Diagnosis: {plan.diagnosis.failure_type} — {plan.diagnosis.root_cause}",
            *[f"Would suggest: {step}" for step in plan.steps],
        ]
        if plan.risk_level == "high":
            notes.append("High-risk plan: explicit human approval strongly recommended.")
        return SandboxResult(
            applied=False,
            advisory_only=True,
            dry_run=True,
            risk_level=plan.risk_level,
            notes=notes,
        )

    def _safe_command(self, plan: RepairPlan, consent: RepairConsent) -> list[str] | None:
        diagnosis = plan.diagnosis
        if diagnosis.failure_type == "missing_dependency" and consent.allow_install:
            package = str(diagnosis.details.get("package", "")).strip()
            if package not in _PLACEHOLDER_PACKAGES and _SAFE_PACKAGE_RE.match(package):
                return [sys.executable, "-m", "pip", "install", package]
        return None

    def apply(self, plan: RepairPlan, consent: RepairConsent | None = None) -> SandboxResult:
        """Apply a repair if (and only if) every consent gate is satisfied."""

        consent = consent or RepairConsent()
        risk = plan.risk_level

        gated = (
            consent.allow_apply
            and plan.auto_apply_allowed
            and plan.diagnosis.failure_type in set(consent.allowed_types)
        )
        if not gated:
            return SandboxResult(
                applied=False,
                advisory_only=True,
                dry_run=consent.dry_run,
                risk_level=risk,
                notes=[
                    "Not applied: requires allow_apply, an auto-applicable plan, and an "
                    "allowed failure type.",
                    *[f"Would suggest: {step}" for step in plan.steps],
                ],
            )

        command = self._safe_command(plan, consent)
        if command is None:
            return SandboxResult(
                applied=False,
                advisory_only=True,
                dry_run=consent.dry_run,
                risk_level=risk,
                notes=[
                    "No whitelisted safe command for this plan "
                    "(e.g. set consent.allow_install=True for missing dependencies)."
                ],
            )

        if consent.dry_run:
            return SandboxResult(
                applied=False,
                advisory_only=False,
                dry_run=True,
                risk_level=risk,
                command=command,
                notes=[f"DRY RUN — would execute: {' '.join(command)}"],
            )

        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, validated package
                command,
                capture_output=True,
                text=True,
                timeout=consent.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                applied=False,
                advisory_only=False,
                dry_run=False,
                risk_level=risk,
                command=command,
                notes=[f"Timed out after {consent.timeout}s"],
            )

        return SandboxResult(
            applied=proc.returncode == 0,
            advisory_only=False,
            dry_run=False,
            risk_level=risk,
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout[-_MAX_CAPTURE:],
            stderr=proc.stderr[-_MAX_CAPTURE:],
            notes=[f"Executed: {' '.join(command)}", f"returncode={proc.returncode}"],
        )

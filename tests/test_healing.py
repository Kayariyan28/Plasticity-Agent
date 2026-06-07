"""Tests for advisory self-healing: diagnosis and repair planning."""

from __future__ import annotations

from plasticity_agent.healing.diagnosis import FailureDiagnosis, diagnose
from plasticity_agent.healing.repair import heal, plan_repair
from plasticity_agent.healing.sandbox import RepairConsent, Sandbox


def test_missing_dependency_from_exception() -> None:
    diagnosis = diagnose(ModuleNotFoundError("No module named 'foo'"))
    assert diagnosis.failure_type == "missing_dependency"
    assert "foo" in diagnosis.root_cause


def test_tool_schema_error_from_type_error() -> None:
    diagnosis = diagnose(TypeError("f() missing 1 required positional argument: 'bar'"))
    assert diagnosis.failure_type == "tool_schema_error"


def test_timeout() -> None:
    assert diagnose(TimeoutError("operation timed out")).failure_type == "timeout"


def test_file_not_found() -> None:
    diagnosis = diagnose(FileNotFoundError("No such file or directory: 'x.txt'"))
    assert diagnosis.failure_type == "file_not_found"


def test_permission_error() -> None:
    assert diagnose(PermissionError("Permission denied")).failure_type == "permission_error"


def test_value_error() -> None:
    assert diagnose(ValueError("invalid literal for int()")).failure_type == "value_error"


def test_unknown_has_low_confidence() -> None:
    diagnosis = diagnose("an opaque message with no recognisable signal")
    assert diagnosis.failure_type == "unknown"
    assert diagnosis.confidence < 0.5


def test_string_error_is_classified() -> None:
    diagnosis = diagnose("ModuleNotFoundError: No module named 'requests'")
    assert diagnosis.failure_type == "missing_dependency"


def test_safe_repair_is_opt_in_applicable_others_advisory() -> None:
    # Narrowly safe, reversible repairs become opt-in applicable...
    dependency = heal(ModuleNotFoundError("No module named 'foo'"))
    assert dependency.auto_apply_allowed is True
    assert dependency.advisory_only is False
    assert dependency.steps
    assert dependency.risk_level in {"low", "medium", "high"}
    # ...but everything else stays advisory-only.
    permission = heal(PermissionError("Permission denied"))
    assert permission.auto_apply_allowed is False
    assert permission.advisory_only is True


def test_sandbox_does_not_apply_without_consent() -> None:
    plan = heal(ModuleNotFoundError("No module named 'foo'"))
    result = Sandbox().apply(plan)  # default consent: everything off
    assert result.applied is False
    assert result.advisory_only is True


def test_sandbox_dry_run_builds_command_but_does_not_execute() -> None:
    plan = heal(ModuleNotFoundError("No module named 'rich'"))
    consent = RepairConsent(allow_apply=True, allow_install=True, dry_run=True)
    result = Sandbox().apply(plan, consent)
    assert result.applied is False
    assert result.dry_run is True
    assert result.command is not None
    assert "install" in result.command
    assert "rich" in result.command


def test_sandbox_rejects_unsafe_package_name() -> None:
    # Craft a plan whose package name contains shell metacharacters. Even with
    # full consent and dry_run off, the validator refuses to build a command.
    diagnosis = FailureDiagnosis(
        failure_type="missing_dependency",
        root_cause="crafted",
        repair_strategy="crafted",
        confidence=0.9,
        details={"package": "foo; rm -rf /"},
    )
    plan = plan_repair(diagnosis)
    consent = RepairConsent(allow_apply=True, allow_install=True, dry_run=False)
    result = Sandbox().apply(plan, consent)
    assert result.applied is False
    assert result.command is None

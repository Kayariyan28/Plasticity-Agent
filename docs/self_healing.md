# Self-Healing

!!! warning "Advisory by default; execution is strictly opt-in"
    Self-healing **never edits, moves, or deletes your source files**. By default it only
    diagnoses and recommends. Execution happens only when you pass an explicit `RepairConsent`,
    is limited to a hard allowlist (currently `pip install` of a **validated** package name), runs
    via `subprocess` with **no shell** and a timeout, and is gated behind multiple flags.

## Diagnose an error

```python
from plasticity_agent.healing.diagnosis import diagnose

diagnosis = diagnose(ModuleNotFoundError("No module named 'pandas'"))
# FailureDiagnosis(failure_type='missing_dependency', confidence=0.9, ...)
```

`diagnose` accepts an exception **or** a raw message string. Failure types:

`tool_schema_error`, `missing_dependency`, `timeout`, `import_error`, `type_error`,
`value_error`, `permission_error`, `file_not_found`, `unknown`.

| Error | Classified as |
| --- | --- |
| `ModuleNotFoundError` | `missing_dependency` (suggests `uv add` / `pip install`) |
| `TypeError: missing ... argument` | `tool_schema_error` |
| `TimeoutError` | `timeout` (suggests retry + backoff) |
| `FileNotFoundError` | `file_not_found` |
| `PermissionError` | `permission_error` |
| `ValueError` | `value_error` |

## Get a repair plan

```python
from plasticity_agent.healing.repair import heal

plan = heal(ModuleNotFoundError("No module named 'pandas'"))
print(plan.risk_level)            # 'medium'
print(plan.auto_apply_allowed)    # False
for step in plan.steps:
    print("-", step)
```

`RepairPlan` carries the `diagnosis`, ordered `steps`, a `risk_level` (`low`/`medium`/`high`),
and the advisory flags.

## From the agent or CLI

```python
plan = agent.heal(error)          # also emits a healing_diagnosed trace
```

```bash
plasticity heal "ModuleNotFoundError: No module named 'pandas'"
```

## Opt-in sandboxed repair

Only narrow, reversible repairs are *eligible* (`auto_apply_allowed=True` — currently
`missing_dependency`). Even then, nothing runs without a `RepairConsent`:

```python
from plasticity_agent import RepairConsent

agent.apply_repair(error)                                            # advisory — runs nothing
agent.apply_repair(error, RepairConsent(allow_apply=True, allow_install=True, dry_run=True))
# -> SandboxResult(applied=False, dry_run=True, command=[python, -m, pip, install, <pkg>])

agent.apply_repair(error, RepairConsent(allow_apply=True, allow_install=True, dry_run=False))
# -> actually installs the package into the current environment (reversible)
```

The gates, all of which must pass: the plan must be `auto_apply_allowed`; `consent.allow_apply`
must be `True`; the failure type must be in `consent.allowed_types`; the relevant capability
(e.g. `allow_install`) must be enabled; and `dry_run` must be `False` to execute. Package names
are validated against a strict pattern, so shell metacharacters are rejected before anything runs.

```bash
plasticity apply "ModuleNotFoundError: No module named 'rich'" --install            # dry run
plasticity apply "ModuleNotFoundError: No module named 'rich'" --install --execute  # run it
```

## Detecting failures in traces

```python
from plasticity_agent.healing.detector import detect_failures
diagnoses = detect_failures(memory.load_traces())   # scans run_failed events
```

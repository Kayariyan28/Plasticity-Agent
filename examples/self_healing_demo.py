"""Advisory self-healing: diagnose common errors and propose repair plans.

Nothing is ever auto-applied in v0.1.0 — these are recommendations only.

Run:  uv run python examples/self_healing_demo.py
"""

from __future__ import annotations

from plasticity_agent.healing.repair import heal

ERRORS = [
    ModuleNotFoundError("No module named 'pandas'"),
    TypeError("run() missing 1 required positional argument: 'config'"),
    TimeoutError("request timed out after 30s"),
    FileNotFoundError("No such file or directory: 'data/input.csv'"),
    PermissionError("Permission denied: '/etc/hosts'"),
]


def main() -> None:
    for error in ERRORS:
        plan = heal(error)
        diagnosis = plan.diagnosis
        print(f"=== {type(error).__name__}: {error} ===")
        print(f"  type:       {diagnosis.failure_type}  (confidence {diagnosis.confidence:.2f})")
        print(f"  root cause: {diagnosis.root_cause}")
        print(f"  risk:       {plan.risk_level}  |  advisory_only={plan.advisory_only}  "
              f"auto_apply_allowed={plan.auto_apply_allowed}")
        print(f"  step 1:     {plan.steps[0]}\n")


if __name__ == "__main__":
    main()

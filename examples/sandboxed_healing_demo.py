"""Opt-in sandboxed healing — advisory by default, explicit consent to execute.

Run:  uv run python examples/sandboxed_healing_demo.py
"""

from __future__ import annotations

from plasticity_agent.healing.repair import heal
from plasticity_agent.healing.sandbox import RepairConsent, Sandbox


def main() -> None:
    plan = heal(ModuleNotFoundError("No module named 'cowsay'"))
    sandbox = Sandbox()

    print("== default (no consent): nothing is executed ==")
    result = sandbox.apply(plan)
    print(f"  applied={result.applied}  advisory_only={result.advisory_only}")

    print("\n== consent + dry-run: shows the command, still does not run ==")
    result = sandbox.apply(plan, RepairConsent(allow_apply=True, allow_install=True, dry_run=True))
    print(f"  applied={result.applied}  dry_run={result.dry_run}")
    print(f"  would run: {' '.join(result.command or [])}")

    print(
        "\nTo actually execute, pass dry_run=False — it installs the package into the\n"
        "current environment (a reversible repair). It never edits your source files."
    )


if __name__ == "__main__":
    main()

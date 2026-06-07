"""Tests for the Typer CLI (imports cleanly and core commands run)."""

from __future__ import annotations

from typer.testing import CliRunner

from plasticity_agent.cli import app

runner = CliRunner()


def test_cli_app_imports() -> None:
    assert app is not None


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "plasticity-agent" in result.output


def test_cli_init(tmp_path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path / "mem")])
    assert result.exit_code == 0
    assert (tmp_path / "mem").exists()


def test_cli_remember_and_recall(tmp_path) -> None:
    memory_dir = str(tmp_path / "mem")
    recorded = runner.invoke(
        app,
        ["remember", "Cache warmup reduced latency", "--type", "semantic", "--tag", "important",
         "--memory", memory_dir],
    )
    assert recorded.exit_code == 0
    recalled = runner.invoke(app, ["recall", "latency", "--memory", memory_dir])
    assert recalled.exit_code == 0


def test_cli_market() -> None:
    result = runner.invoke(app, ["market", "Choose a repair strategy"])
    assert result.exit_code == 0


def test_cli_heal() -> None:
    result = runner.invoke(app, ["heal", "ModuleNotFoundError: No module named 'foo'"])
    assert result.exit_code == 0
    assert "missing_dependency" in result.output


def test_cli_sleep_and_report(tmp_path) -> None:
    memory_dir = str(tmp_path / "mem")
    runner.invoke(app, ["remember", "a note", "--memory", memory_dir])
    sleep_result = runner.invoke(app, ["sleep", "--memory", memory_dir])
    assert sleep_result.exit_code == 0
    report_result = runner.invoke(app, ["report", "--memory", memory_dir])
    assert report_result.exit_code == 0


def test_cli_evaluate_and_skills(tmp_path) -> None:
    memory_dir = str(tmp_path / "mem")
    runner.invoke(app, ["remember", "evaluate me", "--memory", memory_dir])
    assert runner.invoke(app, ["evaluate", "--memory", memory_dir]).exit_code == 0
    assert runner.invoke(app, ["skills", "--memory", memory_dir]).exit_code == 0

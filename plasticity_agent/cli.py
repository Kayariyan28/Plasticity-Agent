"""The ``plasticity`` command-line interface (Typer + Rich).

A readable, local-first console for inspecting and exercising an agent's memory,
sleep cycle, healing advice, reasoning market, skills, and energy report.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from plasticity_agent import __version__
from plasticity_agent.core.agent import PlasticAgent
from plasticity_agent.core.config import PlasticityConfig
from plasticity_agent.core.trace import load_trace_records
from plasticity_agent.healing.repair import heal as heal_error
from plasticity_agent.healing.sandbox import RepairConsent, Sandbox
from plasticity_agent.memory.memory_os import MemoryOS
from plasticity_agent.memory.schemas import MemoryType
from plasticity_agent.reasoning.market import ReasoningMarket
from plasticity_agent.thermodynamics.energy_report import build_energy_report

app = typer.Typer(
    help="Plasticity Agent Runtime — neuroplastic memory, self-healing, and reasoning.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_VALID_TYPES = {"episodic", "semantic", "procedural", "reflective", "constitutional"}


def _memory(memory_dir: str) -> MemoryOS:
    return MemoryOS(memory_dir=memory_dir)


@app.command()
def version() -> None:
    """Print the installed version."""

    console.print(f"plasticity-agent [bold cyan]{__version__}[/]")


@app.command()
def init(
    path: str = typer.Argument("./memory", help="Memory directory to initialize."),
) -> None:
    """Initialize a local-first memory directory (SQLite + traces)."""

    config = PlasticityConfig.from_memory_dir(path)
    config.ensure_dirs()
    memory = MemoryOS(config=config)
    memory.close()
    console.print(
        Panel(
            f"Initialized memory at [bold]{config.memory_dir}[/]\n"
            f"• database: {config.db_path}\n• traces:   {config.traces_dir}",
            title="plasticity init",
            border_style="green",
        )
    )


@app.command()
def remember(
    text: str = typer.Argument(..., help="The memory content."),
    memory_type: str = typer.Option("episodic", "--type", "-t", help="Memory type."),
    tag: list[str] = typer.Option([], "--tag", help="Tag (repeatable)."),
    reward: float = typer.Option(0.0, "--reward", "-r"),
    confidence: float = typer.Option(0.7, "--confidence", "-c"),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Record a new memory."""

    if memory_type not in _VALID_TYPES:
        console.print(
            f"[red]Invalid --type '{memory_type}'.[/] Choose from {sorted(_VALID_TYPES)}."
        )
        raise typer.Exit(code=1)

    memory = _memory(memory_dir)
    record = memory.record(
        text, cast(MemoryType, memory_type), tags=list(tag), reward=reward, confidence=confidence
    )
    memory.close()
    console.print(
        Panel(
            f"[bold]{record.content}[/]\n"
            f"type={record.memory_type}  salience={record.salience:.2f}  "
            f"confidence={record.confidence:.2f}  contradiction={record.contradiction_score:.2f}\n"
            f"tags={record.tags or '—'}  id={record.id}",
            title="memory recorded",
            border_style="cyan",
        )
    )


@app.command()
def recall(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(5, "--limit", "-n"),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Recall memories relevant to a query."""

    memory = _memory(memory_dir)
    results = memory.recall(query, limit=limit)
    memory.close()

    if not results:
        console.print("[yellow]No matching memories.[/]")
        return
    table = Table(title=f"recall: {query}")
    table.add_column("score", justify="right", style="cyan")
    table.add_column("type")
    table.add_column("content")
    table.add_column("why", style="dim")
    for result in results:
        table.add_row(
            f"{result.score:.3f}",
            result.memory.memory_type,
            result.memory.content[:80],
            result.match_reason,
        )
    console.print(table)


@app.command()
def evaluate(
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Evaluate the quality of every memory."""

    memory = _memory(memory_dir)
    reports = memory.evaluate_all()
    memory.close()

    if not reports:
        console.print("[yellow]No memories to evaluate.[/]")
        return
    table = Table(title="memory quality")
    table.add_column("utility", justify="right", style="cyan")
    table.add_column("recommendation")
    table.add_column("salience", justify="right")
    table.add_column("contradiction", justify="right")
    table.add_column("reasons", style="dim")
    for report in sorted(reports, key=lambda r: r.utility_score, reverse=True):
        table.add_row(
            f"{report.utility_score:.3f}",
            report.recommendation,
            f"{report.salience:.2f}",
            f"{report.contradiction_score:.2f}",
            "; ".join(report.reasons),
        )
    console.print(table)


@app.command()
def sleep(
    path: str = typer.Argument(None, help="Optional external traces directory to analyze."),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Run a sleep/consolidation cycle and print a report."""

    memory = _memory(memory_dir)
    report = memory.sleep(traces_path=path)
    memory.close()

    lines = [
        f"[green]✓[/] {report.traces_analyzed} traces analyzed",
        f"[green]✓[/] {report.weak_memories_decayed} weak memories decayed",
        f"[green]✓[/] {report.memories_consolidated} memories consolidated",
        f"[green]✓[/] {report.contradictions_detected} contradictions detected",
        f"[green]✓[/] {report.skills_created} reusable skills created",
        f"[green]✓[/] {report.policies_improved} prompt policies improved",
        f"[green]✓[/] agent plasticity score: {report.plasticity_score:.0f}/100",
    ]
    console.print(Panel("\n".join(lines), title="plasticity sleep", border_style="magenta"))


@app.command()
def report(
    path: str = typer.Argument(None, help="Optional external traces directory to analyze."),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Print a thermodynamic-style energy report."""

    memory = _memory(memory_dir)
    trace_records = load_trace_records(path) if path else memory.load_traces()
    energy = build_energy_report(memory.list_memories(), trace_records)
    memory.close()

    table = Table(title="energy report")
    table.add_column("metric", style="cyan")
    table.add_column("value")
    table.add_row("memory_entropy", f"{energy.memory_entropy:.3f}")
    table.add_row("contradiction_pressure", f"{energy.contradiction_pressure:.3f}")
    table.add_row("token_waste", f"{energy.token_waste:.0f} tokens")
    table.add_row("repair_energy", energy.repair_energy)
    table.add_row("confidence_temperature", energy.confidence_temperature)
    table.add_row("plasticity_score", f"{energy.plasticity_score:.0f}/100")
    console.print(table)
    console.print(Panel(energy.summary, border_style="blue"))


@app.command()
def heal(
    error: str = typer.Argument(..., help="The error text to diagnose."),
) -> None:
    """Diagnose an error and print an advisory repair plan."""

    plan = heal_error(error)
    diagnosis = plan.diagnosis
    steps = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(plan.steps))
    console.print(
        Panel(
            f"[bold]type:[/] {diagnosis.failure_type}  "
            f"[bold]confidence:[/] {diagnosis.confidence:.2f}  "
            f"[bold]risk:[/] {plan.risk_level}\n"
            f"[bold]root cause:[/] {diagnosis.root_cause}\n"
            f"[bold]strategy:[/] {diagnosis.repair_strategy}\n\n"
            f"[bold]advisory steps:[/]\n{steps}\n\n"
            f"[dim]advisory_only={plan.advisory_only} • "
            f"auto_apply_allowed={plan.auto_apply_allowed}[/]",
            title="plasticity heal (advisory)",
            border_style="yellow",
        )
    )


@app.command()
def market(
    task: str = typer.Argument(..., help="The task to deliberate on."),
) -> None:
    """Run the reasoning market on a task and show the winning proposal."""

    result = ReasoningMarket().deliberate(task)
    table = Table(title=f"reasoning market: {task}")
    table.add_column("rank", justify="right")
    table.add_column("critic", style="cyan")
    table.add_column("action")
    for index, proposal in enumerate(result.ranked, start=1):
        table.add_row(str(index), proposal.critic_name, proposal.action[:70])
    console.print(table)
    console.print(
        Panel(
            f"[bold green]winner:[/] {result.winner.critic_name}\n"
            f"{result.winner.action}\n\n"
            f"[dim]selection score: {result.selection_score:.3f}[/]",
            border_style="green",
        )
    )


@app.command()
def skills(
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """List learned skills."""

    memory = _memory(memory_dir)
    learned = memory.skills.list_skills()
    memory.close()

    if not learned:
        console.print("[yellow]No skills learned yet.[/] Run some tasks and `plasticity sleep`.")
        return
    table = Table(title="skill library")
    table.add_column("name", style="cyan")
    table.add_column("description")
    table.add_column("usage", justify="right")
    table.add_column("confidence", justify="right")
    for skill in learned:
        table.add_row(
            skill.name, skill.description[:60], str(skill.usage_count), f"{skill.confidence:.2f}"
        )
    console.print(table)


@app.command()
def export(
    path: str = typer.Argument(None, help="Output JSONL path."),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Export all memories to JSONL."""

    memory = _memory(memory_dir)
    target = memory.export_jsonl(path)
    count = memory.count()
    memory.close()
    console.print(f"[green]✓[/] exported {count} memories to [bold]{target}[/]")


@app.command()
def metrics(
    checkpoint: bool = typer.Option(
        False, "--checkpoint", help="Record a new checkpoint before reporting."
    ),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Show cross-run improvement metrics (and optionally checkpoint first)."""

    agent = PlasticAgent(name="cli", memory=memory_dir, reasoning_market=False)
    if checkpoint:
        snapshot = agent.checkpoint("cli")
        console.print(
            f"[green]✓[/] checkpoint recorded: plasticity={snapshot.plasticity_score:.0f}/100, "
            f"utility={snapshot.avg_utility:.3f}, memories={snapshot.memories}"
        )
    report = agent.improvement()
    agent.close()

    if not report.improved and report.snapshots < 2:
        console.print(f"[yellow]{report.summary}[/]")
        return
    verdict = "[green]improved[/]" if report.improved else "[red]regressed/flat[/]"
    table = Table(title=f"improvement across {report.snapshots} checkpoints")
    table.add_column("metric", style="cyan")
    table.add_column("delta", justify="right")
    table.add_row("plasticity", f"{report.plasticity_delta:+.1f}")
    table.add_row("avg_utility", f"{report.utility_delta:+.3f}")
    table.add_row("contradiction", f"{report.contradiction_delta:+.3f}")
    table.add_row("skills", f"{report.skills_delta:+d}")
    table.add_row("score", f"{report.improvement_score:+.3f}")
    console.print(table)
    console.print(Panel(f"Verdict: {verdict}\n{report.summary}", border_style="blue"))


@app.command()
def apply(
    error: str = typer.Argument(..., help="The error text to diagnose and (optionally) repair."),
    install: bool = typer.Option(False, "--install", help="Permit `pip install` repairs."),
    execute: bool = typer.Option(
        False, "--execute", help="Actually run the repair (default is a safe dry-run)."
    ),
) -> None:
    """Sandboxed, opt-in repair. Advisory/dry-run unless --execute is given."""

    plan = heal_error(error)
    consent = RepairConsent(allow_apply=True, allow_install=install, dry_run=not execute)
    result = Sandbox().apply(plan, consent)
    style = "green" if result.applied else ("yellow" if result.dry_run else "red")
    body = [
        f"[bold]failure:[/] {plan.diagnosis.failure_type}",
        f"[bold]applied:[/] {result.applied}   [bold]dry_run:[/] {result.dry_run}",
    ]
    if result.command:
        body.append(f"[bold]command:[/] {' '.join(result.command)}")
    body.extend(result.notes)
    console.print(Panel("\n".join(body), title="plasticity apply (sandboxed)", border_style=style))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Start the FastAPI server."""

    try:
        from plasticity_agent.server.api import serve as run_server
    except ImportError as exc:  # pragma: no cover
        console.print(f"[red]FastAPI/uvicorn not installed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Serving[/] on http://{host}:{port} (memory={memory_dir})")
    run_server(host=host, port=port, memory_dir=memory_dir)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port"),
    memory_dir: str = typer.Option("./memory", "--memory", "-m"),
) -> None:
    """Launch the Streamlit dashboard."""

    import subprocess

    import plasticity_agent.server as server_pkg

    dash_path = Path(server_pkg.__file__).parent / "dashboard.py"
    env = dict(os.environ, PLASTICITY_MEMORY_DIR=memory_dir)
    console.print(f"[green]Launching dashboard[/] on port {port} (memory={memory_dir})")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dash_path), "--server.port", str(port)],
        env=env,
        check=False,
    )


if __name__ == "__main__":
    app()

"""Cross-run improvement metrics.

Persists periodic metric *snapshots* (plasticity, utility, contradiction,
entropy, skill/memory counts) and computes whether the agent is actually getting
better over time — a signed improvement score plus per-metric deltas and a
trend. Storage is a small SQLite table inside the local-first memory directory.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MetricSnapshot(BaseModel):
    """A point-in-time measurement of agent health.

    ``grounded_utility`` weights each memory's utility by how much it has
    actually been *used* (recalled), so simply storing favorable-looking
    memories does not inflate it — only memories that prove useful do.
    """

    label: str = "checkpoint"
    timestamp: datetime = Field(default_factory=_utcnow)
    memories: int = 0
    skills: int = 0
    plasticity_score: float = 0.0
    avg_utility: float = 0.0
    grounded_utility: float = 0.0
    contradiction_pressure: float = 0.0
    memory_entropy: float = 0.0


class ImprovementReport(BaseModel):
    """Did the agent improve across recorded snapshots?"""

    snapshots: int
    improved: bool
    improvement_score: float
    plasticity_delta: float = 0.0
    utility_delta: float = 0.0
    grounded_utility_delta: float = 0.0
    contradiction_delta: float = 0.0
    entropy_delta: float = 0.0
    skills_delta: int = 0
    summary: str = ""
    first: MetricSnapshot | None = None
    last: MetricSnapshot | None = None


class ImprovementTracker:
    """SQLite-backed store of :class:`MetricSnapshot` history + analysis."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if str(self.db_path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_snapshots (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    label TEXT,
                    data TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def record(self, snapshot: MetricSnapshot) -> MetricSnapshot:
        with self._lock:
            self._conn.execute(
                "INSERT INTO metric_snapshots (ts, label, data) VALUES (?, ?, ?)",
                (snapshot.timestamp.isoformat(), snapshot.label, snapshot.model_dump_json()),
            )
            self._conn.commit()
        return snapshot

    def history(self) -> list[MetricSnapshot]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM metric_snapshots ORDER BY seq ASC"
            ).fetchall()
        return [MetricSnapshot.model_validate_json(row["data"]) for row in rows]

    def report(self) -> ImprovementReport:
        """Compare the first and latest snapshots into an improvement verdict."""

        history = self.history()
        if len(history) < 2:
            return ImprovementReport(
                snapshots=len(history),
                improved=False,
                improvement_score=0.0,
                summary="Need at least two checkpoints to measure improvement.",
                first=history[0] if history else None,
                last=history[-1] if history else None,
            )

        first, last = history[0], history[-1]
        plasticity_delta = last.plasticity_score - first.plasticity_score
        utility_delta = last.avg_utility - first.avg_utility
        grounded_delta = last.grounded_utility - first.grounded_utility
        contradiction_delta = last.contradiction_pressure - first.contradiction_pressure
        entropy_delta = last.memory_entropy - first.memory_entropy
        skills_delta = last.skills - first.skills

        # Reward signals that are hard to fake by simply storing nice-looking
        # memories: less contradiction, *used* (grounded) utility, and learned
        # skills. Raw salience/plasticity is reported but deliberately NOT in the
        # verdict, because it inflates trivially when you add high-reward memories.
        skills_term = max(-1.0, min(1.0, skills_delta / 4.0))
        improvement_score = (
            0.45 * (-contradiction_delta)
            + 0.40 * grounded_delta
            + 0.15 * skills_term
        )
        improved = improvement_score > 0.0
        verdict = "improved" if improved else "regressed or flat"
        summary = (
            f"Over {len(history)} checkpoints the agent {verdict} "
            f"(score {improvement_score:+.3f}): contradiction {contradiction_delta:+.3f}, "
            f"grounded-utility {grounded_delta:+.3f}, skills {skills_delta:+d} "
            f"(plasticity {plasticity_delta:+.1f} shown for context, not scored)."
        )
        return ImprovementReport(
            snapshots=len(history),
            improved=improved,
            improvement_score=round(improvement_score, 4),
            plasticity_delta=round(plasticity_delta, 3),
            utility_delta=round(utility_delta, 4),
            grounded_utility_delta=round(grounded_delta, 4),
            contradiction_delta=round(contradiction_delta, 4),
            entropy_delta=round(entropy_delta, 4),
            skills_delta=skills_delta,
            summary=summary,
            first=first,
            last=last,
        )

    def to_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for snapshot in self.history():
                handle.write(snapshot.model_dump_json() + "\n")
        return target

    def close(self) -> None:
        with self._lock:
            self._conn.close()

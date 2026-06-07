"""Runtime configuration for a Plasticity agent.

Everything is local-first: a single SQLite database plus JSONL trace logs live
under one ``memory`` directory. The defaults match the paths documented in the
README (``./memory/plasticity.sqlite`` and ``./memory/traces/``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_MEMORY_DIR = "./memory"


class PlasticityConfig(BaseModel):
    """Paths and tunable constants for the runtime.

    The scoring/decay knobs are surfaced here so they can be overridden per
    agent without editing module-level constants.
    """

    memory_dir: Path = Field(default=Path(DEFAULT_MEMORY_DIR))
    db_filename: str = "plasticity.sqlite"
    metrics_filename: str = "metrics.sqlite"
    traces_dirname: str = "traces"

    # Forgetting / decay tuning.
    decay_increment: float = 0.05
    decay_age_days_threshold: float = 14.0
    low_usage_threshold: int = 2
    prune_utility_threshold: float = 0.15

    # Consolidation tuning.
    consolidation_min_cluster: int = 2
    consolidation_similarity: float = 0.34

    model_config = {"arbitrary_types_allowed": True}

    @property
    def db_path(self) -> Path:
        return self.memory_dir / self.db_filename

    @property
    def metrics_path(self) -> Path:
        return self.memory_dir / self.metrics_filename

    @property
    def traces_dir(self) -> Path:
        return self.memory_dir / self.traces_dirname

    def ensure_dirs(self) -> None:
        """Create the memory and traces directories if they do not exist."""

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_memory_dir(cls, memory_dir: str | Path | None = None) -> PlasticityConfig:
        """Build a config rooted at ``memory_dir`` (defaults to ``./memory``)."""

        return cls(memory_dir=Path(memory_dir or DEFAULT_MEMORY_DIR))

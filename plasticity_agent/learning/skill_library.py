"""Voyager-style skill library.

Successful, repeatable traces become named, reusable skills. Skills persist in
their own SQLite table inside the same local-first memory directory, so a skill
learned in one run is available to the next.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from plasticity_agent.memory.retrieval import lexical_similarity, token_set


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_skill_id() -> str:
    return f"skill_{uuid.uuid4().hex[:12]}"


class Skill(BaseModel):
    """A reusable, named capability mined from successful experience."""

    id: str = Field(default_factory=_new_skill_id)
    name: str
    description: str = ""
    trigger_patterns: list[str] = Field(default_factory=list)
    successful_trace: dict[str, Any] = Field(default_factory=dict)
    usage_count: int = 0
    confidence: float = 0.5
    reward: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class SkillLibrary:
    """SQLite-backed CRUD + lexical search over :class:`Skill` objects."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        is_memory = str(self.db_path) == ":memory:"
        if not is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if not is_memory:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    data TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def _write(self, skill: Skill) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO skills (id, name, data) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name, data = excluded.data
                """,
                (skill.id, skill.name, skill.model_dump_json()),
            )
            self._conn.commit()

    def _by_name(self, name: str) -> Skill | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM skills WHERE name = ?", (name,)
            ).fetchone()
        return Skill.model_validate_json(row["data"]) if row else None

    def get(self, skill_id: str) -> Skill | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
        return Skill.model_validate_json(row["data"]) if row else None

    def save(
        self,
        name: str,
        successful_trace: dict[str, Any] | None = None,
        *,
        description: str | None = None,
        trigger_patterns: list[str] | None = None,
        confidence: float = 0.6,
        reward: float = 0.0,
    ) -> Skill:
        """Create or update a skill by ``name`` and return it.

        Re-saving an existing skill reinforces it: usage increments and
        confidence/reward move toward the new evidence.
        """

        with self._lock:
            return self._save_locked(
                name,
                successful_trace,
                description=description,
                trigger_patterns=trigger_patterns,
                confidence=confidence,
                reward=reward,
            )

    def _save_locked(
        self,
        name: str,
        successful_trace: dict[str, Any] | None,
        *,
        description: str | None,
        trigger_patterns: list[str] | None,
        confidence: float,
        reward: float,
    ) -> Skill:
        existing = self._by_name(name)
        if existing is not None:
            existing.usage_count += 1
            existing.confidence = min(1.0, (existing.confidence + confidence) / 2 + 0.05)
            existing.reward = (existing.reward + reward) / 2
            if successful_trace:
                existing.successful_trace = successful_trace
            if description:
                existing.description = description
            if trigger_patterns:
                existing.trigger_patterns = sorted(
                    set(existing.trigger_patterns) | set(trigger_patterns)
                )
            existing.updated_at = _utcnow()
            self._write(existing)
            return existing

        skill = Skill(
            name=name,
            description=description or f"Skill: {name}",
            trigger_patterns=trigger_patterns or [],
            successful_trace=successful_trace or {},
            confidence=confidence,
            reward=reward,
        )
        self._write(skill)
        return skill

    def promote_from_trace(
        self,
        name: str,
        successful_trace: dict[str, Any],
        *,
        description: str | None = None,
        trigger_patterns: list[str] | None = None,
        confidence: float = 0.6,
        reward: float = 0.0,
    ) -> Skill:
        """Promote a successful trace into a named skill (alias of :meth:`save`)."""

        return self.save(
            name,
            successful_trace,
            description=description,
            trigger_patterns=trigger_patterns,
            confidence=confidence,
            reward=reward,
        )

    def list_skills(self) -> list[Skill]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM skills").fetchall()
        skills = [Skill.model_validate_json(row["data"]) for row in rows]
        skills.sort(key=lambda s: (s.usage_count, s.confidence), reverse=True)
        return skills

    def find(self, query: str, *, limit: int = 5) -> list[Skill]:
        """Lexically rank skills against a free-text query."""

        query_tokens = token_set(query)
        scored: list[tuple[float, Skill]] = []
        for skill in self.list_skills():
            haystack = " ".join([skill.name, skill.description, *skill.trigger_patterns])
            score = lexical_similarity(query, haystack)
            if any(pattern.lower() in query_tokens for pattern in skill.trigger_patterns):
                score = min(1.0, score + 0.2)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _, skill in scored[:limit]]

    def use(self, name_or_id: str) -> Skill | None:
        """Mark a skill as used (increments ``usage_count``) and return it."""

        with self._lock:
            skill = self._by_name(name_or_id) or self.get(name_or_id)
            if skill is None:
                return None
            skill.usage_count += 1
            skill.updated_at = _utcnow()
            self._write(skill)
            return skill

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

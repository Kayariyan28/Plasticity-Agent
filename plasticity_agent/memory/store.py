"""SQLite persistence for memories and their embedding vectors.

Local-first and concurrency-safe: WAL journaling for reader/writer concurrency,
a busy timeout, a process-wide lock guarding the shared connection (so the store
can be shared across threads, e.g. by the FastAPI server), and automatic retry
on transient ``database is locked`` errors from other processes.

Memories are stored as JSON blobs; embeddings are stored as packed float BLOBs
in a parallel ``vectors`` table for dense/hybrid retrieval.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from plasticity_agent.memory.schemas import Memory

T = TypeVar("T")

_MAX_RETRIES = 6
_RETRY_BASE_SLEEP = 0.05


class MemoryStore:
    """Thread-safe, WAL-backed SQLite CRUD for memories and vectors."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._is_memory = str(self.db_path) == ":memory:"
        if not self._is_memory and self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        if not self._is_memory:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    # -- low-level helpers ----------------------------------------------------

    def _run(self, fn: Callable[[], T]) -> T:
        """Run a DB operation under the lock, retrying on transient locks."""

        with self._lock:
            last: sqlite3.OperationalError | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    return fn()
                except sqlite3.OperationalError as exc:  # pragma: no cover - timing dependent
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    last = exc
                    time.sleep(_RETRY_BASE_SLEEP * (2**attempt))
            assert last is not None
            raise last

    def _init_schema(self) -> None:
        def op() -> None:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT,
                    updated_at TEXT,
                    data TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vec BLOB NOT NULL
                )
                """
            )
            self._conn.commit()

        self._run(op)

    # -- memory writes --------------------------------------------------------

    def upsert(self, memory: Memory) -> None:
        def op() -> None:
            self._conn.execute(
                """
                INSERT INTO memories (id, memory_type, updated_at, data)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    updated_at = excluded.updated_at,
                    data = excluded.data
                """,
                (memory.id, memory.memory_type, memory.updated_at.isoformat(),
                 memory.model_dump_json()),
            )
            self._conn.commit()

        self._run(op)

    def upsert_many(self, memories: Iterable[Memory]) -> None:
        rows = [
            (m.id, m.memory_type, m.updated_at.isoformat(), m.model_dump_json())
            for m in memories
        ]
        if not rows:
            return

        def op() -> None:
            self._conn.executemany(
                """
                INSERT INTO memories (id, memory_type, updated_at, data)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    updated_at = excluded.updated_at,
                    data = excluded.data
                """,
                rows,
            )
            self._conn.commit()

        self._run(op)

    def delete(self, memory_id: str) -> bool:
        def op() -> bool:
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM vectors WHERE id = ?", (memory_id,))
            self._conn.commit()
            return cursor.rowcount > 0

        return self._run(op)

    def clear(self) -> None:
        def op() -> None:
            self._conn.execute("DELETE FROM memories")
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()

        self._run(op)

    # -- memory reads ---------------------------------------------------------

    def get(self, memory_id: str) -> Memory | None:
        def op() -> Memory | None:
            row = self._conn.execute(
                "SELECT data FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return Memory.model_validate_json(row["data"]) if row else None

        return self._run(op)

    def all(self) -> list[Memory]:
        def op() -> list[Memory]:
            rows = self._conn.execute(
                "SELECT data FROM memories ORDER BY updated_at DESC"
            ).fetchall()
            return [Memory.model_validate_json(row["data"]) for row in rows]

        return self._run(op)

    def by_type(self, memory_type: str) -> list[Memory]:
        def op() -> list[Memory]:
            rows = self._conn.execute(
                "SELECT data FROM memories WHERE memory_type = ? ORDER BY updated_at DESC",
                (memory_type,),
            ).fetchall()
            return [Memory.model_validate_json(row["data"]) for row in rows]

        return self._run(op)

    def count(self) -> int:
        def op() -> int:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
            return int(row["n"]) if row else 0

        return self._run(op)

    # -- vectors --------------------------------------------------------------

    def upsert_vector(self, memory_id: str, vector: Sequence[float]) -> None:
        blob = struct.pack(f"<{len(vector)}f", *vector)

        def op() -> None:
            self._conn.execute(
                """
                INSERT INTO vectors (id, dim, vec) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET dim = excluded.dim, vec = excluded.vec
                """,
                (memory_id, len(vector), blob),
            )
            self._conn.commit()

        self._run(op)

    def get_vector(self, memory_id: str) -> list[float] | None:
        def op() -> list[float] | None:
            row = self._conn.execute(
                "SELECT dim, vec FROM vectors WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            return list(struct.unpack(f"<{row['dim']}f", row["vec"]))

        return self._run(op)

    def all_vectors(self) -> dict[str, list[float]]:
        def op() -> dict[str, list[float]]:
            rows = self._conn.execute("SELECT id, dim, vec FROM vectors").fetchall()
            return {
                row["id"]: list(struct.unpack(f"<{row['dim']}f", row["vec"])) for row in rows
            }

        return self._run(op)

    def vector_count(self) -> int:
        def op() -> int:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()
            return int(row["n"]) if row else 0

        return self._run(op)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

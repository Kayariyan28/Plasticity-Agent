"""Tests that the SQLite-backed stores are safe under concurrent threads."""

from __future__ import annotations

import threading

from plasticity_agent.memory.memory_os import MemoryOS


def test_concurrent_record_no_corruption(tmp_path) -> None:
    memory = MemoryOS(memory_dir=str(tmp_path / "m"))
    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(20):
                memory.record(
                    f"memory {worker_id}-{i}", "episodic", check_contradictions=False
                )
        except Exception as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert not errors
        assert memory.count() == 100
    finally:
        memory.close()


def test_concurrent_skill_save_stays_single(tmp_path) -> None:
    memory = MemoryOS(memory_dir=str(tmp_path / "m"))
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for i in range(20):
                memory.skills.save("shared_skill", {"i": i})
        except Exception as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert not errors
        assert memory.skills.count() == 1  # all saves collapse onto one named skill
    finally:
        memory.close()

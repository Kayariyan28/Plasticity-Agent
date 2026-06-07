"""Tests for the Voyager-style skill library."""

from __future__ import annotations

from plasticity_agent.learning.skill_library import SkillLibrary


def _lib(tmp_path) -> SkillLibrary:
    return SkillLibrary(str(tmp_path / "skills.sqlite"))


def test_save_and_find(tmp_path) -> None:
    lib = _lib(tmp_path)
    try:
        lib.save("debug_import_error", {"steps": ["read traceback", "uv add"]})
        found = lib.find("import error")
        assert found
        assert found[0].name == "debug_import_error"
    finally:
        lib.close()


def test_use_increments_usage(tmp_path) -> None:
    lib = _lib(tmp_path)
    try:
        saved = lib.save("skill_a", {})
        used = lib.use("skill_a")
        assert used is not None
        assert used.usage_count == saved.usage_count + 1
    finally:
        lib.close()


def test_resaving_reinforces(tmp_path) -> None:
    lib = _lib(tmp_path)
    try:
        lib.save("s", {}, confidence=0.5)
        again = lib.save("s", {}, confidence=0.7)
        assert again.usage_count >= 1
        assert lib.count() == 1
    finally:
        lib.close()


def test_promote_from_trace_and_list(tmp_path) -> None:
    lib = _lib(tmp_path)
    try:
        lib.promote_from_trace("t", {"task": "x"}, trigger_patterns=["x"])
        assert lib.count() == 1
        assert lib.list_skills()[0].name == "t"
    finally:
        lib.close()


def test_use_unknown_returns_none(tmp_path) -> None:
    lib = _lib(tmp_path)
    try:
        assert lib.use("does_not_exist") is None
    finally:
        lib.close()

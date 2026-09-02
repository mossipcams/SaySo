"""Tests for the Home Assistant compatibility matrix configuration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

try:
    from scripts.compat_matrix import (
        COMPAT_TEST_CATEGORIES,
        COMPAT_TEST_PATHS,
        CURRENT_HA_VERSION,
        DECLARED_MINIMUM_HA_VERSION,
        MATRIX,
        MINIMUM_HA_VERSION,
        get_entry,
        pytest_command,
    )
except ModuleNotFoundError:  # pragma: no cover - import path fallback
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.compat_matrix import (
        COMPAT_TEST_CATEGORIES,
        COMPAT_TEST_PATHS,
        CURRENT_HA_VERSION,
        DECLARED_MINIMUM_HA_VERSION,
        MATRIX,
        MINIMUM_HA_VERSION,
        get_entry,
        pytest_command,
    )


def _read_pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _read_readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_matrix_has_minimum_and_current_entries() -> None:
    """The matrix exercises exactly the minimum and current HA releases."""
    assert len(MATRIX) == 2
    assert get_entry("minimum").homeassistant == MINIMUM_HA_VERSION
    assert get_entry("current").homeassistant == CURRENT_HA_VERSION
    assert MINIMUM_HA_VERSION == "2024.12.0"
    assert CURRENT_HA_VERSION.startswith("2026.8.")


def test_matrix_does_not_use_ha_2025_1_4() -> None:
    """Task 21 explicitly avoids the 2025.1.4 pin."""
    versions = {entry.homeassistant for entry in MATRIX}
    assert "2025.1.4" not in versions


def test_declared_minimum_matches_pyproject() -> None:
    """README/pyproject support floor stays at Home Assistant 2024.12."""
    pyproject = _read_pyproject()
    assert 'homeassistant>=2024.12.0' in pyproject.replace(" ", "")
    assert DECLARED_MINIMUM_HA_VERSION == "2024.12.0"


def test_readme_minimum_matches_pyproject() -> None:
    """README documents the same minimum Home Assistant version as pyproject."""
    readme = _read_readme()
    assert re.search(r"Home Assistant 2024\.12", readme)
    assert "2024.8" not in readme


def test_compat_test_paths_cover_required_categories() -> None:
    """Matrix runs transcript, compiler, boundary, routing, contract, and eval tests."""
    required = {
        "transcript",
        "compiler",
        "boundary",
        "routing",
        "request_contract",
        "offline_eval",
    }
    assert set(COMPAT_TEST_CATEGORIES.values()) == required
    for path in COMPAT_TEST_PATHS:
        assert (ROOT / path).is_file()
        assert path in COMPAT_TEST_CATEGORIES


def test_pytest_command_targets_compat_suite() -> None:
    """Generated pytest command includes every compatibility test module."""
    command = pytest_command(ROOT / ".venv")
    assert command[0].endswith("/pytest") or command[0].endswith("\\pytest")
    assert command[1:] == list(COMPAT_TEST_PATHS)


@pytest.mark.parametrize("entry_id", ["minimum", "current"])
def test_matrix_python_versions(entry_id: str) -> None:
    """Each matrix entry pins a Python version compatible with its HA release."""
    entry = get_entry(entry_id)
    if entry_id == "minimum":
        assert entry.python == "3.12"
        assert entry.homeassistant == "2024.12.0"
    else:
        assert entry.python == "3.14"
        assert entry.homeassistant == CURRENT_HA_VERSION

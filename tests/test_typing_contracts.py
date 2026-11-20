import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_mypy(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", target],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_core_type_checks_under_mypy_strict() -> None:
    result = _run_mypy("custom_components/sayso/core")
    assert result.returncode == 0, result.stdout + result.stderr


def test_integration_type_checks_under_mypy() -> None:
    result = _run_mypy("custom_components/sayso")
    assert result.returncode == 0, result.stdout + result.stderr

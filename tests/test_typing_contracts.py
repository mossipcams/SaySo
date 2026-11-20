import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_type_checks_under_mypy_strict() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "custom_components/sayso/core"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

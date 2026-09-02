"""Home Assistant compatibility matrix for CI and manual verification.

Creates isolated standard virtualenvs with exact resolved Home Assistant versions.
Never modifies the worktree ``.venv``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

ROOT = Path(__file__).resolve().parents[1]

# Exact resolved versions exercised by the matrix (not open ranges).
MINIMUM_HA_VERSION: Final = "2024.12.0"
CURRENT_HA_VERSION: Final = "2026.8.3"

# Declared support floor; must stay aligned with pyproject.toml and README.
DECLARED_MINIMUM_HA_VERSION: Final = "2024.12.0"

# Categories required by Task 21; mapped to existing test modules.
COMPAT_TEST_PATHS: Final[tuple[str, ...]] = (
    "tests/test_conversation.py",  # transcript
    "tests/test_schema.py",  # compiler
    "tests/test_diagnostics.py",  # boundary
    "tests/test_routing.py",  # routing
    "tests/test_client.py",  # request contract
    "tests/test_eval.py",  # offline eval
)

COMPAT_TEST_CATEGORIES: Final[dict[str, str]] = {
    "tests/test_conversation.py": "transcript",
    "tests/test_schema.py": "compiler",
    "tests/test_diagnostics.py": "boundary",
    "tests/test_routing.py": "routing",
    "tests/test_client.py": "request_contract",
    "tests/test_eval.py": "offline_eval",
}


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    """One HA/Python pair in the two-entry compatibility matrix."""

    id: str
    label: str
    homeassistant: str
    python: str


MATRIX: Final[tuple[MatrixEntry, ...]] = (
    MatrixEntry(
        id="minimum",
        label="Home Assistant minimum supported",
        homeassistant=MINIMUM_HA_VERSION,
        python="3.12",
    ),
    MatrixEntry(
        id="current",
        label="Home Assistant current known-good",
        homeassistant=CURRENT_HA_VERSION,
        python="3.14",
    ),
)


def get_entry(entry_id: str) -> MatrixEntry:
    """Return the matrix entry for ``entry_id``."""
    for entry in MATRIX:
        if entry.id == entry_id:
            return entry
    known = ", ".join(item.id for item in MATRIX)
    raise SystemExit(f"Unknown matrix entry {entry_id!r}; expected one of: {known}")


def default_venv_dir(entry_id: str) -> Path:
    """Return an isolated venv path outside the worktree ``.venv``."""
    base = os.environ.get("SAYSO_COMPAT_VENV_ROOT")
    root = Path(base) if base else Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    return root / f"sayso-compat-{entry_id}"


def install_commands(entry: MatrixEntry, venv_dir: Path) -> list[list[str]]:
    """Return shell-safe pip install steps for one matrix entry."""
    pip = venv_dir / "bin" / "pip"
    return [
        [sys.executable, "-m", "venv", str(venv_dir)],
        [str(pip), "install", "--upgrade", "pip"],
        [
            str(pip),
            "install",
            f"homeassistant=={entry.homeassistant}",
            "pytest>=8.0",
            "pytest-asyncio>=0.24",
            "pytest-homeassistant-custom-component>=0.13",
        ],
        [str(pip), "install", "-e", str(ROOT)],
    ]


def pytest_command(venv_dir: Path, test_paths: Sequence[str] | None = None) -> list[str]:
    """Build the pytest invocation for a matrix venv."""
    paths = list(test_paths or COMPAT_TEST_PATHS)
    return [str(venv_dir / "bin" / "pytest"), *paths]


def run_command(command: list[str], *, cwd: Path = ROOT) -> None:
    """Run a command and exit non-zero on failure."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def setup_venv(entry_id: str, venv_dir: Path | None = None) -> Path:
    """Create an isolated venv and install exact resolved dependencies."""
    entry = get_entry(entry_id)
    target = venv_dir or default_venv_dir(entry_id)
    if target.resolve() == (ROOT / ".venv").resolve():
        raise SystemExit("Refusing to install into the worktree .venv")
    for command in install_commands(entry, target):
        run_command(command)
    return target


def run_tests(entry_id: str, venv_dir: Path | None = None) -> None:
    """Run the compatibility test suite for one matrix entry."""
    entry = get_entry(entry_id)
    target = venv_dir or default_venv_dir(entry_id)
    if not (target / "bin" / "pytest").exists():
        target = setup_venv(entry_id, target)
    run_command(pytest_command(target))
    print(
        f"compat matrix passed for {entry.id} "
        f"(homeassistant=={entry.homeassistant}, python {entry.python})",
        flush=True,
    )


def emit_matrix_json() -> None:
    """Print matrix metadata as JSON for CI consumers."""
    payload = {
        "declared_minimum": DECLARED_MINIMUM_HA_VERSION,
        "entries": [asdict(entry) for entry in MATRIX],
        "test_paths": list(COMPAT_TEST_PATHS),
        "test_categories": COMPAT_TEST_CATEGORIES,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Print matrix entries")

    setup = sub.add_parser("setup", help="Create an isolated venv for one entry")
    setup.add_argument("entry", choices=[item.id for item in MATRIX])
    setup.add_argument(
        "--venv-dir",
        type=Path,
        help="Target venv directory (must not be the worktree .venv)",
    )

    run = sub.add_parser("run-tests", help="Run compatibility tests for one entry")
    run.add_argument("entry", choices=[item.id for item in MATRIX])
    run.add_argument(
        "--venv-dir",
        type=Path,
        help="Existing or new venv directory (must not be the worktree .venv)",
    )
    run.add_argument(
        "--setup",
        action="store_true",
        help="Create or refresh the venv before running tests",
    )

    sub.add_parser("matrix-json", help="Emit matrix metadata as JSON")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "list":
        for entry in MATRIX:
            print(
                f"{entry.id}: homeassistant=={entry.homeassistant} "
                f"(python {entry.python}) — {entry.label}"
            )
        return 0

    if args.command == "matrix-json":
        emit_matrix_json()
        return 0

    venv_dir = args.venv_dir
    if venv_dir is not None and venv_dir.resolve() == (ROOT / ".venv").resolve():
        parser.error("Refusing to use the worktree .venv")

    if args.command == "setup":
        setup_venv(args.entry, venv_dir)
        return 0

    if args.command == "run-tests":
        if args.setup or venv_dir is None:
            venv_dir = setup_venv(args.entry, venv_dir)
        run_tests(args.entry, venv_dir)
        return 0

    parser.error(f"Unhandled command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

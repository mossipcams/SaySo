"""CLI filtering is not a third runner."""

from __future__ import annotations

from evals.cli import main


def test_cli_requires_explicit_endpoint_server() -> None:
    assert main(["run", "--suite", "smoke", "--adapter", "endpoint"]) == 2


def test_cli_rejects_in_memory_adapter() -> None:
    assert main(["run", "--suite", "smoke", "--adapter", "in_memory"]) == 2


def test_cli_requires_a_filter() -> None:
    assert main(["run", "--adapter", "endpoint", "--server", "http://127.0.0.1:8080"]) == 2

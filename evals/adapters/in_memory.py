"""Model invocation only: a callable that returns a completion body."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class InMemoryAdapter:
    """Training-checkpoint evaluation. ``complete`` is a local predict function."""

    name = "in_memory"

    def __init__(self, complete: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]) -> None:
        self._complete = complete

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        return self._complete(messages, tools)

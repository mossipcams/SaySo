
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class InMemoryAdapter:

    name = "in_memory"

    def __init__(self, complete: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]) -> None:
        self._complete = complete

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        return self._complete(messages, tools)

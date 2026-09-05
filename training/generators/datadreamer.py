"""Optional DataDreamer integration — never selects labels or tools."""

from __future__ import annotations

from typing import Any


def paraphrase_scenario(scenario: dict[str, Any]) -> str | None:
    """Paraphrase utterance from semantic scenario only. Disabled by default."""
    raise NotImplementedError("DataDreamer paraphrasing is optional and disabled by default")

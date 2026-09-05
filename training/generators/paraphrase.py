"""Optional DataDreamer paraphrasing — loaded only when enabled."""

from __future__ import annotations

from typing import Any, Callable


def load_paraphraser(enabled: bool) -> Callable[[dict[str, Any]], str | None] | None:
    """Return paraphrase function or None. Never selects tools/targets/labels."""
    if not enabled:
        return None
    try:
        from generators.datadreamer import paraphrase_scenario  # noqa: WPS433

        return paraphrase_scenario
    except ImportError:
        return None

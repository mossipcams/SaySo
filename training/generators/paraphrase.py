
from __future__ import annotations

from typing import Any, Callable


def load_paraphraser(enabled: bool) -> Callable[[dict[str, Any]], str | None] | None:
    if not enabled:
        return None
    try:
        from generators.datadreamer import paraphrase_scenario

        return paraphrase_scenario
    except ImportError:
        return None

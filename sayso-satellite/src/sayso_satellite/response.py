"""Render speech returned by Home Assistant Assist."""

from __future__ import annotations

from collections.abc import Callable


def render_assist_response(
    content: str | None,
    sink: Callable[[str], None] | None = None,
) -> str | None:
    """Print Assist speech and return it when present."""

    if content:
        writer = sink or print
        writer(content)
        return content
    return None

__all__ = ["render_assist_response"]

"""Render server response policy on the satellite (print earcon or short text)."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

EARCON_TOKEN = "\a"


class ResponseMode(StrEnum):
    EARCON = "earcon"
    TEXT = "text"


def render_response(
    *,
    mode: str,
    content: str | None,
    sink: Callable[[str], None] | None = None,
) -> str | None:
    """Print earcon or short text. Returns rendered content (None for earcon)."""

    if mode == ResponseMode.EARCON.value:
        token = content or EARCON_TOKEN
        if sink is None:
            print(token, end="", flush=True)
        else:
            sink(token)
        return None
    if content:
        writer = sink or print
        writer(content)
        return content
    return None


def render_text_response_payload(
    payload: dict[str, Any],
    *,
    sink: Callable[[str], None] | None = None,
) -> str | None:
    """Apply response policy fields from a text_response envelope payload."""

    mode = payload.get("response_mode", ResponseMode.TEXT.value)
    content = payload.get("response_content")
    if content is None and mode != ResponseMode.EARCON.value:
        content = payload.get("reason")
    return render_response(mode=mode, content=content, sink=sink)


__all__ = [
    "EARCON_TOKEN",
    "ResponseMode",
    "render_response",
    "render_text_response_payload",
]

"""Where a request comes from and which area it is about.

The model reads this block in every turn, and the training generator and the
offline eval render it through the same functions, so a satellite in the
kitchen looks identical in a training row, an eval case, and a live request.

Pure: no Home Assistant imports. The conversation agent supplies the satellite
area and the area registry; everything else is decided here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

AreaSource = Literal["explicit", "multiple", "satellite", "none"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Home Assistant's Assist API adds one of these sentences itself. SaySo replaces
# them with the structured block below so there is exactly one area format.
_LEGACY_AREA_LINES = (
    re.compile(
        r"^You are in area .+ and all generic commands like 'turn on the lights'"
        r" should target this area\.$"
    ),
    re.compile(
        r"^When a user asks to turn on all devices of a specific type, ask the user"
        r" to specify an area, unless there is only one device of that type\.$"
    ),
    re.compile(r"^area=.*$"),
)


@dataclass(frozen=True, slots=True)
class AreaContext:
    """The requesting satellite's area and the area the request targets."""

    satellite_area: str | None
    target_area: str | None
    target_area_source: AreaSource

    def as_dict(self) -> dict[str, Any]:
        return {
            "satellite_area": self.satellite_area,
            "target_area": self.target_area,
            "target_area_source": self.target_area_source,
        }


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold().replace("'s", ""))


def named_areas(utterance: str, areas: Mapping[str, Iterable[str]]) -> list[str]:
    """Canonical areas the utterance names, by name or alias, in spoken order.

    Longest phrases win, so "living room" is not also read as a "room" alias.
    """
    words = _tokens(utterance)
    phrases = sorted(
        (
            (tuple(_tokens(phrase)), area)
            for area, aliases in areas.items()
            for phrase in {area, *aliases}
            if _tokens(phrase)
        ),
        key=lambda item: -len(item[0]),
    )
    taken: set[int] = set()
    found: list[tuple[int, str]] = []
    for phrase, area in phrases:
        width = len(phrase)
        for start in range(len(words) - width + 1):
            span = set(range(start, start + width))
            if tuple(words[start : start + width]) == phrase and not span & taken:
                taken |= span
                found.append((start, area))
    ordered: list[str] = []
    for _start, area in sorted(found):
        if area not in ordered:
            ordered.append(area)
    return ordered


def build_area_context(
    utterance: str,
    areas: Mapping[str, Iterable[str]],
    satellite_area: str | None,
) -> AreaContext:
    """An explicitly named area overrides the satellite's; otherwise it applies."""
    named = named_areas(utterance, areas)
    if len(named) == 1:
        return AreaContext(satellite_area, named[0], "explicit")
    if named:
        return AreaContext(satellite_area, None, "multiple")
    if satellite_area:
        return AreaContext(satellite_area, satellite_area, "satellite")
    return AreaContext(None, None, "none")


def render_area_context(context: AreaContext) -> str:
    """The block the model reads. ``none`` stands for an absent value."""
    return "\n".join(
        [
            "Area context:",
            f"satellite_area: {context.satellite_area or 'none'}",
            f"target_area: {context.target_area or 'none'}",
            f"target_area_source: {context.target_area_source}",
        ]
    )


def render_system_prompt(system_prompt: str, context: AreaContext) -> str:
    """Drop legacy area sentences and append the structured area block."""
    lines = [
        line
        for line in system_prompt.split("\n")
        if not any(pattern.match(line) for pattern in _LEGACY_AREA_LINES)
    ]
    return "\n".join([*lines, render_area_context(context)])


def apply_area_context(
    messages: list[dict[str, Any]], context: AreaContext
) -> list[dict[str, Any]]:
    """Return messages whose system prompt carries the area block."""
    if messages and messages[0].get("role") == "system":
        first = {
            **messages[0],
            "content": render_system_prompt(messages[0].get("content") or "", context),
        }
        return [first, *messages[1:]]
    return [
        {"role": "system", "content": render_area_context(context)},
        *messages,
    ]

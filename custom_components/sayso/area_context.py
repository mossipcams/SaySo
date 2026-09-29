
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

AreaSource = Literal["explicit", "multiple", "satellite", "none"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

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
    named = named_areas(utterance, areas)
    if len(named) == 1:
        return AreaContext(satellite_area, named[0], "explicit")
    if named:
        return AreaContext(satellite_area, None, "multiple")
    if satellite_area:
        return AreaContext(satellite_area, satellite_area, "satellite")
    return AreaContext(None, None, "none")


def render_area_context(context: AreaContext) -> str:
    return "\n".join(
        [
            "Area context:",
            f"satellite_area: {context.satellite_area or 'none'}",
            f"target_area: {context.target_area or 'none'}",
            f"target_area_source: {context.target_area_source}",
        ]
    )


def render_system_prompt(system_prompt: str, context: AreaContext) -> str:
    lines = [
        line
        for line in system_prompt.split("\n")
        if not any(pattern.match(line) for pattern in _LEGACY_AREA_LINES)
    ]
    return "\n".join([*lines, render_area_context(context)])


def apply_area_context(
    messages: list[dict[str, Any]], context: AreaContext
) -> list[dict[str, Any]]:
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

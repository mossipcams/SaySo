"""Shared area context contract for runtime and training prompts."""

from __future__ import annotations

import re
from typing import Any, Iterable, NamedTuple

EXPLICIT_AREA = "explicit_area"
SATELLITE_FALLBACK = "satellite_fallback"
NAMED_TARGET = "named_target"
AMBIGUOUS = "ambiguous"
MISSING_AREA = "missing_area"


class AreaContext(NamedTuple):
    satellite_area: str | None
    target_area: str | None
    target_area_source: str


def _contains_phrase(text: str, phrase: str) -> bool:
    words = re.findall(r"[^\W_]+", phrase.casefold(), flags=re.UNICODE)
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    return bool(words) and any(
        tokens[i : i + len(words)] == words for i in range(len(tokens))
    )


def _explicit_area_matches(text: str, areas: Iterable[str]) -> list[str]:
    lowered = text.casefold()
    return [
        area for area in areas
        if re.search(
            r"\b(?:in|at|inside|within)\s+(?:the\s+)?"
            + re.escape(area.casefold())
            + r"\b",
            lowered,
        )
    ]


def resolve_area_context(
    utterance: str,
    *,
    satellite_area: str | None,
    areas: Iterable[str] = (),
    entities: Iterable[dict[str, Any]] = (),
) -> AreaContext:
    """Resolve area meaning without guessing between equally valid matches."""
    area_matches = [area for area in areas if _contains_phrase(utterance, area)]
    explicit_matches = _explicit_area_matches(utterance, area_matches)
    if len(explicit_matches) > 1:
        return AreaContext(satellite_area, None, AMBIGUOUS)
    if explicit_matches:
        return AreaContext(satellite_area, explicit_matches[0], EXPLICIT_AREA)

    named = [entity for entity in entities if any(
        _contains_phrase(utterance, str(name))
        for name in (entity.get("name"), *(entity.get("aliases") or ()))
        if name
    )]
    if len(named) > 1:
        return AreaContext(satellite_area, None, AMBIGUOUS)

    if named:
        # Area words inside the matched entity name are naming evidence, not an
        # area request ("turn on Kitchen Light"). A separate room phrase still
        # overrides it ("turn on Kitchen Light in the Bedroom").
        named_area = named[0].get("area")
        area_matches = [area for area in area_matches if area != named_area]
    if len(area_matches) > 1:
        return AreaContext(satellite_area, None, AMBIGUOUS)
    if area_matches:
        return AreaContext(satellite_area, area_matches[0], EXPLICIT_AREA)
    if named:
        return AreaContext(satellite_area, named[0].get("area"), NAMED_TARGET)
    if satellite_area:
        return AreaContext(satellite_area, satellite_area, SATELLITE_FALLBACK)
    return AreaContext(None, None, MISSING_AREA)


def render_area_context(context: AreaContext) -> str:
    """Render the exact three-field block used by runtime and training."""
    return "\n".join(
        (
            f"satellite_area={context.satellite_area or 'None'}",
            f"target_area={context.target_area or 'None'}",
            f"target_area_source={context.target_area_source}",
        )
    )

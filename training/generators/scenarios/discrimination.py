"""Entity-discrimination wording: describe a target without naming it."""

from __future__ import annotations

import re
import random
from typing import Any

from generators.utterances import (
    _BRAND_TOKENS,
    _STOP_TOKENS,
    _plural,
    describe_target,
)

# Capabilities that can yield a description-based row. A description singles out
# one entity from several of the same domain in one area, so it is only
# producible where homes actually place siblings: lights, covers and switches.
DISCRIMINATION_CAPABILITIES: frozenset[str] = frozenset(
    {"lights", "covers", "switches"}
)

# Measured delivery ceiling for description-based rows (2026-09-13).
DISCRIMINATION_DELIVERY_CEILING = 0.02


def call_carries_value(call: dict[str, Any]) -> bool:
    """True when the call needs an argument a description cannot convey."""
    arguments = call.get("arguments") or {}
    value_keys = {
        "brightness", "color", "temperature", "percentage", "volume",
        "position", "mode", "fan_speed", "duration", "seconds", "hours",
        "minutes", "query", "media_id", "search_query",
    }
    return bool(value_keys & set(arguments))


def entity_descriptors(entity: dict[str, Any], area: str | None) -> dict[str, str | None]:
    """Pull genuinely descriptive tokens out of an entity's own name."""
    name = str(entity.get("name") or "")
    area = area or str(entity.get("area") or "")
    tokens = [t for t in re.split(r"[\s\-]+", name) if t]
    area_tokens = [t.lower() for t in re.split(r"[\s\-]+", area) if t]
    while tokens and tokens[0].lower() in area_tokens:
        tokens.pop(0)
    tokens = [
        t for t in tokens
        if t.lower() not in _STOP_TOKENS and not t.lower().endswith("'s")
    ]
    brand = next((t for t in tokens if t.lower() in _BRAND_TOKENS), None)
    modifier = next(
        (t for t in tokens if t.lower() not in _BRAND_TOKENS and len(t) > 2),
        None,
    )
    if modifier and modifier.lower() in {
        _plural(str(entity.get("domain") or "")).lower(),
        str(entity.get("domain") or "").lower(),
    }:
        modifier = None
    return {"modifier": modifier, "brand": brand}


def sibling_entities(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Same-domain, same-area entities other than the target."""
    home = scenario.get("home") or {}
    target = scenario.get("target_entity") or {}
    domain = target.get("domain")
    area = target.get("area")
    if not domain or not area:
        return []
    return [
        entity
        for entity in home.get("entities", [])
        if entity.get("domain") == domain
        and entity.get("area") == area
        and entity.get("entity_id") != target.get("entity_id")
    ]


def uniqueness_safe(
    text: str,
    target: dict[str, Any],
    siblings: list[dict[str, Any]],
) -> bool:
    """Reject a description that any sibling could also satisfy."""
    lowered = text.lower()
    own = {str(target.get("name") or "").lower()}
    own.update(str(a).lower() for a in (target.get("aliases") or ()))
    own.discard("")
    if any(name in lowered for name in own):
        return False
    for sibling in siblings:
        names = {str(sibling.get("name") or "").lower()}
        names.update(str(a).lower() for a in (sibling.get("aliases") or ()))
        names.discard("")
        if any(name in lowered for name in names):
            return False
    return True


def pick_discriminating_description(
    scenario: dict[str, Any],
    rng: random.Random,
) -> str | None:
    """Render a description that can only refer to the target."""
    target = scenario.get("target_entity") or {}
    domain = target.get("domain")
    area = target.get("area")
    if not domain or not area or not target.get("name"):
        return None
    siblings = sibling_entities(scenario)
    if not siblings:
        return None
    for _ in range(24):
        descriptors = entity_descriptors(target, area)
        text = describe_target(
            domain, area, rng,
            modifier=descriptors["modifier"],
            brand=descriptors["brand"],
        )
        if text and uniqueness_safe(text, target, siblings):
            mod = descriptors["modifier"]
            if mod and any(
                mod.lower() in str(s.get("name") or "").lower() for s in siblings
            ):
                continue
            return text
    return None

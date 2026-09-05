"""Context serialization matching inference format."""

from __future__ import annotations

import json
from typing import Any


def serialize_context(home: dict[str, Any]) -> str:
    """Serialize exposed entity context exactly as inference does."""
    context = [
        {
            "name": entity["name"],
            "aliases": entity.get("aliases", []),
            "domain": entity["domain"],
            "device_class": entity.get("device_class"),
            "area": entity["area"],
            "floor": entity.get("floor"),
            "state": entity["state"],
            "capabilities": entity.get("capabilities", []),
        }
        for entity in home.get("entities", [])
    ]
    sayso_area = home.get("sayso_entity_area", "")
    return (
        "You are SaySo, a concise Home Assistant conversation agent. Use only the supplied "
        "Home Assistant tools and preserve canonical entity names exactly. "
        f"This SaySo conversation entity area is {sayso_area!r}. "
        "Current exposed context: "
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )


def system_prompt(home: dict[str, Any]) -> str:
    return serialize_context(home)

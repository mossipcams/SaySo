"""Map SaySo semantic actions to Home Assistant service calls."""

from __future__ import annotations

from typing import Any

from .const import ACTION_PAYLOAD_BRIGHTNESS, ACTION_PAYLOAD_TEMPERATURE

_POWER_SERVICES: dict[str, str] = {
    "on": "turn_on",
    "off": "turn_off",
    "toggle": "toggle",
}


def map_action_to_ha_service(
    *,
    entity_id: str,
    entity_domain: str,
    action: str,
    request: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Return the Home Assistant domain, service, and data for a semantic action."""

    if action in _POWER_SERVICES:
        return (
            entity_domain,
            _POWER_SERVICES[action],
            {"entity_id": entity_id},
        )

    if action == "set_brightness":
        return (
            entity_domain,
            "turn_on",
            {
                "entity_id": entity_id,
                "brightness_pct": request[ACTION_PAYLOAD_BRIGHTNESS],
            },
        )

    if action == "set_temperature":
        return (
            entity_domain,
            "set_temperature",
            {
                "entity_id": entity_id,
                "temperature": request[ACTION_PAYLOAD_TEMPERATURE],
            },
        )

    if action == "scene":
        return ("scene", "turn_on", {"entity_id": entity_id})

    if action == "script":
        return ("script", "turn_on", {"entity_id": entity_id})

    msg = f"unsupported action: {action}"
    raise ValueError(msg)

"""Conservative Home Graph capability mapping for Home Assistant entities."""

from __future__ import annotations

from typing import Any

from homeassistant.components.climate import ClimateEntityFeature
from homeassistant.components.light import ColorMode
from homeassistant.core import HomeAssistant, State

from .const import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    CLIMATE_QUERY_ATTRIBUTES,
    POWER_DOMAINS,
    QUERY_ONLY_DOMAINS,
    SENSOR_QUERY_ATTRIBUTES,
)

_ACTION_CAPABILITY_KINDS: dict[str, str] = {
    "on": "power",
    "off": "power",
    "toggle": "power",
    "set_brightness": "brightness",
    "set_temperature": "temperature",
    "query": "query",
    "scene": "scene",
    "script": "script",
}

_ONOFF_COLOR_MODES = frozenset({ColorMode.ONOFF, "onoff"})


def entity_capabilities(
    domain: str,
    attributes: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Map a Home Assistant entity domain/state to executable capabilities."""

    attrs = attributes or {}

    if domain in QUERY_ONLY_DOMAINS:
        return [_query_capability(_present_attributes(attrs, SENSOR_QUERY_ATTRIBUTES))]

    if domain not in POWER_DOMAINS and domain != "climate":
        return []

    capabilities: list[dict[str, Any]] = [{"kind": "power"}]

    if domain == "light" and _supports_brightness(attrs):
        capabilities.append(
            {
                "kind": "brightness",
                "min_value": BRIGHTNESS_MIN,
                "max_value": BRIGHTNESS_MAX,
            },
        )

    if domain == "climate":
        if _supports_temperature(attrs):
            temperature: dict[str, Any] = {"kind": "temperature"}
            if "min_temp" in attrs:
                temperature["min_value"] = float(attrs["min_temp"])
            if "max_temp" in attrs:
                temperature["max_value"] = float(attrs["max_temp"])
            capabilities.append(temperature)
        capabilities.append(
            _query_capability(_present_attributes(attrs, CLIMATE_QUERY_ATTRIBUTES)),
        )

    return capabilities


def entity_capabilities_from_state(domain: str, state: State | None) -> list[dict[str, Any]]:
    """Map capabilities from a live Home Assistant state object."""

    if state is None:
        return entity_capabilities(domain, None)
    return entity_capabilities(domain, dict(state.attributes))


def action_capability_kind(action: str) -> str | None:
    """Return the executable capability kind required by a SaySo action."""

    return _ACTION_CAPABILITY_KINDS.get(action)


def entity_capabilities_for_entity(
    hass: HomeAssistant,
    entity_id: str,
    domain: str,
) -> list[dict[str, Any]]:
    """Return exposed executable capabilities for a concrete entity."""

    if domain == "scene":
        return scene_capabilities()
    if domain == "script":
        return script_capabilities()
    return entity_capabilities_from_state(domain, hass.states.get(entity_id))


def scene_capabilities() -> list[dict[str, Any]]:
    return [{"kind": "scene"}]


def script_capabilities() -> list[dict[str, Any]]:
    return [{"kind": "script"}]


def _supports_brightness(attributes: dict[str, Any]) -> bool:
    modes = attributes.get("supported_color_modes")
    if not modes:
        return False
    return any(mode not in _ONOFF_COLOR_MODES for mode in modes)


def _supports_temperature(attributes: dict[str, Any]) -> bool:
    supported_features = int(attributes.get("supported_features", 0))
    if supported_features & (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
    ):
        return True
    return "min_temp" in attributes and "max_temp" in attributes


def _present_attributes(
    attributes: dict[str, Any],
    candidates: tuple[str, ...],
) -> list[str]:
    return [name for name in candidates if name in attributes]


def _query_capability(attributes: list[str]) -> dict[str, Any]:
    if attributes:
        return {"kind": "query", "attributes": attributes}
    return {"kind": "query"}

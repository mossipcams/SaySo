"""Conservative Home Graph capability mapping tests."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.components.light import ColorMode

from custom_components.sayso.capabilities import (
    entity_capabilities,
    scene_capabilities,
    script_capabilities,
)


@pytest.mark.parametrize(
    ("domain", "attributes", "expected_kinds"),
    [
        (
            "light",
            {
                "brightness": 200,
                "supported_color_modes": [ColorMode.BRIGHTNESS],
                "color_mode": "brightness",
                "rgb_color": [255, 0, 0],
                "effect": "rainbow",
            },
            ["power", "brightness"],
        ),
        (
            "light",
            {"brightness": 200, "supported_color_modes": [ColorMode.ONOFF]},
            ["power"],
        ),
        ("light", {"brightness": 200}, ["power"]),
        ("switch", {}, ["power"]),
        ("sensor", {"unit_of_measurement": "°F"}, ["query"]),
        ("binary_sensor", {"device_class": "door"}, ["query"]),
        (
            "climate",
            {
                "hvac_mode": "heat",
                "temperature": 72,
                "current_temperature": 70,
                "min_temp": 50,
                "max_temp": 90,
                "supported_features": 1,
            },
            ["power", "temperature", "query"],
        ),
    ],
)
def test_entity_capabilities_expose_only_executable_kinds(
    domain: str,
    attributes: dict,
    expected_kinds: list[str],
) -> None:
    """Only executable capabilities are exposed for each domain."""

    caps = entity_capabilities(domain, attributes)
    assert [cap["kind"] for cap in caps] == expected_kinds


def test_entity_capabilities_ignore_unsupported_attributes() -> None:
    """Color, effects, and other extras must not become capabilities."""

    caps = entity_capabilities(
        "light",
        {
            "brightness": 128,
            "supported_color_modes": [ColorMode.HS],
            "color_mode": "hs",
            "hs_color": [240, 100],
            "rgb_color": [0, 0, 255],
            "xy_color": [0.1, 0.2],
            "effect": "colorloop",
            "effect_list": ["none", "colorloop"],
        },
    )

    kinds = {cap["kind"] for cap in caps}
    assert kinds == {"power", "brightness"}
    for cap in caps:
        assert "attributes" not in cap or cap.get("attributes") is None


def test_brightness_capability_uses_percentage_range() -> None:
    caps = entity_capabilities(
        "light",
        {"supported_color_modes": [ColorMode.BRIGHTNESS]},
    )
    brightness = next(cap for cap in caps if cap["kind"] == "brightness")
    assert brightness["min_value"] == 1
    assert brightness["max_value"] == 100


def test_scene_and_script_capabilities() -> None:
    assert scene_capabilities() == [{"kind": "scene"}]
    assert script_capabilities() == [{"kind": "script"}]


@pytest.mark.asyncio
async def test_registry_entities_get_expected_capabilities(hass: HomeAssistant) -> None:
    """Registry-backed entities map to the conservative capability set."""

    entity_reg = er.async_get(hass)

    kitchen = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    hass.states.async_set(
        kitchen.entity_id,
        "on",
        {"brightness": 200},
    )

    garage_sensor = entity_reg.async_get_or_create(
        "sensor",
        "test",
        "garage_temp",
        suggested_object_id="garage_temp",
    )
    hass.states.async_set(
        garage_sensor.entity_id,
        "72",
        {"unit_of_measurement": "°F"},
    )

    outlet = entity_reg.async_get_or_create(
        "switch",
        "test",
        "outlet",
        suggested_object_id="outlet",
    )
    hass.states.async_set(outlet.entity_id, "off")

    assert entity_capabilities("light", hass.states.get(kitchen.entity_id).attributes) == [
        {"kind": "power"},
    ]
    assert entity_capabilities(
        "sensor",
        hass.states.get(garage_sensor.entity_id).attributes,
    ) == [{"kind": "query", "attributes": ["unit_of_measurement"]}]
    assert entity_capabilities("switch", hass.states.get(outlet.entity_id).attributes) == [
        {"kind": "power"},
    ]

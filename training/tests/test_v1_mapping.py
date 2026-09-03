"""Tests for v1 service mapping from Home-LLM piles."""

from __future__ import annotations

import pytest

from generators.v1_map import (
    build_tool_call,
    drop_reason_for_service,
    is_mappable_service,
    is_mappable_status_device,
)


@pytest.mark.parametrize(
    "service",
    [
        "climate.turn_on",
        "vacuum.return_to_base",
        "media_player.volume_up",
        "timer.start",
        "todo.add_item",
        "humidifier.set_humidity",
    ],
)
def test_non_v1_services_are_dropped(service: str) -> None:
    assert is_mappable_service(service) is False
    assert drop_reason_for_service(service) is not None


@pytest.mark.parametrize(
    "service",
    [
        "light.turn_on",
        "light.turn_off",
        "fan.increase_speed",
        "blinds.open_cover",
        "garage_door.close_cover",
        "lock.lock",
        "switch.turn_on",
    ],
)
def test_v1_capable_services_map(service: str) -> None:
    assert is_mappable_service(service) is True
    call = build_tool_call(service, "Kitchen Light")
    assert call is not None
    assert call.name in {
        "HassTurnOn",
        "HassTurnOff",
        "HassFanSetSpeed",
        "HassLightSet",
    }
    assert "entity_id" not in call.arguments
    assert "service" not in call.arguments


def test_brightness_templates_map_to_hass_light_set() -> None:
    call = build_tool_call(
        "light.turn_on",
        "Desk Lamp",
        has_brightness=True,
        brightness=40,
    )
    assert call is not None
    assert call.name == "HassLightSet"
    assert call.arguments["brightness"] == 40


def test_toggle_actions_are_dropped() -> None:
    assert is_mappable_service("light.toggle") is False


def test_status_device_types() -> None:
    assert is_mappable_status_device("light")
    assert is_mappable_status_device("fan")
    assert not is_mappable_status_device("climate")
    assert not is_mappable_status_device("media_player")

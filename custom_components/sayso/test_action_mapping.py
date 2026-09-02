"""Typed semantic action → Home Assistant service mapping tests."""

from __future__ import annotations

import json
from typing import Any

import pytest
from homeassistant.components.light import ColorMode
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.action_mapping import map_action_to_ha_service
from custom_components.sayso.const import (
    ACTION_PAYLOAD_BRIGHTNESS,
    ACTION_PAYLOAD_TEMPERATURE,
    CONF_ACTION_ALLOWLIST,
    CONF_DOMAIN_ALLOWLIST,
    CONF_ENTITY_IDS,
    CONF_EXPOSURE_MODE,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    EXPOSURE_MODE_ENTITY,
)
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import SaySoConnectionCoordinator
from custom_components.sayso.test_coordinator import fast_timing
from custom_components.sayso.test_deltas import _wait_until


@pytest.mark.parametrize(
    ("action", "service"),
    [
        ("on", "turn_on"),
        ("off", "turn_off"),
        ("toggle", "toggle"),
    ],
)
def test_power_actions_map_to_turn_services(action: str, service: str) -> None:
    ha_domain, ha_service, data = map_action_to_ha_service(
        entity_id="light.kitchen",
        entity_domain="light",
        action=action,
        request={"action": action},
    )

    assert ha_domain == "light"
    assert ha_service == service
    assert data == {"entity_id": "light.kitchen"}


def test_set_brightness_maps_to_light_turn_on_with_brightness_pct() -> None:
    ha_domain, ha_service, data = map_action_to_ha_service(
        entity_id="light.kitchen",
        entity_domain="light",
        action="set_brightness",
        request={ACTION_PAYLOAD_BRIGHTNESS: 40},
    )

    assert ha_domain == "light"
    assert ha_service == "turn_on"
    assert data == {"entity_id": "light.kitchen", "brightness_pct": 40}


def test_set_temperature_maps_to_climate_set_temperature() -> None:
    ha_domain, ha_service, data = map_action_to_ha_service(
        entity_id="climate.living_room",
        entity_domain="climate",
        action="set_temperature",
        request={ACTION_PAYLOAD_TEMPERATURE: 72.0},
    )

    assert ha_domain == "climate"
    assert ha_service == "set_temperature"
    assert data == {"entity_id": "climate.living_room", "temperature": 72.0}


def test_scene_action_maps_to_scene_turn_on() -> None:
    ha_domain, ha_service, data = map_action_to_ha_service(
        entity_id="scene.movie_time",
        entity_domain="scene",
        action="scene",
        request={"action": "scene"},
    )

    assert ha_domain == "scene"
    assert ha_service == "turn_on"
    assert data == {"entity_id": "scene.movie_time"}


def test_script_action_maps_to_script_turn_on() -> None:
    ha_domain, ha_service, data = map_action_to_ha_service(
        entity_id="script.good_morning",
        entity_domain="script",
        action="script",
        request={"action": "script"},
    )

    assert ha_domain == "script"
    assert ha_service == "turn_on"
    assert data == {"entity_id": "script.good_morning"}


def _action_request(
    *,
    request_id: str = "req-1",
    entity_id: str,
    domain: str,
    action: str,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "entity_id": entity_id,
        "domain": domain,
        "action": action,
    }
    if extra:
        payload.update(extra)
    return json.dumps(
        {
            "version": 1,
            "type": "action_request",
            "correlation_id": request_id,
            "payload": payload,
        },
    )


class _RecordingServiceCaller:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    async def __call__(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.calls.append((domain, service, data))


@pytest.mark.asyncio
async def test_coordinator_dispatches_set_brightness_with_exact_payload(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["light"],
            CONF_ACTION_ALLOWLIST: ["set_brightness"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_ENTITY_IDS: ["light.kitchen"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    hass.states.async_set(
        light.entity_id,
        "on",
        {"supported_color_modes": [ColorMode.BRIGHTNESS]},
    )
    await hass.async_block_till_done()

    fake_ws = FakeWebSocket()
    service_caller = _RecordingServiceCaller()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(
        hass,
        entry,
        ws_connect=connect,
        service_caller=service_caller,
    )
    await coordinator.async_start()
    await _wait_until(lambda: coordinator.connected, timeout=2.0)

    coordinator._conversation_contexts["req-1"] = Context()

    fake_ws.push(
        _action_request(
            entity_id=light.entity_id,
            domain="light",
            action="set_brightness",
            extra={ACTION_PAYLOAD_BRIGHTNESS: 25},
        ),
    )

    await _wait_until(lambda: service_caller.calls, timeout=2.0)

    assert service_caller.calls == [
        ("light", "turn_on", {"entity_id": light.entity_id, "brightness_pct": 25}),
    ]

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_coordinator_dispatches_set_temperature_with_exact_payload(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["climate"],
            CONF_ACTION_ALLOWLIST: ["set_temperature"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_ENTITY_IDS: ["climate.living_room"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    climate = entity_reg.async_get_or_create(
        "climate",
        "test",
        "living_room",
        suggested_object_id="living_room",
    )
    hass.states.async_set(
        climate.entity_id,
        "heat",
        {
            "temperature": 70,
            "min_temp": 50,
            "max_temp": 90,
            "supported_features": 1,
        },
    )
    await hass.async_block_till_done()

    fake_ws = FakeWebSocket()
    service_caller = _RecordingServiceCaller()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(
        hass,
        entry,
        ws_connect=connect,
        service_caller=service_caller,
    )
    await coordinator.async_start()
    await _wait_until(lambda: coordinator.connected, timeout=2.0)

    coordinator._conversation_contexts["req-1"] = Context()

    fake_ws.push(
        _action_request(
            entity_id=climate.entity_id,
            domain="climate",
            action="set_temperature",
            extra={ACTION_PAYLOAD_TEMPERATURE: 68.5},
        ),
    )

    await _wait_until(lambda: service_caller.calls, timeout=2.0)

    assert service_caller.calls == [
        (
            "climate",
            "set_temperature",
            {"entity_id": climate.entity_id, "temperature": 68.5},
        ),
    ]

    await coordinator.async_stop()

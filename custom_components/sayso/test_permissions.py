"""Action permission enforcement tests for SaySo."""

from __future__ import annotations

import json
from typing import Any

import pytest
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import (
    CONF_ACTION_ALLOWLIST,
    CONF_AREA_IDS,
    CONF_DOMAIN_ALLOWLIST,
    CONF_ENTITY_IDS,
    CONF_EXPOSURE_MODE,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    EXPOSURE_MODE_AREA,
    EXPOSURE_MODE_ENTITY,
)
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import SaySoConnectionCoordinator
from custom_components.sayso.test_coordinator import fast_timing
from custom_components.sayso.test_deltas import _wait_until


def _action_request(
    *,
    request_id: str = "req-1",
    entity_id: str,
    domain: str,
    action: str,
) -> str:
    return json.dumps(
        {
            "version": 1,
            "type": "action_request",
            "correlation_id": request_id,
            "payload": {
                "request_id": request_id,
                "entity_id": entity_id,
                "domain": domain,
                "action": action,
            },
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
async def test_mismatched_domain_never_calls_home_assistant(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Spoofed payload domain must not bypass allowlists keyed on entity_id."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["light"],
            CONF_ACTION_ALLOWLIST: ["on", "off"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_AREA_IDS: [],
            CONF_ENTITY_IDS: ["switch.desk"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    switch = entity_reg.async_get_or_create(
        "switch",
        "test",
        "desk",
        suggested_object_id="desk",
    )
    hass.states.async_set(switch.entity_id, "off")
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
            entity_id=switch.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(
        lambda: any(
            json.loads(message).get("type") == "action_result"
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    assert service_caller.calls == []
    results = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message).get("type") == "action_result"
    ]
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "domain_mismatch"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_disallowed_domain_never_calls_home_assistant(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Domain allowlist violations must reject before any HA service call."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["switch"],
            CONF_ACTION_ALLOWLIST: ["on", "off"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_AREA_IDS: [],
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
    hass.states.async_set(light.entity_id, "off")
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
            action="on",
        ),
    )

    await _wait_until(
        lambda: any(
            json.loads(message).get("type") == "action_result"
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    assert service_caller.calls == []
    results = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message).get("type") == "action_result"
    ]
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "domain_not_allowed"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_disallowed_action_never_calls_home_assistant(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Action allowlist violations must reject before any HA service call."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["light"],
            CONF_ACTION_ALLOWLIST: ["off"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_AREA_IDS: [],
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
    hass.states.async_set(light.entity_id, "off")
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
            action="on",
        ),
    )

    await _wait_until(
        lambda: any(
            json.loads(message).get("type") == "action_result"
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    assert service_caller.calls == []
    results = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message).get("type") == "action_result"
    ]
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "action_not_allowed"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_unexposed_entity_never_calls_home_assistant(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Hidden entities must reject before any HA service call."""

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create("Living Room")
    garage = area_reg.async_create("Garage")

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["light"],
            CONF_ACTION_ALLOWLIST: ["on"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_AREA,
            CONF_AREA_IDS: [living_room.id],
            CONF_ENTITY_IDS: [],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    hidden_light = entity_reg.async_get_or_create(
        "light",
        "test",
        "garage",
        suggested_object_id="garage",
    )
    entity_reg.async_update_entity(hidden_light.entity_id, area_id=garage.id)

    exposed_light = entity_reg.async_get_or_create(
        "light",
        "test",
        "living",
        suggested_object_id="living",
    )
    entity_reg.async_update_entity(exposed_light.entity_id, area_id=living_room.id)

    hass.states.async_set(hidden_light.entity_id, "off")
    hass.states.async_set(exposed_light.entity_id, "off")
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
            entity_id=hidden_light.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(
        lambda: any(
            json.loads(message).get("type") == "action_result"
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    assert service_caller.calls == []
    results = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message).get("type") == "action_result"
    ]
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "entity_not_exposed"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_unsupported_capability_never_calls_home_assistant(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Actions requiring missing capabilities must reject before service execution."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["switch"],
            CONF_ACTION_ALLOWLIST: ["set_brightness"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_AREA_IDS: [],
            CONF_ENTITY_IDS: ["switch.desk"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    switch = entity_reg.async_get_or_create(
        "switch",
        "test",
        "desk",
        suggested_object_id="desk",
    )
    hass.states.async_set(switch.entity_id, "off")
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
            entity_id=switch.entity_id,
            domain="switch",
            action="set_brightness",
        ),
    )

    await _wait_until(
        lambda: any(
            json.loads(message).get("type") == "action_result"
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    assert service_caller.calls == []
    results = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message).get("type") == "action_result"
    ]
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "capability_not_supported"

    await coordinator.async_stop()

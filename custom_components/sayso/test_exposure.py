"""Exposure enforcement tests for SaySo Home Graph payloads."""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import (
    CONF_AREA_IDS,
    CONF_ENTITY_IDS,
    CONF_EXPOSURE_MODE,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    EXPOSURE_MODE_AREA,
    EXPOSURE_MODE_ENTITY,
    get_entry_options,
)
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import SaySoConnectionCoordinator
from custom_components.sayso.snapshot import build_home_graph_snapshot
from custom_components.sayso.test_deltas import _wait_until


@pytest.mark.asyncio
async def test_area_exposure_excludes_entities_outside_selected_areas_from_snapshot(
    hass: HomeAssistant,
) -> None:
    """Entities outside selected areas must not appear in graph snapshots."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret"},
        options={
            "domain_allowlist": [],
            "action_allowlist": [],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_AREA,
            CONF_AREA_IDS: ["living_room"],
            CONF_ENTITY_IDS: [],
        },
    )
    entry.add_to_hass(hass)

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create("Living Room")
    garage = area_reg.async_create("Garage")

    entity_reg = er.async_get(hass)
    exposed_light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
        original_name="Kitchen Light",
    )
    entity_reg.async_update_entity(exposed_light.entity_id, area_id=living_room.id)

    hidden_sensor = entity_reg.async_get_or_create(
        "sensor",
        "test",
        "garage_temp",
        suggested_object_id="garage_temp",
        original_name="Garage Temperature",
    )
    entity_reg.async_update_entity(hidden_sensor.entity_id, area_id=garage.id)

    hass.states.async_set(exposed_light.entity_id, "on")
    hass.states.async_set(hidden_sensor.entity_id, "72")

    snapshot = build_home_graph_snapshot(
        hass,
        home_id="exposure-test-home",
        sequence=1,
        options=get_entry_options(entry),
    )

    entity_ids = {entity["entity_id"] for entity in snapshot["entities"]}
    assert exposed_light.entity_id in entity_ids
    assert hidden_sensor.entity_id not in entity_ids


@pytest.mark.asyncio
async def test_entity_exposure_includes_only_selected_entities_in_snapshot(
    hass: HomeAssistant,
) -> None:
    """Entity mode must expose only explicitly selected entity ids."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret"},
        options={
            "domain_allowlist": [],
            "action_allowlist": [],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_AREA_IDS: [],
            CONF_ENTITY_IDS: ["light.kitchen"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    exposed_light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    hidden_switch = entity_reg.async_get_or_create(
        "switch",
        "test",
        "desk",
        suggested_object_id="desk",
    )

    hass.states.async_set(exposed_light.entity_id, "on")
    hass.states.async_set(hidden_switch.entity_id, "off")

    snapshot = build_home_graph_snapshot(
        hass,
        home_id="exposure-test-home",
        sequence=1,
        options=get_entry_options(entry),
    )

    entity_ids = {entity["entity_id"] for entity in snapshot["entities"]}
    assert entity_ids == {exposed_light.entity_id}


@pytest_asyncio.fixture
async def area_filtered_coordinator(
    hass: HomeAssistant,
) -> tuple[SaySoConnectionCoordinator, FakeWebSocket, MockConfigEntry]:
    """Coordinator with area exposure limited to living_room."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            "domain_allowlist": [],
            "action_allowlist": [],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_AREA,
            CONF_AREA_IDS: ["living_room"],
            CONF_ENTITY_IDS: [],
        },
    )
    entry.add_to_hass(hass)

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create("Living Room")
    garage = area_reg.async_create("Garage")

    entity_reg = er.async_get(hass)
    exposed_light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    entity_reg.async_update_entity(exposed_light.entity_id, area_id=living_room.id)

    hidden_sensor = entity_reg.async_get_or_create(
        "sensor",
        "test",
        "garage_temp",
        suggested_object_id="garage_temp",
    )
    entity_reg.async_update_entity(hidden_sensor.entity_id, area_id=garage.id)

    hass.states.async_set(exposed_light.entity_id, "off")
    hass.states.async_set(hidden_sensor.entity_id, "72")
    await hass.async_block_till_done()

    fake_ws = FakeWebSocket()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    await _wait_until(
        lambda: coordinator.connected
        and any(json.loads(message)["type"] == "graph_snapshot" for message in fake_ws.sent),
        timeout=2.0,
    )

    yield coordinator, fake_ws, entry
    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_unexposed_entity_state_change_does_not_send_delta(
    hass: HomeAssistant,
    area_filtered_coordinator,
) -> None:
    """State changes for hidden entities must not emit state deltas."""

    _coordinator, fake_ws, _entry = area_filtered_coordinator

    entity_reg = er.async_get(hass)
    hidden_sensor = entity_reg.async_get("sensor.garage_temp")
    assert hidden_sensor is not None

    sent_before = len(fake_ws.sent)
    hass.states.async_set(hidden_sensor.entity_id, "68")
    await hass.async_block_till_done()
    await asyncio.sleep(0.05)

    new_messages = [
        json.loads(message)
        for message in fake_ws.sent[sent_before:]
        if json.loads(message)["type"] == "state_delta"
        and json.loads(message)["payload"].get("entity_id") == hidden_sensor.entity_id
    ]
    assert new_messages == []


@pytest.mark.asyncio
async def test_unexposed_registry_update_does_not_send_delta(
    hass: HomeAssistant,
    area_filtered_coordinator,
) -> None:
    """Registry updates for hidden entities must not emit registry deltas."""

    _coordinator, fake_ws, _entry = area_filtered_coordinator

    entity_reg = er.async_get(hass)
    hidden_sensor = entity_reg.async_get("sensor.garage_temp")
    assert hidden_sensor is not None

    sent_before = len(fake_ws.sent)
    entity_reg.async_update_entity(hidden_sensor.entity_id, aliases=["garage probe"])
    await hass.async_block_till_done()

    await _wait_until(lambda: len(fake_ws.sent) >= sent_before, timeout=0.5)

    new_messages = [
        json.loads(message)
        for message in fake_ws.sent[sent_before:]
        if json.loads(message)["type"] == "registry_delta"
        and json.loads(message)["payload"].get("entity_id") == hidden_sensor.entity_id
    ]
    assert new_messages == []

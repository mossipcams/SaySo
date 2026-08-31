"""Incremental Home Graph delta tests for SaySo."""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import SaySoConnectionCoordinator
from custom_components.sayso.test_coordinator import fast_timing


def _parse_sent(messages: list[str]) -> list[dict]:
    return [json.loads(message) for message in messages]


async def _wait_until(
    predicate,
    *,
    timeout: float = 1.0,
    interval: float = 0.01,
) -> None:
    for _ in range(int(timeout / interval)):
        if predicate():
            return
        await asyncio.sleep(interval)
    pytest.fail("condition not met before timeout")


@pytest_asyncio.fixture
async def connected_coordinator(
    hass: HomeAssistant,
    fast_timing,
) -> tuple[SaySoConnectionCoordinator, FakeWebSocket, MockConfigEntry]:
    """Start a coordinator with a fake socket and wait until the graph snapshot is sent."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    def connected_with_snapshot() -> bool:
        if not coordinator.connected:
            return False
        return any(
            json.loads(message)["type"] == "graph_snapshot"
            for message in fake_ws.sent
        )

    await _wait_until(connected_with_snapshot, timeout=2.0)
    yield coordinator, fake_ws, entry
    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_state_change_sends_state_delta_not_snapshot(
    hass: HomeAssistant,
    connected_coordinator,
) -> None:
    """One HA state change must emit a state delta, not another full snapshot."""

    _coordinator, fake_ws, entry = connected_coordinator
    home_id = entry.entry_id

    entity_reg = er.async_get(hass)
    light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
        original_name="Kitchen Light",
    )
    hass.states.async_set(light.entity_id, "off")
    await hass.async_block_till_done()

    await _wait_until(
        lambda: any(
            json.loads(message)["type"] == "state_delta"
            and json.loads(message)["payload"].get("entity_id") == light.entity_id
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    post_connect = _parse_sent(fake_ws.sent)
    snapshot_count = sum(1 for message in post_connect if message["type"] == "graph_snapshot")
    state_deltas = [
        message
        for message in post_connect
        if message["type"] == "state_delta"
        and message["payload"].get("entity_id") == light.entity_id
    ]

    assert snapshot_count == 1
    assert len(state_deltas) == 1
    payload = state_deltas[0]["payload"]
    assert payload["home_id"] == home_id
    assert payload["entity_id"] == light.entity_id
    assert payload["state"]["value"] == "off"
    assert "sequence" in payload


@pytest.mark.asyncio
async def test_registry_change_sends_registry_delta_not_snapshot(
    hass: HomeAssistant,
    connected_coordinator,
) -> None:
    """Entity registry updates must emit registry deltas, not full snapshots."""

    _coordinator, fake_ws, entry = connected_coordinator
    home_id = entry.entry_id

    entity_reg = er.async_get(hass)
    light = entity_reg.async_get_or_create(
        "light",
        "test",
        "pantry",
        suggested_object_id="pantry",
        original_name="Pantry Light",
    )
    hass.states.async_set(light.entity_id, "on")
    await hass.async_block_till_done()

    await _wait_until(
        lambda: any(
            json.loads(message)["type"] == "state_delta"
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    sent_before = len(fake_ws.sent)
    entity_reg.async_update_entity(light.entity_id, aliases=["pantry lamp"])
    await hass.async_block_till_done()

    await _wait_until(lambda: len(fake_ws.sent) > sent_before, timeout=2.0)

    new_messages = _parse_sent(fake_ws.sent[sent_before:])
    assert new_messages, "expected a registry delta after alias update"
    assert all(message["type"] != "graph_snapshot" for message in new_messages)

    registry_deltas = [message for message in new_messages if message["type"] == "registry_delta"]
    assert len(registry_deltas) == 1
    payload = registry_deltas[0]["payload"]
    assert payload["home_id"] == home_id
    assert payload["entity_id"] == light.entity_id
    assert payload["change"] == "update"
    assert payload["entity"]["entity_id"] == light.entity_id
    assert "pantry lamp" in payload["entity"]["aliases"]
    assert "sequence" in payload


@pytest.mark.asyncio
async def test_one_entity_state_change_yields_exactly_one_delta(
    hass: HomeAssistant,
    connected_coordinator,
) -> None:
    """A single entity state transition produces one delta with matching identity."""

    _coordinator, fake_ws, entry = connected_coordinator
    home_id = entry.entry_id

    entity_reg = er.async_get(hass)
    switch = entity_reg.async_get_or_create(
        "switch",
        "test",
        "desk",
        suggested_object_id="desk",
        original_name="Desk Switch",
    )
    hass.states.async_set(switch.entity_id, "off")
    await hass.async_block_till_done()

    await _wait_until(
        lambda: any(
            json.loads(message)["type"] == "state_delta"
            and json.loads(message)["payload"].get("entity_id") == switch.entity_id
            for message in fake_ws.sent
        ),
        timeout=2.0,
    )

    sent_before = len(fake_ws.sent)

    hass.states.async_set(switch.entity_id, "on", {"friendly_name": "Desk Switch"})
    await hass.async_block_till_done()

    await _wait_until(
        lambda: len(fake_ws.sent) > sent_before
        and any(
            message["type"] == "state_delta"
            and message["payload"].get("entity_id") == switch.entity_id
            and message["payload"].get("state", {}).get("value") == "on"
            for message in _parse_sent(fake_ws.sent[sent_before:])
        ),
        timeout=2.0,
    )

    new_deltas = [
        message
        for message in _parse_sent(fake_ws.sent[sent_before:])
        if message["type"] == "state_delta"
        and message["payload"].get("entity_id") == switch.entity_id
    ]
    assert len(new_deltas) == 1

    matching = [
        message
        for message in _parse_sent(fake_ws.sent)
        if message["type"] == "state_delta"
        and message["payload"].get("entity_id") == switch.entity_id
        and message["payload"].get("state", {}).get("value") == "on"
    ]
    assert len(matching) == 1
    payload = matching[0]["payload"]
    assert payload["home_id"] == home_id
    assert payload["entity_id"] == switch.entity_id
    assert payload["state"]["value"] == "on"

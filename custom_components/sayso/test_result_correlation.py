"""Result correlation tests for concurrent SaySo action requests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import (
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
from custom_components.sayso.results import ActionResultStatus
from custom_components.sayso.test_coordinator import fast_timing
from custom_components.sayso.test_deltas import _wait_until


def _action_request(
    *,
    request_id: str,
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


def _action_results(messages: list[str]) -> list[dict[str, Any]]:
    return [
        json.loads(message)
        for message in messages
        if json.loads(message).get("type") == "action_result"
    ]


class _GatedServiceCaller:
    """Block service calls until the test releases each entity gate."""

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def arm(self, entity_id: str) -> None:
        self.gates[entity_id] = asyncio.Event()

    async def __call__(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        data = data or {}
        entity_id = data["entity_id"]
        gate = self.gates[entity_id]
        await gate.wait()
        self.calls.append((domain, service, data))


@pytest.mark.asyncio
async def test_concurrent_requests_correlate_results_by_request_id(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Two in-flight allowed requests must not cross-match result payloads."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["light"],
            CONF_ACTION_ALLOWLIST: ["on"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_ENTITY_IDS: ["light.kitchen", "light.bedroom"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    kitchen = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    bedroom = entity_reg.async_get_or_create(
        "light",
        "test",
        "bedroom",
        suggested_object_id="bedroom",
    )
    hass.states.async_set(kitchen.entity_id, "off")
    hass.states.async_set(bedroom.entity_id, "off")
    await hass.async_block_till_done()

    fake_ws = FakeWebSocket()
    service_caller = _GatedServiceCaller()
    service_caller.arm(kitchen.entity_id)
    service_caller.arm(bedroom.entity_id)

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

    coordinator._conversation_contexts["req-kitchen"] = Context()
    coordinator._conversation_contexts["req-bedroom"] = Context()

    fake_ws.push(
        _action_request(
            request_id="req-kitchen",
            entity_id=kitchen.entity_id,
            domain="light",
            action="on",
        ),
    )
    fake_ws.push(
        _action_request(
            request_id="req-bedroom",
            entity_id=bedroom.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(
        lambda: len(
            [
                result
                for result in _action_results(fake_ws.sent)
                if result["payload"]["status"] == ActionResultStatus.ACCEPTED
            ],
        )
        >= 2,
        timeout=2.0,
    )

    accepted = [
        result["payload"]
        for result in _action_results(fake_ws.sent)
        if result["payload"]["status"] == ActionResultStatus.ACCEPTED
    ]
    assert {payload["request_id"] for payload in accepted} == {
        "req-kitchen",
        "req-bedroom",
    }

    # Complete bedroom first, then kitchen — results must stay correlated.
    service_caller.gates[bedroom.entity_id].set()
    await _wait_until(
        lambda: any(
            result["payload"]["request_id"] == "req-bedroom"
            and result["payload"]["status"] == ActionResultStatus.COMPLETED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    service_caller.gates[kitchen.entity_id].set()
    await _wait_until(
        lambda: any(
            result["payload"]["request_id"] == "req-kitchen"
            and result["payload"]["status"] == ActionResultStatus.COMPLETED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    completed = [
        result["payload"]
        for result in _action_results(fake_ws.sent)
        if result["payload"]["status"] == ActionResultStatus.COMPLETED
    ]
    assert {payload["request_id"] for payload in completed} == {
        "req-kitchen",
        "req-bedroom",
    }

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_rejected_result_carries_request_id_under_concurrency(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Permission rejection must include the originating request_id."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
        options={
            CONF_DOMAIN_ALLOWLIST: ["light"],
            CONF_ACTION_ALLOWLIST: ["on"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_ENTITY_IDS: ["light.kitchen"],
        },
    )
    entry.add_to_hass(hass)

    entity_reg = er.async_get(hass)
    kitchen = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    hass.states.async_set(kitchen.entity_id, "off")
    await hass.async_block_till_done()

    fake_ws = FakeWebSocket()
    service_caller = _GatedServiceCaller()
    service_caller.arm(kitchen.entity_id)

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

    coordinator._conversation_contexts["req-slow"] = Context()
    coordinator._conversation_contexts["req-reject"] = Context()

    fake_ws.push(
        _action_request(
            request_id="req-slow",
            entity_id=kitchen.entity_id,
            domain="light",
            action="on",
        ),
    )
    fake_ws.push(
        _action_request(
            request_id="req-reject",
            entity_id=kitchen.entity_id,
            domain="switch",
            action="on",
        ),
    )

    await _wait_until(
        lambda: any(
            result["payload"]["request_id"] == "req-reject"
            and result["payload"]["status"] == ActionResultStatus.REJECTED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    rejected = next(
        result["payload"]
        for result in _action_results(fake_ws.sent)
        if result["payload"]["request_id"] == "req-reject"
    )
    assert rejected["status"] == ActionResultStatus.REJECTED
    assert rejected["reason"] == "domain_mismatch"

    service_caller.gates[kitchen.entity_id].set()
    await _wait_until(
        lambda: any(
            result["payload"]["request_id"] == "req-slow"
            and result["payload"]["status"] == ActionResultStatus.COMPLETED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    await coordinator.async_stop()

"""State verification tests for SaySo action_request handling."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
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
    REASON_STATE_CHANGED,
    REASON_STATE_UNCHANGED,
    REASON_STATE_VERIFICATION_TIMEOUT,
)
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import SaySoConnectionCoordinator
from custom_components.sayso.results import ActionResultStatus
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


def _action_results(messages: list[str]) -> list[dict[str, Any]]:
    return [
        json.loads(message)
        for message in messages
        if json.loads(message).get("type") == "action_result"
    ]


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
async def test_completed_waits_for_state_change(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Accepted actions must not complete until the entity state changes."""

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
        state_verification_timeout=0.5,
    )
    await coordinator.async_start()
    await _wait_until(lambda: coordinator.connected, timeout=2.0)

    fake_ws.push(
        _action_request(
            entity_id=light.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(
        lambda: service_caller.calls,
        timeout=2.0,
    )
    await _wait_until(
        lambda: any(
            result["payload"]["status"] == ActionResultStatus.ACCEPTED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    assert not any(
        result["payload"]["status"] == ActionResultStatus.COMPLETED
        for result in _action_results(fake_ws.sent)
    )

    hass.states.async_set(light.entity_id, "on")
    await hass.async_block_till_done()

    await _wait_until(
        lambda: any(
            result["payload"]["status"] == ActionResultStatus.COMPLETED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    completed = next(
        result["payload"]
        for result in _action_results(fake_ws.sent)
        if result["payload"]["status"] == ActionResultStatus.COMPLETED
    )
    assert completed["reason"] == REASON_STATE_CHANGED

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_unchanged_outcome_when_state_value_is_unchanged(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """A state event with the same value must not look like a successful change."""

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
    light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    hass.states.async_set(light.entity_id, "on")
    await hass.async_block_till_done()

    fake_ws = FakeWebSocket()

    class _SameStateServiceCaller:
        async def __call__(
            self,
            _domain: str,
            _service: str,
            data: dict[str, Any] | None = None,
        ) -> None:
            data = data or {}
            hass.states.async_set(
                data["entity_id"],
                "on",
                {"sayso_pulse": 1},
            )

    service_caller = _SameStateServiceCaller()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(
        hass,
        entry,
        ws_connect=connect,
        service_caller=service_caller,
        state_verification_timeout=0.5,
    )
    await coordinator.async_start()
    await _wait_until(lambda: coordinator.connected, timeout=2.0)

    fake_ws.push(
        _action_request(
            entity_id=light.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(
        lambda: any(
            result["payload"]["status"] == ActionResultStatus.COMPLETED
            for result in _action_results(fake_ws.sent)
        ),
        timeout=2.0,
    )

    completed = next(
        result["payload"]
        for result in _action_results(fake_ws.sent)
        if result["payload"]["status"] == ActionResultStatus.COMPLETED
    )
    assert completed["reason"] == REASON_STATE_UNCHANGED

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_timeout_when_no_state_event(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Missing state feedback must fail verification instead of looking successful."""

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

    with patch(
        "custom_components.sayso.coordinator.STATE_VERIFICATION_TIMEOUT",
        0.05,
    ):
        coordinator = SaySoConnectionCoordinator(
            hass,
            entry,
            ws_connect=connect,
            service_caller=service_caller,
            state_verification_timeout=0.05,
        )
        await coordinator.async_start()
        await _wait_until(lambda: coordinator.connected, timeout=2.0)

        fake_ws.push(
            _action_request(
                entity_id=light.entity_id,
                domain="light",
                action="on",
            ),
        )

        await _wait_until(
            lambda: any(
                result["payload"]["status"] == ActionResultStatus.FAILED
                for result in _action_results(fake_ws.sent)
            ),
            timeout=2.0,
        )

    failed = next(
        result["payload"]
        for result in _action_results(fake_ws.sent)
        if result["payload"]["status"] == ActionResultStatus.FAILED
    )
    assert failed["reason"] == REASON_STATE_VERIFICATION_TIMEOUT
    assert not any(
        result["payload"]["status"] == ActionResultStatus.COMPLETED
        for result in _action_results(fake_ws.sent)
    )

    await coordinator.async_stop()

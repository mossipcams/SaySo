"""Connection coordinator tests for SaySo."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import Context, HomeAssistant
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
    HA_WS_MAX_MSG_SIZE,
    HELLO_ACK_TIMEOUT,
    MSG_ACTION_REQUEST,
    MSG_ACTION_RESULT,
    MSG_CONVERSATION_REQUEST,
    MSG_CONVERSATION_RESPONSE,
    MSG_PREPARE,
    MSG_PREPARE_RESPONSE,
    RECONNECT_INITIAL_DELAY,
    RECONNECT_MAX_DELAY,
    WS_CONNECT_TIMEOUT,
    WS_PATH,
)
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import (
    SaySoConnectionCoordinator,
    SaySoConversationError,
    _default_ws_connect as _real_default_ws_connect,
    http_to_ws_url,
    resolve_conversation_source,
)


@pytest.fixture
def fast_timing():
    """Use short coordinator timing so tests stay fast."""

    patched_heartbeat = 0.02
    with (
        patch(
            "custom_components.sayso.coordinator.RECONNECT_INITIAL_DELAY",
            0.01,
        ),
        patch(
            "custom_components.sayso.coordinator.RECONNECT_MAX_DELAY",
            0.05,
        ),
        patch(
            "custom_components.sayso.coordinator.HEARTBEAT_INTERVAL",
            patched_heartbeat,
        ),
    ):
        yield {"heartbeat_interval": patched_heartbeat}


async def _wait_until_connected(coordinator: SaySoConnectionCoordinator) -> None:
    for _ in range(50):
        if coordinator.connected:
            return
        await asyncio.sleep(0.01)
    pytest.fail("coordinator never connected")


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    for _ in range(int(timeout / interval)):
        if predicate():
            return
        await asyncio.sleep(interval)
    pytest.fail(f"condition not met within {timeout}s")


def _conversation_requests(ws: FakeWebSocket) -> list[dict[str, Any]]:
    return [
        json.loads(message)
        for message in ws.sent
        if json.loads(message)["type"] == MSG_CONVERSATION_REQUEST
    ]


def _action_request(
    *,
    turn_correlation_id: str,
    entity_id: str,
    domain: str,
    action: str,
    envelope_correlation_id: str = "hello-session-1",
) -> str:
    return json.dumps(
        {
            "version": 1,
            "type": MSG_ACTION_REQUEST,
            "correlation_id": envelope_correlation_id,
            "payload": {
                "request_id": turn_correlation_id,
                "entity_id": entity_id,
                "domain": domain,
                "action": action,
            },
        },
    )


def _action_results(ws: FakeWebSocket) -> list[dict[str, Any]]:
    return [
        json.loads(message)
        for message in ws.sent
        if json.loads(message)["type"] == MSG_ACTION_RESULT
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


def _conversation_response(
    *,
    correlation_id: str,
    speech: str,
    response_type: str = "no_action",
) -> str:
    return json.dumps(
        {
            "version": 1,
            "type": MSG_CONVERSATION_RESPONSE,
            "correlation_id": correlation_id,
            "payload": {
                "speech": speech,
                "response_type": response_type,
            },
        },
    )


def _register_satellite_device(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    *,
    area_name: str | None = "Living Room",
) -> tuple[dr.DeviceEntry, ar.AreaEntry | None]:
    area_reg = ar.async_get(hass)
    area = area_reg.async_create(area_name) if area_name is not None else None

    device_reg = dr.async_get(hass)
    device = device_reg.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "mac_satellite")},
    )
    if area is not None:
        device_reg.async_update_device(device.id, area_id=area.id)
    return device, area


@pytest.mark.asyncio
async def test_device_id_resolves_through_registry_to_area_in_conversation_request(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    device, area = _register_satellite_device(hass, entry)
    fake_ws = FakeWebSocket()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lamp",
            device_id=device.id,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    payload = requests[0]["payload"]
    assert payload["source_id"] == device.id
    assert payload["area_id"] == area.id
    assert "satellite_id" not in payload

    fake_ws.push(
        _conversation_response(
            correlation_id=requests[0]["correlation_id"],
            speech="Done.",
        ),
    )
    await request_task
    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_satellite_id_resolves_to_same_device_and_area(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    device, area = _register_satellite_device(hass, entry)
    fake_ws = FakeWebSocket()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lamp",
            satellite_id=device.id,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    payload = requests[0]["payload"]
    assert payload["source_id"] == device.id
    assert payload["area_id"] == area.id
    assert "satellite_id" not in payload

    fake_ws.push(
        _conversation_response(
            correlation_id=requests[0]["correlation_id"],
            speech="Done.",
        ),
    )
    await request_task
    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_device_without_area_produces_no_fabricated_origin(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    device, _area = _register_satellite_device(hass, entry, area_name=None)
    fake_ws = FakeWebSocket()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lamp",
            device_id=device.id,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    payload = requests[0]["payload"]
    assert payload["source_id"] == device.id
    assert "area_id" not in payload
    assert "satellite_id" not in payload

    fake_ws.push(
        _conversation_response(
            correlation_id=requests[0]["correlation_id"],
            speech="Done.",
        ),
    )
    await request_task
    await coordinator.async_stop()


def test_resolve_conversation_source_prefers_device_id_over_satellite_id(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create("Living Room")
    kitchen = area_reg.async_create("Kitchen")

    device_reg = dr.async_get(hass)
    primary = device_reg.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "primary")},
    )
    device_reg.async_update_device(primary.id, area_id=living_room.id)
    secondary = device_reg.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "secondary")},
    )
    device_reg.async_update_device(secondary.id, area_id=kitchen.id)

    source_id, area_id = resolve_conversation_source(
        hass,
        device_id=primary.id,
        satellite_id=secondary.id,
    )
    assert source_id == primary.id
    assert area_id == living_room.id


@pytest.mark.asyncio
async def test_starting_conversation_stores_exact_context_under_correlation_id(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    turn_context = Context(user_id="user-abc", parent_id="parent-xyz")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lamp",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    correlation_id = requests[0]["correlation_id"]
    assert coordinator._conversation_contexts[correlation_id] is turn_context

    fake_ws.push(
        _conversation_response(
            correlation_id=correlation_id,
            speech="Done.",
        ),
    )
    await request_task
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_concurrent_turns_keep_distinct_contexts(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    context_a = Context(user_id="user-a")
    context_b = Context(user_id="user-b")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    async def respond_both() -> None:
        for _ in range(100):
            requests = _conversation_requests(fake_ws)
            if len(requests) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected two conversation_request messages")

        stored = coordinator._conversation_contexts
        by_transcript = {
            request["payload"]["transcript"]: request["correlation_id"]
            for request in requests
        }
        assert stored[by_transcript["turn A"]] is context_a
        assert stored[by_transcript["turn B"]] is context_b

        for request in requests:
            fake_ws.push(
                _conversation_response(
                    correlation_id=request["correlation_id"],
                    speech=f"response-{request['payload']['transcript']}",
                ),
            )

    responder = asyncio.create_task(respond_both())
    try:
        await asyncio.gather(
            coordinator.async_request_conversation(
                transcript="turn A",
                satellite_id="macbook",
                context=context_a,
            ),
            coordinator.async_request_conversation(
                transcript="turn B",
                satellite_id="macbook",
                context=context_b,
            ),
        )
    finally:
        await responder

    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_context_never_appears_in_websocket_payload(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    turn_context = Context(user_id="user-abc", parent_id="parent-xyz")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lamp",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    raw_messages = [
        message
        for message in fake_ws.sent
        if json.loads(message)["type"] == MSG_CONVERSATION_REQUEST
    ]
    for raw in raw_messages:
        assert "Context" not in raw
        assert "user_id" not in raw
        assert "parent_id" not in raw
        payload = json.loads(raw)["payload"]
        assert "context" not in payload
        assert "context_id" not in payload

    correlation_id = requests[0]["correlation_id"]
    fake_ws.push(
        _conversation_response(
            correlation_id=correlation_id,
            speech="Done.",
        ),
    )
    await request_task
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_action_without_matching_stored_context_is_rejected(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    )
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the kitchen light",
            satellite_id="macbook",
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    turn_correlation_id = requests[0]["correlation_id"]
    fake_ws.push(
        _action_request(
            turn_correlation_id=turn_correlation_id,
            entity_id=light.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(lambda: _action_results(fake_ws), timeout=2.0)

    assert service_caller.calls == []
    results = _action_results(fake_ws)
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "missing_context"

    fake_ws.push(
        _conversation_response(
            correlation_id=turn_correlation_id,
            speech="Done.",
        ),
    )
    await request_task

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_orphan_action_request_without_stored_context_is_rejected(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Orphan action_request with no in-flight turn must fail closed."""

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
    )
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    fake_ws.push(
        _action_request(
            turn_correlation_id="orphan-turn-1",
            entity_id=light.entity_id,
            domain="light",
            action="on",
        ),
    )

    await _wait_until(lambda: _action_results(fake_ws), timeout=2.0)

    assert service_caller.calls == []
    results = _action_results(fake_ws)
    assert results[-1]["payload"]["status"] == "rejected"
    assert results[-1]["payload"]["reason"] == "missing_context"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_default_service_caller_passes_context_with_blocking(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    turn_context = Context(user_id="user-abc", parent_id="parent-xyz")
    service_call = AsyncMock()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the kitchen light",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    turn_correlation_id = requests[0]["correlation_id"]
    recorded: list[tuple[str, str, dict[str, Any], Context | None]] = []

    async def record_default_service_call(
        _hass: HomeAssistant,
        domain: str,
        service: str,
        data: dict[str, Any],
        *,
        context: Context | None = None,
    ) -> None:
        recorded.append((domain, service, data, context))

    with patch(
        "custom_components.sayso.coordinator._default_service_caller",
        side_effect=record_default_service_call,
    ):
        fake_ws.push(
            _action_request(
                turn_correlation_id=turn_correlation_id,
                entity_id=light.entity_id,
                domain="light",
                action="on",
            ),
        )
        await _wait_until(lambda: recorded, timeout=2.0)

    assert recorded == [
        ("light", "turn_on", {"entity_id": light.entity_id}, turn_context),
    ]

    fake_ws.push(
        _conversation_response(
            correlation_id=turn_correlation_id,
            speech="Done.",
        ),
    )
    await request_task
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_concurrent_conversation_requests_receive_matching_responses(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    async def respond_in_reverse_order() -> None:
        for _ in range(100):
            requests = _conversation_requests(fake_ws)
            if len(requests) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected two conversation_request messages")

        second = requests[1]["correlation_id"]
        first = requests[0]["correlation_id"]
        fake_ws.push(
            _conversation_response(
                correlation_id=second,
                speech="response for B",
            ),
        )
        fake_ws.push(
            _conversation_response(
                correlation_id=first,
                speech="response for A",
            ),
        )

    responder = asyncio.create_task(respond_in_reverse_order())
    try:
        result_a, result_b = await asyncio.gather(
            coordinator.async_request_conversation(
                transcript="turn A",
                satellite_id="macbook",
            ),
            coordinator.async_request_conversation(
                transcript="turn B",
                satellite_id="macbook",
            ),
        )
    finally:
        await responder

    assert result_a["speech"] == "response for A"
    assert result_b["speech"] == "response for B"
    assert coordinator._pending_conversations == {}
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_clarification_response_clears_retained_context(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    turn_context = Context(user_id="user-clarify")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lights",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    correlation_id = requests[0]["correlation_id"]
    assert coordinator._conversation_contexts[correlation_id] is turn_context

    fake_ws.push(
        _conversation_response(
            correlation_id=correlation_id,
            speech="Which room?",
            response_type="clarification",
        ),
    )
    result = await request_task
    assert result["response_type"] == "clarification"
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_invalid_conversation_response_clears_retained_context(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    turn_context = Context(user_id="user-error")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="turn on the lamp",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    correlation_id = requests[0]["correlation_id"]
    assert coordinator._conversation_contexts[correlation_id] is turn_context

    fake_ws.push(
        json.dumps(
            {
                "version": 1,
                "type": MSG_CONVERSATION_RESPONSE,
                "correlation_id": correlation_id,
                "payload": "not-a-dict",
            },
        ),
    )

    with pytest.raises(SaySoConversationError):
        await request_task

    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_conversation_request_timeout_clears_pending_state(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    turn_context = Context(user_id="user-timeout")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    with patch(
        "custom_components.sayso.coordinator.CONVERSATION_REQUEST_TIMEOUT",
        0.05,
    ):
        with pytest.raises(asyncio.TimeoutError):
            await coordinator.async_request_conversation(
                transcript="slow turn",
                satellite_id="macbook",
                context=turn_context,
            )

    assert coordinator._pending_conversations == {}
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_conversation_request_disconnect_fails_pending_and_clears_state(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    turn_context = Context(user_id="user-disconnect")
    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="pending turn",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    assert coordinator._conversation_contexts[requests[0]["correlation_id"]] is turn_context

    fake_ws.disconnect()

    with pytest.raises(SaySoConversationError):
        await request_task

    assert coordinator._pending_conversations == {}
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_cancelled_conversation_request_clears_retained_context(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    turn_context = Context(user_id="user-cancel")

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()
    await _wait_until_connected(coordinator)

    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="pending turn",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    assert coordinator._conversation_contexts[requests[0]["correlation_id"]] is turn_context

    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert coordinator._pending_conversations == {}
    assert coordinator._conversation_contexts == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_conversation_request_shutdown_fails_pending_and_clears_state(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    turn_context = Context(user_id="user-shutdown")
    request_task = asyncio.create_task(
        coordinator.async_request_conversation(
            transcript="pending turn",
            satellite_id="macbook",
            context=turn_context,
        ),
    )

    for _ in range(100):
        requests = _conversation_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("conversation_request was never sent")

    assert coordinator._conversation_contexts[requests[0]["correlation_id"]] is turn_context

    stop_task = asyncio.create_task(coordinator.async_stop())

    with pytest.raises(SaySoConversationError):
        await request_task

    await stop_task
    assert coordinator._pending_conversations == {}
    assert coordinator._conversation_contexts == {}


@pytest.mark.asyncio
async def test_http_to_ws_url_converts_scheme_and_path() -> None:
    assert (
        http_to_ws_url("http://127.0.0.1:8765")
        == f"ws://127.0.0.1:8765{WS_PATH}"
    )
    assert (
        http_to_ws_url("https://sayso.example.com/")
        == f"wss://sayso.example.com{WS_PATH}"
    )


@pytest.mark.asyncio
async def test_disconnect_clears_connected_immediately(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()

    async def connect(_url: str, token: str) -> FakeWebSocket:
        assert token == "secret-token"
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    async def wait_until_connected() -> None:
        for _ in range(50):
            if coordinator.connected:
                return
            await asyncio.sleep(0.01)
        pytest.fail("coordinator never connected")

    await wait_until_connected()
    assert coordinator.connected is True

    fake_ws.disconnect()

    for _ in range(20):
        if not coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("connected stayed true after disconnect")

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_reconnect_after_disconnect_restores_connected(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    sockets: list[FakeWebSocket] = []

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        ws = FakeWebSocket()
        sockets.append(ws)
        return ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    async def wait_for_socket_count(count: int) -> None:
        for _ in range(100):
            if len(sockets) >= count:
                return
            await asyncio.sleep(0.01)
        pytest.fail(f"expected {count} socket(s), got {len(sockets)}")

    await wait_for_socket_count(1)
    assert coordinator.connected is True

    sockets[0].disconnect()

    for _ in range(50):
        if not coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("connected stayed true after first disconnect")

    await wait_for_socket_count(2)
    assert coordinator.connected is True

    await coordinator.async_stop()


def _snapshot_messages(ws: FakeWebSocket) -> list[dict]:
    return [
        json.loads(message)
        for message in ws.sent
        if json.loads(message)["type"] == "graph_snapshot"
    ]


@pytest.mark.asyncio
async def test_reconnect_after_kill_restores_graph_snapshot(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Kill/reconnect must send a fresh snapshot before the integration is ready."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    sockets: list[FakeWebSocket] = []

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        ws = FakeWebSocket()
        sockets.append(ws)
        return ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    async def wait_for_socket_count(count: int) -> None:
        for _ in range(100):
            if len(sockets) >= count:
                return
            await asyncio.sleep(0.01)
        pytest.fail(f"expected {count} socket(s), got {len(sockets)}")

    await wait_for_socket_count(1)
    assert coordinator.connected is True
    assert len(_snapshot_messages(sockets[0])) == 1

    sockets[0].disconnect()

    for _ in range(50):
        if not coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("connected stayed true after kill")

    await wait_for_socket_count(2)

    second = sockets[1]
    snapshots = _snapshot_messages(second)
    assert len(snapshots) == 1
    assert coordinator.connected is True
    assert snapshots[0]["payload"]["home_id"] == entry.entry_id

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_connected_sends_heartbeats(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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

    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("coordinator never connected")

    await asyncio.sleep(fast_timing["heartbeat_interval"] + 0.05)

    ping_messages = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message)["type"] == "ping"
    ]
    assert ping_messages, "expected at least one heartbeat ping"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_ws_connect_uses_bearer_auth_not_logged(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    captured: dict[str, Any] = {}

    async def connect(url: str, token: str) -> FakeWebSocket:
        captured["url"] = url
        captured["token"] = token
        return FakeWebSocket()

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    with caplog.at_level("DEBUG", logger="custom_components.sayso.coordinator"):
        await coordinator.async_start()
        for _ in range(50):
            if coordinator.connected:
                break
            await asyncio.sleep(0.01)

    assert captured["url"] == f"ws://127.0.0.1:8765{WS_PATH}"
    assert captured["token"] == "secret-token"
    assert "secret-token" not in caplog.text

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_reconnect_backoff_delay_is_bounded(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    connect = AsyncMock(side_effect=ConnectionError("offline"))

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    delays: list[float] = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay: float) -> None:
        delays.append(delay)
        await original_sleep(0)

    with patch(
        "custom_components.sayso.coordinator.asyncio.sleep",
        side_effect=record_sleep,
    ):
        await coordinator.async_start()
        await original_sleep(0.2)
        await coordinator.async_stop()

    reconnect_delays = [delay for delay in delays if delay > 0]
    assert reconnect_delays
    assert max(reconnect_delays) <= RECONNECT_MAX_DELAY


@pytest.mark.asyncio
async def test_hello_ack_requires_wsmsgtype_not_string_type(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Fail if coordinator compares message.type to string literals."""

    from aiohttp import WSMsgType

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    hello_ack = json.dumps({"version": 1, "type": "hello_ack", "payload": {}})

    class StrictAiohttpMessageWebSocket(FakeWebSocket):
        async def receive(self) -> Any:
            if self.closed:
                return type(
                    "WSMessage",
                    (),
                    {"type": WSMsgType.CLOSE, "data": None},
                )()
            message = await self._recv_queue.get()
            if message is None:
                self.closed = True
                return type(
                    "WSMessage",
                    (),
                    {"type": WSMsgType.CLOSE, "data": None},
                )()
            # String comparison would never match real aiohttp enum values.
            assert WSMsgType.TEXT != "TEXT"
            return type(
                "WSMessage",
                (),
                {"type": WSMsgType.TEXT, "data": message},
            )()

    fake_ws = StrictAiohttpMessageWebSocket()

    async def connect(_url: str, _token: str) -> StrictAiohttpMessageWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("hello_ack not recognized — message.type may be compared to strings")

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_connection_failure_logs_warning_with_exception(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        msg = "server unreachable"
        raise ConnectionError(msg)

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    with caplog.at_level("WARNING", logger="custom_components.sayso.coordinator"):
        await coordinator.async_start()
        await asyncio.sleep(0.05)
        await coordinator.async_stop()

    assert any(
        "SaySo connection failed" in record.message and record.exc_info is not None
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_default_ws_connect_uses_large_max_msg_size(
    hass: HomeAssistant,
) -> None:
    from unittest.mock import MagicMock

    shared_session = MagicMock()
    shared_session.ws_connect = AsyncMock(return_value=FakeWebSocket())

    with patch(
        "custom_components.sayso.coordinator.async_get_clientsession",
        return_value=shared_session,
    ):
        await _real_default_ws_connect(hass, f"ws://127.0.0.1:8765{WS_PATH}", "token")

    shared_session.ws_connect.assert_called_once_with(
        f"ws://127.0.0.1:8765{WS_PATH}",
        headers={"Authorization": "Bearer token"},
        max_msg_size=HA_WS_MAX_MSG_SIZE,
    )


@pytest.mark.asyncio
async def test_connect_once_uses_bounded_ws_and_hello_ack_timeouts(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    async def hang_forever(_url: str, _token: str) -> FakeWebSocket:
        await asyncio.Event().wait()
        return FakeWebSocket()

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=hang_forever)
    wait_for_calls: list[float] = []
    original_wait_for = asyncio.wait_for

    async def record_wait_for(coro, *, timeout):  # type: ignore[no-untyped-def]
        wait_for_calls.append(timeout)
        return await original_wait_for(coro, timeout=timeout)

    with (
        patch(
            "custom_components.sayso.coordinator.asyncio.wait_for",
            side_effect=record_wait_for,
        ),
        patch("custom_components.sayso.coordinator.WS_CONNECT_TIMEOUT", 0.01),
    ):
        with pytest.raises(asyncio.TimeoutError):
            await coordinator._connect_once()

    assert wait_for_calls == [0.01]

    class SilentWebSocket(FakeWebSocket):
        async def receive(self) -> Any:
            await asyncio.Event().wait()
            return await super().receive()

    async def connect(_url: str, _token: str) -> SilentWebSocket:
        return SilentWebSocket()

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    wait_for_calls.clear()

    with (
        patch(
            "custom_components.sayso.coordinator.asyncio.wait_for",
            side_effect=record_wait_for,
        ),
        patch("custom_components.sayso.coordinator.HELLO_ACK_TIMEOUT", 0.01),
    ):
        with pytest.raises(asyncio.TimeoutError):
            await coordinator._connect_once()

    assert wait_for_calls == [WS_CONNECT_TIMEOUT, 0.01]


@pytest.mark.asyncio
async def test_default_ws_connect_uses_hass_shared_session(
    hass: HomeAssistant,
) -> None:
    """Production connect must not allocate a private ClientSession per attempt."""

    from unittest.mock import MagicMock

    shared_session = MagicMock()
    shared_session.ws_connect = AsyncMock(return_value=FakeWebSocket())

    with (
        patch("aiohttp.ClientSession") as session_ctor,
        patch(
            "custom_components.sayso.coordinator.async_get_clientsession",
            return_value=shared_session,
        ),
        patch(
            "custom_components.sayso.coordinator._default_ws_connect",
            _real_default_ws_connect,
        ),
    ):
        await _real_default_ws_connect(hass, f"ws://127.0.0.1:8765{WS_PATH}", "token")

    session_ctor.assert_not_called()
    shared_session.ws_connect.assert_called_once()


@pytest.mark.asyncio
async def test_private_connect_session_closed_when_socket_ends(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Per-connect sessions attached to a socket are closed when the socket ends."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    fake_session = AsyncMock()
    fake_ws._sayso_session = fake_session

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("coordinator never connected")

    fake_ws.disconnect()

    for _ in range(20):
        if fake_session.close.await_count:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("private connect session was not closed when socket ended")

    await coordinator.async_stop()


def _prepare_requests(ws: FakeWebSocket) -> list[dict[str, Any]]:
    return [
        json.loads(message)
        for message in ws.sent
        if json.loads(message)["type"] == MSG_PREPARE
    ]


def _prepare_response(
    *,
    correlation_id: str,
    connected: bool = True,
    graph_ready: bool = True,
    model_ready: bool = True,
) -> str:
    return json.dumps(
        {
            "version": 1,
            "type": MSG_PREPARE_RESPONSE,
            "correlation_id": correlation_id,
            "payload": {
                "connected": connected,
                "graph_ready": graph_ready,
                "model_ready": model_ready,
            },
        },
    )


@pytest.mark.asyncio
async def test_prepare_request_reports_readiness_without_conversation_or_action(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    prepare_task = asyncio.create_task(coordinator.async_request_prepare())

    for _ in range(100):
        requests = _prepare_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("prepare request was never sent")

    assert _conversation_requests(fake_ws) == []
    assert requests[0]["payload"] == {}

    fake_ws.push(_prepare_response(correlation_id=requests[0]["correlation_id"]))
    payload = await prepare_task

    assert payload == {
        "connected": True,
        "graph_ready": True,
        "model_ready": True,
    }
    assert coordinator._pending_prepares == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_prepare_request_negative_readiness_raises_without_conversation(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    prepare_task = asyncio.create_task(coordinator.async_request_prepare())

    for _ in range(100):
        requests = _prepare_requests(fake_ws)
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("prepare request was never sent")

    fake_ws.push(
        _prepare_response(
            correlation_id=requests[0]["correlation_id"],
            model_ready=False,
        ),
    )

    with pytest.raises(SaySoConversationError, match="not ready"):
        await prepare_task

    assert _conversation_requests(fake_ws) == []
    assert coordinator._pending_prepares == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_prepare_request_timeout_clears_pending_state(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    with patch(
        "custom_components.sayso.coordinator.PREPARE_REQUEST_TIMEOUT",
        0.05,
    ):
        with pytest.raises(asyncio.TimeoutError):
            await coordinator.async_request_prepare()

    assert coordinator._pending_prepares == {}

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_concurrent_prepare_requests_receive_matching_responses(
    hass: HomeAssistant,
    fast_timing,
) -> None:
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
    await _wait_until_connected(coordinator)

    async def respond_both() -> None:
        for _ in range(100):
            requests = _prepare_requests(fake_ws)
            if len(requests) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected two prepare messages")

        by_correlation = {request["correlation_id"]: request for request in requests}
        for correlation_id in by_correlation:
            fake_ws.push(
                _prepare_response(
                    correlation_id=correlation_id,
                    model_ready=correlation_id == requests[0]["correlation_id"],
                ),
            )

    responder = asyncio.create_task(respond_both())
    try:
        first, second = await asyncio.gather(
            coordinator.async_request_prepare(),
            coordinator.async_request_prepare(),
            return_exceptions=True,
        )
    finally:
        await responder

    ready_payload = {
        "connected": True,
        "graph_ready": True,
        "model_ready": True,
    }
    results = [first, second]
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(
        isinstance(result, SaySoConversationError) and "not ready" in str(result)
        for result in results
    ) == 1
    assert ready_payload in [result for result in results if isinstance(result, dict)]
    assert coordinator._pending_prepares == {}

    await coordinator.async_stop()


"""Live Home Assistant WebSocket action request client tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from sayso_server.api import API_VERSION
from sayso_server.gateway import _process_graph_messages
from sayso_server.graph_store import HomeGraphStore
from sayso_server.ha_ws_client import BoundHaWsActionClient, HaWsActionClient
from sayso_server.messages import MessageType
from sayso_server.results import ActionResultStatus
from sayso_server.session import HaGatewayBinding, HaSession
from sayso_server.test_gateway import FakeGatewayWebSocket


def _action_result_envelope(
    *,
    request_id: str,
    status: str,
    reason: str | None = None,
    correlation_id: str = "corr-1",
) -> str:
    payload: dict[str, object] = {"request_id": request_id, "status": status}
    if reason is not None:
        payload["reason"] = reason
    return json.dumps(
        {
            "version": API_VERSION,
            "type": MessageType.ACTION_RESULT.value,
            "correlation_id": correlation_id,
            "payload": payload,
        },
    )


@pytest.mark.asyncio
async def test_ha_ws_client_sends_action_request_envelope() -> None:
    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-send", graph=HomeGraphStore())
    client = HaWsActionClient(ws, session, correlation_id="corr-send")
    processor = asyncio.create_task(_process_graph_messages(ws, session))

    client.send_action_request(
        request_id="req-send-1",
        entity_id="light.floor_lamp",
        domain="light",
        action="off",
        data={"brightness": 10},
    )
    await asyncio.sleep(0)

    assert len(ws.sent) == 1
    sent = json.loads(ws.sent[0])
    assert sent["type"] == MessageType.ACTION_REQUEST.value
    assert sent["correlation_id"] == "corr-send"
    assert sent["payload"] == {
        "request_id": "req-send-1",
        "entity_id": "light.floor_lamp",
        "domain": "light",
        "action": "off",
        "data": {"brightness": 10},
    }

    ws.push(None)
    await processor


@pytest.mark.asyncio
async def test_ha_ws_client_take_action_results_returns_correlated_payloads() -> None:
    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-results", graph=HomeGraphStore())
    client = HaWsActionClient(ws, session, correlation_id="corr-results")
    processor = asyncio.create_task(_process_graph_messages(ws, session))

    ws.push(
        _action_result_envelope(
            request_id="req-1",
            status=ActionResultStatus.ACCEPTED.value,
        ),
    )
    ws.push(
        _action_result_envelope(
            request_id="req-1",
            status=ActionResultStatus.COMPLETED.value,
            reason="state_changed",
        ),
    )
    await asyncio.sleep(0)

    results = client.take_action_results("req-1")
    assert [result.status for result in results] == [
        ActionResultStatus.ACCEPTED,
        ActionResultStatus.COMPLETED,
    ]
    assert results[-1].reason == "state_changed"
    assert client.take_action_results("req-1") == []

    ws.push(None)
    await processor


@pytest.mark.asyncio
async def test_gateway_does_not_drop_action_result_for_other_request_ids() -> None:
    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-multi", graph=HomeGraphStore())
    client = HaWsActionClient(ws, session, correlation_id="corr-multi")
    processor = asyncio.create_task(_process_graph_messages(ws, session))

    ws.push(
        _action_result_envelope(
            request_id="req-a",
            status=ActionResultStatus.REJECTED.value,
            reason="permission_denied",
        ),
    )
    ws.push(
        _action_result_envelope(
            request_id="req-b",
            status=ActionResultStatus.ACCEPTED.value,
        ),
    )
    await asyncio.sleep(0)

    assert len(client.take_action_results("req-a")) == 1
    assert len(client.take_action_results("req-b")) == 1
    assert client.take_action_results("req-c") == []

    ws.push(None)
    await processor


@pytest.mark.asyncio
async def test_ha_ws_client_collect_action_results_without_deadlock() -> None:
    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-collect", graph=HomeGraphStore())
    client = HaWsActionClient(ws, session, correlation_id="corr-collect")
    processor = asyncio.create_task(_process_graph_messages(ws, session))
    request_id = "req-collect-1"

    async def respond_to_action_request() -> None:
        for _ in range(200):
            if ws.sent:
                break
            await asyncio.sleep(0)
        assert ws.sent, "action_request was not flushed onto the websocket"
        sent = json.loads(ws.sent[0])
        assert sent["type"] == MessageType.ACTION_REQUEST.value
        assert sent["payload"]["request_id"] == request_id
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status=ActionResultStatus.ACCEPTED.value,
                correlation_id="corr-collect",
            ),
        )
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status=ActionResultStatus.COMPLETED.value,
                reason="state_changed",
                correlation_id="corr-collect",
            ),
        )

    responder = asyncio.create_task(respond_to_action_request())
    client.send_action_request(
        request_id=request_id,
        entity_id="light.floor_lamp",
        domain="light",
        action="off",
    )
    results = await asyncio.wait_for(
        client.collect_action_results(request_id),
        timeout=2.0,
    )

    assert [result.status for result in results] == [
        ActionResultStatus.ACCEPTED,
        ActionResultStatus.COMPLETED,
    ]
    assert results[-1].reason == "state_changed"

    ws.push(None)
    await responder
    await processor


@pytest.mark.asyncio
async def test_bound_ha_ws_action_client_collects_via_gateway_binding() -> None:
    binding = HaGatewayBinding()
    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-bound", graph=HomeGraphStore())
    binding.attach(session, ws)
    client = BoundHaWsActionClient(binding)
    processor = asyncio.create_task(_process_graph_messages(ws, session))
    request_id = "req-bound-1"

    async def respond_to_action_request() -> None:
        for _ in range(200):
            if ws.sent:
                break
            await asyncio.sleep(0)
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status=ActionResultStatus.ACCEPTED.value,
                correlation_id="corr-bound",
            ),
        )
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status=ActionResultStatus.COMPLETED.value,
                reason="state_changed",
                correlation_id="corr-bound",
            ),
        )

    responder = asyncio.create_task(respond_to_action_request())
    client.send_action_request(
        request_id=request_id,
        entity_id="light.floor_lamp",
        domain="light",
        action="off",
    )
    results = await asyncio.wait_for(
        client.collect_action_results(request_id),
        timeout=2.0,
    )

    assert [result.status for result in results] == [
        ActionResultStatus.ACCEPTED,
        ActionResultStatus.COMPLETED,
    ]

    ws.push(None)
    await responder
    await processor

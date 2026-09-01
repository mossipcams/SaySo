"""Live Home Assistant WebSocket action request client tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from sayso_server.api import API_VERSION
from sayso_server.gateway import _process_graph_messages
from sayso_server.graph_store import HomeGraphStore
from sayso_server.ha_ws_client import HaWsActionClient
from sayso_server.messages import MessageType
from sayso_server.results import ActionResultStatus
from sayso_server.session import HaSession
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

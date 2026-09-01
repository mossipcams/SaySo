"""HA WebSocket session gateway tests."""

from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.const import WS_PATH
from sayso_server.gateway import handle_ha_connection
from sayso_server.graph_store import HomeGraphStore
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.messages import MessageType

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


class FakeGatewayWebSocket:
    """Minimal async WebSocket stand-in for gateway tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._recv_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def push(self, message: str) -> None:
        self._recv_queue.put_nowait(message)

    async def receive_str(self) -> str | None:
        return await self._recv_queue.get()


def _hello_envelope(*, version: int = API_VERSION, correlation_id: str = "req-hello-1") -> str:
    return json.dumps(
        {
            "version": version,
            "type": MessageType.HELLO.value,
            "correlation_id": correlation_id,
            "payload": {},
        },
    )


def test_ws_path_matches_ha_integration() -> None:
    assert WS_PATH == "/api/v1/ws"


def test_bearer_token_valid_rejects_wrong_token() -> None:
    assert bearer_token_valid(authorization="Bearer wrong", expected_token="secret") is False


def test_bearer_token_valid_accepts_matching_token() -> None:
    assert bearer_token_valid(authorization="Bearer secret", expected_token="secret") is True


def test_bearer_token_valid_uses_constant_time_compare() -> None:
    calls: list[tuple[str, str]] = []
    original = hmac.compare_digest

    def spy(left: str, right: str) -> bool:
        calls.append((left, right))
        return original(left, right)

    with patch("sayso_server.auth.hmac.compare_digest", side_effect=spy):
        assert bearer_token_valid(authorization="Bearer secret", expected_token="secret")

    assert calls == [("secret", "secret")]


@pytest.mark.asyncio
async def test_wrong_token_closes_without_session() -> None:
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope())

    session = await handle_ha_connection(
        ws,
        authorization="Bearer wrong-token",
        server_token="secret-token",
    )

    assert session is None
    assert ws.closed is True
    assert ws.sent == []


@pytest.mark.asyncio
async def test_wrong_api_version_closes_without_session() -> None:
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(version=2))

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
    )

    assert session is None
    assert ws.closed is True
    assert ws.sent == []


@pytest.mark.asyncio
async def test_valid_v1_handshake_returns_hello_ack() -> None:
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="corr-123"))

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
    )

    assert session is not None
    assert session.correlation_id == "corr-123"
    assert ws.closed is False
    assert len(ws.sent) == 1

    ack = json.loads(ws.sent[0])
    assert ack["version"] == API_VERSION
    assert ack["type"] == MessageType.HELLO_ACK.value
    assert ack["correlation_id"] == "corr-123"
    assert ack["payload"] == {}


def test_create_aiohttp_app_registers_ws_route() -> None:
    from sayso_server.app import create_aiohttp_app
    from sayso_server.health import HEALTH_PATH

    app = create_aiohttp_app("secret-token")
    paths = {route.resource.canonical for route in app.router.routes()}
    assert WS_PATH in paths
    assert HEALTH_PATH in paths


@pytest.mark.asyncio
async def test_aiohttp_ws_handler_rejects_wrong_token() -> None:
    from aiohttp import web
    from unittest.mock import AsyncMock, MagicMock

    from sayso_server.app import create_aiohttp_app

    app = create_aiohttp_app("secret-token")
    handler = _ws_handler(app)

    request = MagicMock()
    request.headers = {"Authorization": "Bearer wrong-token"}

    with pytest.raises(web.HTTPUnauthorized):
        await handler(request)


@pytest.mark.asyncio
async def test_aiohttp_ws_handler_passes_server_token_to_gateway() -> None:
    from aiohttp import web
    from unittest.mock import AsyncMock, MagicMock, patch

    from sayso_server.app import create_aiohttp_app

    app = create_aiohttp_app("secret-token")
    handler = _ws_handler(app)

    request = MagicMock()
    request.headers = {"Authorization": "Bearer secret-token"}

    ws = MagicMock()
    ws.prepare = AsyncMock()
    ws.closed = False
    ws.send_str = AsyncMock()
    ws.receive = AsyncMock(
        return_value=MagicMock(
            type=web.WSMsgType.TEXT,
            data=_hello_envelope(correlation_id="ws-route-1"),
        ),
    )

    with patch("sayso_server.app.web.WebSocketResponse", return_value=ws):
        with patch(
            "sayso_server.app.handle_ha_connection",
            new_callable=AsyncMock,
            return_value=MagicMock(correlation_id="ws-route-1"),
        ) as mock_handle:
            await handler(request)

    mock_handle.assert_awaited_once()
    _, kwargs = mock_handle.call_args
    assert kwargs["authorization"] == "Bearer secret-token"
    assert kwargs["server_token"] == "secret-token"
    assert kwargs["graph_store"] is app["graph_store"]
    assert "token" not in kwargs


def _load_graph_fixture() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _graph_envelope(*, msg_type: str, payload: dict, correlation_id: str = "corr-1") -> str:
    return json.dumps(
        {
            "version": API_VERSION,
            "type": msg_type,
            "correlation_id": correlation_id,
            "payload": payload,
        },
    )


@pytest.mark.asyncio
async def test_gateway_updates_shared_graph_store_not_session_orphan() -> None:
    shared = HomeGraphStore()
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="shared-graph-1"))
    snapshot = _load_graph_fixture()
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=snapshot.model_dump(mode="json"),
        ),
    )
    ws.push(
        _graph_envelope(
            msg_type=MessageType.STATE_DELTA.value,
            payload={
                "version": 1,
                "home_id": "eval-home",
                "sequence": 43,
                "entity_id": "light.floor_lamp",
                "state": {"value": "on", "attributes": {"brightness": 90}},
            },
        ),
    )
    ws.push(None)

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
    )

    assert session is not None
    assert session.graph is shared
    assert shared.sequence == 43
    assert shared.snapshot is not None
    lamp = next(
        entity for entity in shared.snapshot.entities if entity.entity_id == "light.floor_lamp"
    )
    assert lamp.state.value == "on"


@pytest.mark.asyncio
async def test_gateway_rejects_stale_delta_on_shared_graph_store() -> None:
    shared = HomeGraphStore()
    shared.replace_snapshot(_load_graph_fixture())
    before_sequence = shared.sequence
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="stale-delta-1"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.STATE_DELTA.value,
            payload={
                "version": 1,
                "home_id": "eval-home",
                "sequence": before_sequence,
                "entity_id": "light.floor_lamp",
                "state": {"value": "on", "attributes": {}},
            },
        ),
    )
    ws.push(
        _graph_envelope(
            msg_type=MessageType.STATE_DELTA.value,
            payload={
                "version": 1,
                "home_id": "wrong-home",
                "sequence": before_sequence + 1,
                "entity_id": "light.floor_lamp",
                "state": {"value": "on", "attributes": {}},
            },
        ),
    )
    ws.push(None)

    await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
    )

    assert shared.sequence == before_sequence
    lamp = next(
        entity for entity in shared.snapshot.entities if entity.entity_id == "light.floor_lamp"
    )
    assert lamp.state.value == "off"


def _ws_handler(app: object) -> object:
    from aiohttp import web

    for route in app.router.routes():  # type: ignore[attr-defined]
        resource = route.resource
        if isinstance(resource, web.StaticResource):
            continue
        if resource.canonical == WS_PATH:
            return route.handler
    raise AssertionError(f"missing websocket route {WS_PATH}")

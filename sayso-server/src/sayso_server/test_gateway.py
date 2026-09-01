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
from sayso_server.readiness import ReadinessState

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
    assert kwargs["readiness"] is app["readiness"]
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

    end_state: dict[str, object] = {}

    def capture_end_state(_session: object) -> None:
        end_state["sequence"] = shared.sequence
        lamp = next(
            entity for entity in shared.snapshot.entities if entity.entity_id == "light.floor_lamp"
        )
        end_state["lamp_value"] = lamp.state.value

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
        on_session_ended=capture_end_state,
    )

    assert session is not None
    assert session.graph is shared
    assert session.graph_ready is True
    assert end_state["sequence"] == 43
    assert end_state["lamp_value"] == "on"
    assert shared.snapshot is None


@pytest.mark.asyncio
async def test_gateway_rejects_stale_delta_on_shared_graph_store() -> None:
    shared = HomeGraphStore()
    snapshot = _load_graph_fixture()
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="stale-delta-1"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=snapshot.model_dump(mode="json"),
        ),
    )
    before_sequence = snapshot.sequence
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

    end_state: dict[str, object] = {}

    def capture_end_state(_session: object) -> None:
        end_state["sequence"] = shared.sequence
        lamp = next(
            entity for entity in shared.snapshot.entities if entity.entity_id == "light.floor_lamp"
        )
        end_state["lamp_value"] = lamp.state.value

    await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
        on_session_ended=capture_end_state,
    )

    assert end_state["sequence"] == before_sequence
    assert end_state["lamp_value"] == "off"
    assert shared.snapshot is None


@pytest.mark.asyncio
async def test_reconnect_clears_graph_until_fresh_snapshot() -> None:
    """Kill/reconnect must restore the graph only after a new snapshot arrives."""

    shared = HomeGraphStore()
    snapshot = _load_graph_fixture()
    shared.replace_snapshot(snapshot)
    assert shared.snapshot is not None

    ws1 = FakeGatewayWebSocket()
    ws1.push(_hello_envelope(correlation_id="reconnect-1"))
    ws1.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=snapshot.model_dump(mode="json"),
        ),
    )
    ws1.push(None)

    session1 = await handle_ha_connection(
        ws1,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
    )

    assert session1 is not None
    assert session1.graph_ready is True
    assert shared.snapshot is None

    ws2 = FakeGatewayWebSocket()
    ws2.push(_hello_envelope(correlation_id="reconnect-2"))
    resynced = snapshot.model_copy(update={"sequence": 200})
    ws2.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=resynced.model_dump(mode="json"),
        ),
    )
    ws2.push(None)

    end_state: dict[str, int] = {}

    def capture_resync(_session: object) -> None:
        end_state["sequence"] = shared.sequence

    session2 = await handle_ha_connection(
        ws2,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
        on_session_ended=capture_resync,
    )

    assert session2 is not None
    assert session2.graph_ready is True
    assert end_state["sequence"] == 200
    assert shared.snapshot is None


@pytest.mark.asyncio
async def test_json_ping_envelope_gets_pong() -> None:
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="ping-1"))
    ws.push(
        json.dumps(
            {
                "version": API_VERSION,
                "type": MessageType.PING.value,
                "correlation_id": "hb-42",
                "payload": {},
            },
        ),
    )
    ws.push(None)

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
    )

    assert session is not None
    assert len(ws.sent) == 2
    pong = json.loads(ws.sent[1])
    assert pong["type"] == MessageType.PONG.value
    assert pong["correlation_id"] == "hb-42"
    assert pong["payload"] == {}


@pytest.mark.asyncio
async def test_invalid_graph_snapshot_sends_error_and_stays_not_ready() -> None:
    shared = HomeGraphStore()
    readiness = ReadinessState()
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="bad-snapshot-1"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload={"not_a": "snapshot"},
            correlation_id="bad-snapshot-1",
        ),
    )
    ws.push(None)

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
        readiness=readiness,
    )

    assert session is not None
    assert session.graph_ready is False
    assert shared.snapshot is None
    assert readiness.snapshot().ha_connected is False
    error_msgs = [json.loads(message) for message in ws.sent if '"error"' in message]
    assert len(error_msgs) == 1
    assert error_msgs[0]["type"] == MessageType.ERROR.value
    assert error_msgs[0]["correlation_id"] == "bad-snapshot-1"
    assert error_msgs[0]["payload"]["reason"] == "invalid_graph_snapshot"


@pytest.mark.asyncio
async def test_reconnect_without_snapshot_is_not_graph_ready() -> None:
    shared = HomeGraphStore()
    shared.replace_snapshot(_load_graph_fixture())
    readiness = ReadinessState()
    readiness.set_model_ready(True)
    readiness.set_ha_connected(True)

    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="no-snapshot-1"))
    ws.push(None)

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=shared,
        readiness=readiness,
    )

    assert session is not None
    assert session.graph_ready is False
    assert shared.snapshot is None
    assert readiness.snapshot().ha_connected is False


def _ws_handler(app: object) -> object:
    from aiohttp import web

    for route in app.router.routes():  # type: ignore[attr-defined]
        resource = route.resource
        if isinstance(resource, web.StaticResource):
            continue
        if resource.canonical == WS_PATH:
            return route.handler
    raise AssertionError(f"missing websocket route {WS_PATH}")

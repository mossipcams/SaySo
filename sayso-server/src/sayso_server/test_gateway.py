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
        self._sent_ready: asyncio.Event = asyncio.Event()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)
        self._sent_ready.set()

    async def wait_for_sent(self, *, min_count: int = 1, timeout: float = 2.0) -> None:
        if len(self.sent) >= min_count:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if len(self.sent) >= min_count:
                return
            self._sent_ready.clear()
            remaining = deadline - loop.time()
            try:
                await asyncio.wait_for(self._sent_ready.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
        msg = f"timed out waiting for {min_count} sent message(s); got {len(self.sent)}"
        raise AssertionError(msg)

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


def _conversation_envelope(
    *,
    correlation_id: str = "turn-1",
    source_id: str = "device-floor-lamp",
    area_id: str | None = "area_living_room",
    transcript: str = "turn off the floor lamp",
    stt_ms: float = 0.0,
) -> str:
    payload: dict[str, object] = {
        "transcript": transcript,
        "source_id": source_id,
    }
    if area_id is not None:
        payload["area_id"] = area_id
    if stt_ms > 0.0:
        payload["stt_ms"] = stt_ms
    return json.dumps(
        {
            "version": API_VERSION,
            "type": MessageType.CONVERSATION_REQUEST.value,
            "correlation_id": correlation_id,
            "payload": payload,
        },
    )


class RecordingConversationController:
    """Spy stand-in for gateway conversation routing tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response: dict[str, object] = {
            "category": "no_action",
            "reason": "clarification required: which light?",
            "plan": {"outcome": "clarification", "intent": "turn off the floor lamp"},
            "response_mode": "text",
            "response_content": "which light?",
        }
        self.raise_error: Exception | None = None

    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str | None,
        text: str,
        correlation_id: str,
        input_type: str = "text",
        stt_ms: float = 0.0,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "satellite_id": satellite_id,
                "area_id": area_id,
                "text": text,
                "correlation_id": correlation_id,
                "input_type": input_type,
                "stt_ms": stt_ms,
            },
        )
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


def _conversation_gateway_setup(
    controller: RecordingConversationController,
    *,
    with_graph: bool = True,
):
    shared = HomeGraphStore()
    if with_graph:
        shared.replace_snapshot(_load_graph_fixture())
    return shared, controller


async def _run_conversation_gateway(
    ws: FakeGatewayWebSocket,
    *,
    graph_store: HomeGraphStore,
    satellite_registry: object,
    text_controller: object,
    on_session_started: object | None = None,
) -> object:
    return await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=graph_store,
        satellite_registry=satellite_registry,
        text_controller=text_controller,
        on_session_started=on_session_started,
    )


def _sent_envelopes(ws: FakeGatewayWebSocket) -> list[dict[str, object]]:
    return [json.loads(message) for message in ws.sent]


def _action_requests(ws: FakeGatewayWebSocket) -> list[dict[str, object]]:
    return [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.ACTION_REQUEST.value
    ]


@pytest.mark.asyncio
async def test_conversation_request_invokes_text_controller_and_responds() -> None:
    controller = RecordingConversationController()
    graph_store, _controller = _conversation_gateway_setup(controller)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-1"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(_conversation_envelope(correlation_id="turn-corr-1"))
    ws.push(None)

    session = await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    assert session is not None
    assert controller.calls == [
        {
            "satellite_id": "device-floor-lamp",
            "area_id": "area_living_room",
            "text": "turn off the floor lamp",
            "correlation_id": "turn-corr-1",
            "input_type": "audio",
            "stt_ms": 0.0,
        },
    ]
    responses = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.CONVERSATION_RESPONSE.value
    ]
    assert len(responses) == 1
    assert responses[0]["correlation_id"] == "turn-corr-1"
    assert responses[0]["payload"]["response_type"] == "clarification"
    assert _action_requests(ws) == []


@pytest.mark.asyncio
async def test_malformed_conversation_request_returns_correlated_error() -> None:
    controller = RecordingConversationController()
    graph_store, _controller = _conversation_gateway_setup(controller)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-2"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(
        json.dumps(
            {
                "version": API_VERSION,
                "type": MessageType.CONVERSATION_REQUEST.value,
                "correlation_id": "turn-bad-1",
                "payload": {"transcript": ""},
            },
        ),
    )
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    assert controller.calls == []
    errors = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.ERROR.value
    ]
    assert len(errors) == 1
    assert errors[0]["correlation_id"] == "turn-bad-1"
    assert _action_requests(ws) == []


@pytest.mark.asyncio
async def test_conversation_request_rejects_unknown_area() -> None:
    controller = RecordingConversationController()
    graph_store, _controller = _conversation_gateway_setup(controller)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-unknown-area"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(
        _conversation_envelope(
            correlation_id="turn-unknown-area-1",
            area_id="area_deleted_room",
        ),
    )
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    assert controller.calls == []
    errors = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.ERROR.value
    ]
    assert len(errors) == 1
    assert errors[0]["correlation_id"] == "turn-unknown-area-1"
    assert errors[0]["payload"]["reason"] == "unknown_area"
    assert _action_requests(ws) == []


@pytest.mark.asyncio
async def test_conversation_request_uses_ha_supplied_area_not_satellite_registry() -> None:
    from sayso_server.satellites import SatelliteRegistry

    controller = RecordingConversationController()
    graph_store, _controller = _conversation_gateway_setup(controller)
    registry = SatelliteRegistry()
    registry.register("macbook", "area_kitchen")
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-ha-area"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(
        _conversation_envelope(
            correlation_id="turn-ha-area-1",
            source_id="device-floor-lamp",
            area_id="area_living_room",
        ),
    )
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=registry,
        text_controller=controller,
    )

    assert controller.calls == [
        {
            "satellite_id": "device-floor-lamp",
            "area_id": "area_living_room",
            "text": "turn off the floor lamp",
            "correlation_id": "turn-ha-area-1",
            "input_type": "audio",
            "stt_ms": 0.0,
        },
    ]


@pytest.mark.asyncio
async def test_conversation_request_without_area_passes_none_origin() -> None:
    controller = RecordingConversationController()
    graph_store, _controller = _conversation_gateway_setup(controller)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-no-area"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(
        _conversation_envelope(
            correlation_id="turn-no-area-1",
            area_id=None,
            transcript="turn off the floor lamp",
        ),
    )
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    assert controller.calls == [
        {
            "satellite_id": "device-floor-lamp",
            "area_id": None,
            "text": "turn off the floor lamp",
            "correlation_id": "turn-no-area-1",
            "input_type": "audio",
            "stt_ms": 0.0,
        },
    ]


@pytest.mark.asyncio
async def test_conversation_request_without_area_area_relative_returns_clarification() -> None:
    from sayso_server.ha_client import FakeHaClient
    from sayso_server.test_text_api import _ActionPlanRuntime
    from sayso_server.text_api import OrchestratorTextController

    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph_fixture())
    runtime = _ActionPlanRuntime(
        plan={
            "outcome": "action",
            "intent": "turn off the lights",
            "domain": "light",
            "scope": {"kind": "current_area"},
            "state": "off",
        },
    )
    runtime.load()
    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
    )
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-area-clarify"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(
        _conversation_envelope(
            correlation_id="turn-area-clarify-1",
            area_id=None,
            transcript="turn off the lights",
        ),
    )
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    responses = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.CONVERSATION_RESPONSE.value
    ]
    assert len(responses) == 1
    assert responses[0]["correlation_id"] == "turn-area-clarify-1"
    assert responses[0]["payload"]["response_type"] == "clarification"
    assert responses[0]["payload"]["speech"] == "which area?"
    assert _action_requests(ws) == []


@pytest.mark.asyncio
async def test_controller_failure_returns_correlated_error_without_action() -> None:
    controller = RecordingConversationController()
    controller.raise_error = RuntimeError("plan failed")
    graph_store, _controller = _conversation_gateway_setup(controller)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-3"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(_conversation_envelope(correlation_id="turn-fail-1"))
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    assert len(controller.calls) == 1
    errors = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.ERROR.value
    ]
    assert len(errors) == 1
    assert errors[0]["correlation_id"] == "turn-fail-1"
    assert _action_requests(ws) == []


@pytest.mark.asyncio
async def test_conversation_no_action_turn_without_action_request() -> None:
    from sayso_server.runtime import FakeModelRuntime
    from sayso_server.text_api import OrchestratorTextController

    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph_fixture())
    from sayso_server.ha_client import FakeHaClient

    runtime = FakeModelRuntime()
    runtime.load()
    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
    )
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-4"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(
        _conversation_envelope(
            correlation_id="turn-no-action-1",
            transcript="how many lights are on?",
        ),
    )
    ws.push(None)

    await _run_conversation_gateway(
        ws,
        graph_store=graph_store,
        satellite_registry=None,
        text_controller=controller,
    )

    responses = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.CONVERSATION_RESPONSE.value
    ]
    assert len(responses) == 1
    assert responses[0]["correlation_id"] == "turn-no-action-1"
    assert _action_requests(ws) == []


@pytest.mark.asyncio
async def test_conversation_action_turn_sends_action_request_and_responds() -> None:
    from sayso_server.session import HaGatewayBinding
    from sayso_server.test_ha_ws_client import _action_result_envelope
    from sayso_server.test_text_api import _ActionPlanRuntime
    from sayso_server.text_api import create_live_text_controller

    binding = HaGatewayBinding()
    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph_fixture())
    runtime = _ActionPlanRuntime()
    runtime.load()
    controller = create_live_text_controller(
        binding,
        runtime=runtime,
        graph_store=graph_store,
    )
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-conv-5"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    request_id = "turn-action-1"

    async def respond_to_action() -> None:
        assert binding.session is not None
        await asyncio.wait_for(binding.session.wait_for_outbound(), timeout=2.0)
        await ws.wait_for_sent(min_count=1, timeout=2.0)
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status="accepted",
                correlation_id="hello-conv-5",
            ),
        )
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status="completed",
                reason="state_changed",
                correlation_id="hello-conv-5",
            ),
        )
        ws.push(None)

    ws.push(_conversation_envelope(correlation_id=request_id))

    def attach_binding(session: object, bound_ws: object) -> None:
        binding.attach(session, bound_ws)  # type: ignore[arg-type]

    gateway_task = asyncio.create_task(
        _run_conversation_gateway(
            ws,
            graph_store=graph_store,
            satellite_registry=None,
            text_controller=controller,
            on_session_started=attach_binding,
        ),
    )
    responder = asyncio.create_task(respond_to_action())

    await gateway_task
    await responder

    action_requests = _action_requests(ws)
    assert len(action_requests) == 1
    assert action_requests[0]["payload"]["entity_id"] == "light.floor_lamp"
    responses = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.CONVERSATION_RESPONSE.value
    ]
    assert len(responses) == 1
    assert responses[0]["correlation_id"] == request_id
    assert responses[0]["payload"]["response_type"] == "action_done"


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


def _prepare_envelope(*, correlation_id: str = "prep-1") -> str:
    return json.dumps(
        {
            "version": API_VERSION,
            "type": MessageType.PREPARE.value,
            "correlation_id": correlation_id,
            "payload": {},
        },
    )


@pytest.mark.asyncio
async def test_prepare_request_reports_readiness_without_executing_action() -> None:
    controller = RecordingConversationController()
    graph_store, _controller = _conversation_gateway_setup(controller)
    readiness = ReadinessState()
    readiness.set_model_ready(True)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-prep-1"))
    ws.push(
        _graph_envelope(
            msg_type=MessageType.GRAPH_SNAPSHOT.value,
            payload=_load_graph_fixture().model_dump(mode="json"),
        ),
    )
    ws.push(_prepare_envelope(correlation_id="prep-corr-1"))
    ws.push(None)

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=graph_store,
        readiness=readiness,
        text_controller=controller,
    )

    assert session is not None
    assert controller.calls == []
    assert _action_requests(ws) == []
    responses = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.PREPARE_RESPONSE.value
    ]
    assert len(responses) == 1
    assert responses[0]["correlation_id"] == "prep-corr-1"
    assert responses[0]["payload"] == {
        "connected": True,
        "graph_ready": True,
        "model_ready": True,
    }


@pytest.mark.asyncio
async def test_prepare_before_graph_snapshot_reports_not_ready() -> None:
    readiness = ReadinessState()
    readiness.set_model_ready(True)
    ws = FakeGatewayWebSocket()
    ws.push(_hello_envelope(correlation_id="hello-prep-2"))
    ws.push(_prepare_envelope(correlation_id="prep-corr-2"))
    ws.push(None)

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        readiness=readiness,
    )

    assert session is not None
    responses = [
        envelope
        for envelope in _sent_envelopes(ws)
        if envelope.get("type") == MessageType.PREPARE_RESPONSE.value
    ]
    assert len(responses) == 1
    assert responses[0]["correlation_id"] == "prep-corr-2"
    assert responses[0]["payload"] == {
        "connected": False,
        "graph_ready": False,
        "model_ready": True,
    }


@pytest.mark.asyncio
async def test_accepted_action_result_does_not_resolve_terminal_future() -> None:
    from sayso_server.gateway import _record_action_result
    from sayso_server.results import ActionResultStatus
    from sayso_server.session import HaSession

    session = HaSession(correlation_id="corr-future-accepted", graph=HomeGraphStore())
    waiter = asyncio.create_task(session.collect_action_results("req-a", timeout=0.2))

    await asyncio.sleep(0)
    _record_action_result(
        session,
        {
            "request_id": "req-a",
            "status": ActionResultStatus.ACCEPTED.value,
        },
    )

    assert not waiter.done()

    _record_action_result(
        session,
        {
            "request_id": "req-a",
            "status": ActionResultStatus.COMPLETED.value,
            "reason": "state_changed",
        },
    )

    results = await asyncio.wait_for(waiter, timeout=1.0)
    assert [result.status for result in results] == [
        ActionResultStatus.ACCEPTED,
        ActionResultStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_terminal_action_results_resolve_matching_request_future() -> None:
    from sayso_server.gateway import _record_action_result
    from sayso_server.results import ActionResultStatus
    from sayso_server.session import HaSession

    session = HaSession(correlation_id="corr-future-terminal", graph=HomeGraphStore())

    for status, reason in (
        (ActionResultStatus.REJECTED, "permission_denied"),
        (ActionResultStatus.FAILED, "service_failed"),
        (
            ActionResultStatus.COMPLETED,
            "state_unchanged",
        ),
        (
            ActionResultStatus.FAILED,
            "state_verification_timeout",
        ),
    ):
        request_id = f"req-{status.value}"
        waiter = asyncio.create_task(
            session.collect_action_results(request_id, timeout=0.2),
        )
        await asyncio.sleep(0)
        if status is ActionResultStatus.COMPLETED:
            _record_action_result(
                session,
                {
                    "request_id": request_id,
                    "status": ActionResultStatus.ACCEPTED.value,
                },
            )
        _record_action_result(
            session,
            {
                "request_id": request_id,
                "status": status.value,
                "reason": reason,
            },
        )
        results = await asyncio.wait_for(waiter, timeout=1.0)
        assert results[-1].status is status
        assert results[-1].reason == reason


@pytest.mark.asyncio
async def test_action_result_for_other_request_id_does_not_resolve_future() -> None:
    from sayso_server.gateway import _record_action_result
    from sayso_server.results import ActionResultStatus
    from sayso_server.session import HaSession

    session = HaSession(correlation_id="corr-future-isolated", graph=HomeGraphStore())
    waiter = asyncio.create_task(session.collect_action_results("req-a", timeout=0.2))
    await asyncio.sleep(0)

    _record_action_result(
        session,
        {
            "request_id": "req-b",
            "status": ActionResultStatus.COMPLETED.value,
            "reason": "state_changed",
        },
    )

    assert not waiter.done()

    _record_action_result(
        session,
        {
            "request_id": "req-a",
            "status": ActionResultStatus.ACCEPTED.value,
        },
    )
    _record_action_result(
        session,
        {
            "request_id": "req-a",
            "status": ActionResultStatus.COMPLETED.value,
            "reason": "state_changed",
        },
    )

    results = await asyncio.wait_for(waiter, timeout=1.0)
    assert [result.request_id for result in results] == ["req-a", "req-a"]
    req_b_results = session.take_action_results("req-b")
    assert len(req_b_results) == 1


@pytest.mark.asyncio
async def test_action_result_timeout_clears_pending_future() -> None:
    from sayso_server.session import HaSession

    session = HaSession(correlation_id="corr-timeout", graph=HomeGraphStore())
    results = await session.collect_action_results("req-timeout", timeout=0.05)
    assert results == []
    assert session._action_futures == {}


@pytest.mark.asyncio
async def test_action_result_wait_cancellation_clears_pending_future() -> None:
    from sayso_server.session import HaSession

    session = HaSession(correlation_id="corr-cancel", graph=HomeGraphStore())
    waiter = asyncio.create_task(session.collect_action_results("req-cancel", timeout=30.0))
    await asyncio.sleep(0)
    assert "req-cancel" in session._action_futures

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert session._action_futures == {}


@pytest.mark.asyncio
async def test_session_detach_cancels_pending_action_futures() -> None:
    from sayso_server.session import HaGatewayBinding, HaSession

    binding = HaGatewayBinding()
    session = HaSession(correlation_id="corr-detach", graph=HomeGraphStore())
    ws = FakeGatewayWebSocket()
    binding.attach(session, ws)

    waiter = asyncio.create_task(session.collect_action_results("req-detach", timeout=30.0))
    await asyncio.sleep(0)
    assert "req-detach" in session._action_futures

    binding.detach()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert session._action_futures == {}
    assert binding.session is None


@pytest.mark.asyncio
async def test_disconnect_clears_pending_action_futures() -> None:
    from sayso_server.gateway import _process_graph_messages
    from sayso_server.session import HaSession

    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-disconnect", graph=HomeGraphStore())
    waiter = asyncio.create_task(session.collect_action_results("req-disconnect", timeout=30.0))
    await asyncio.sleep(0)

    processor = asyncio.create_task(_process_graph_messages(ws, session))
    ws.push(None)
    await processor

    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert session._action_futures == {}


@pytest.mark.asyncio
async def test_pending_action_waits_do_not_leak_across_turns() -> None:
    from sayso_server.gateway import _record_action_result
    from sayso_server.results import ActionResultStatus
    from sayso_server.session import HaSession

    session = HaSession(correlation_id="corr-turns", graph=HomeGraphStore())

    first = asyncio.create_task(session.collect_action_results("req-turn-1", timeout=0.2))
    await asyncio.sleep(0)
    _record_action_result(
        session,
        {
            "request_id": "req-turn-1",
            "status": ActionResultStatus.COMPLETED.value,
            "reason": "state_changed",
        },
    )
    await first
    assert session._action_futures == {}

    second = asyncio.create_task(session.collect_action_results("req-turn-2", timeout=0.05))
    await second
    assert session._action_futures == {}

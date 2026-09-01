"""HTTP contract tests for POST /api/v1/text."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sayso_server.api import API_VERSION
from sayso_server.const import TEXT_PATH
from sayso_server.graph_store import HomeGraphStore
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.satellites import SatelliteRegistry
from sayso_server.text_api import create_text_handler

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _text_request(
    *,
    correlation_id: str = "corr-text-1",
    satellite_id: str = "macbook",
    text: str = "turn off the floor lamp",
) -> dict[str, object]:
    return {
        "version": API_VERSION,
        "type": "text",
        "correlation_id": correlation_id,
        "payload": {
            "satellite_id": satellite_id,
            "text": text,
        },
    }


class RecordingTextController:
    """Spy stand-in for the text execution controller."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response: dict[str, object] = {
            "category": "completed",
            "reason": "state_changed",
            "plan": {"outcome": "action", "intent": "turn off the floor lamp"},
        }

    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str,
        text: str,
        correlation_id: str,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "satellite_id": satellite_id,
                "area_id": area_id,
                "text": text,
                "correlation_id": correlation_id,
            },
        )
        return self.response


def _build_handler(
    controller: RecordingTextController,
    *,
    satellite_id: str = "macbook",
    area_id: str = "area_living_room",
    with_graph: bool = True,
):
    registry = SatelliteRegistry()
    registry.register(satellite_id, area_id)
    graph_store = HomeGraphStore()
    if with_graph:
        graph_store.replace_snapshot(_load_graph())
    return create_text_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        text_controller=controller,
    )


async def _post_text(
    handler,
    body: dict[str, object] | str,
    *,
    token: str | None = "secret-token",
) -> tuple[int, dict[str, object] | None]:
    request = MagicMock()
    header_values: dict[str, str] = {"Content-Type": "application/json"}
    if token is not None:
        header_values["Authorization"] = f"Bearer {token}"
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: header_values.get(key, default)
    request.text = AsyncMock(return_value=body if isinstance(body, str) else json.dumps(body))
    response = await handler(request)
    if response.body is None:
        return response.status, None
    payload = json.loads(response.text)
    return response.status, payload


def test_text_path_constant() -> None:
    assert TEXT_PATH == "/api/v1/text"


def test_create_aiohttp_app_registers_text_route() -> None:
    from sayso_server.app import create_aiohttp_app
    from sayso_server.health import HEALTH_PATH
    from sayso_server.session import HaGatewayBinding
    from sayso_server.text_api import OrchestratorTextController

    app = create_aiohttp_app("secret-token")
    paths = {route.resource.canonical for route in app.router.routes()}
    assert TEXT_PATH in paths
    assert HEALTH_PATH in paths
    assert isinstance(app["text_controller"], OrchestratorTextController)
    assert app["graph_store"] is app["text_controller"]._graph_store

    shared_binding = HaGatewayBinding()
    wired_app = create_aiohttp_app("secret-token", ha_gateway_binding=shared_binding)
    assert wired_app["ha_gateway_binding"] is shared_binding


def _text_route_handler(app):
    for route in app.router.routes():
        if route.resource.canonical == TEXT_PATH:
            return route.handler
    raise AssertionError(f"missing route {TEXT_PATH}")


@pytest.mark.asyncio
async def test_default_text_endpoint_not_503_not_configured() -> None:
    from sayso_server.app import create_aiohttp_app

    app = create_aiohttp_app("secret-token")
    registry = app["satellite_registry"]
    registry.register("macbook", "area_living_room")
    app["graph_store"].replace_snapshot(_load_graph())

    handler = _text_route_handler(app)
    status, body = await _post_text(handler, _text_request())
    assert status == 200
    assert body is not None
    assert body["type"] == "text_response"
    assert body.get("payload", {}).get("code") != "not_configured"


@pytest.mark.asyncio
async def test_explicit_text_controller_overrides_default() -> None:
    from sayso_server.app import create_aiohttp_app

    controller = RecordingTextController()
    app = create_aiohttp_app("secret-token", text_controller=controller)
    registry = app["satellite_registry"]
    registry.register("macbook", "area_living_room")
    app["graph_store"].replace_snapshot(_load_graph())

    handler = _text_route_handler(app)
    status, body = await _post_text(handler, _text_request(correlation_id="override-1"))
    assert status == 200
    assert body is not None
    assert body["correlation_id"] == "override-1"
    assert controller.calls == [
        {
            "satellite_id": "macbook",
            "area_id": "area_living_room",
            "text": "turn off the floor lamp",
            "correlation_id": "override-1",
        },
    ]


@pytest.mark.asyncio
async def test_default_text_controller_shares_graph_store_with_gateway() -> None:
    import asyncio

    from sayso_server.app import create_aiohttp_app
    from sayso_server.gateway import handle_ha_connection
    from sayso_server.messages import MessageType

    class GatewayWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self._recv_queue: asyncio.Queue[str | None] = asyncio.Queue()
            self._recv_queue.put_nowait(
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.HELLO.value,
                        "correlation_id": "default-graph-1",
                        "payload": {},
                    },
                ),
            )
            self._recv_queue.put_nowait(
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.GRAPH_SNAPSHOT.value,
                        "correlation_id": "default-graph-2",
                        "payload": _load_graph().model_dump(mode="json"),
                    },
                ),
            )

        async def send_str(self, data: str) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def receive_str(self) -> str | None:
            return await self._recv_queue.get()

    app = create_aiohttp_app("secret-token")
    registry = app["satellite_registry"]
    registry.register("macbook", "area_living_room")
    ws = GatewayWebSocket()

    gateway_task = asyncio.create_task(
        handle_ha_connection(
            ws,
            authorization="Bearer secret-token",
            server_token="secret-token",
            graph_store=app["graph_store"],
        ),
    )
    for _ in range(200):
        if app["graph_store"].snapshot is not None:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("graph snapshot never arrived")

    assert app["text_controller"]._graph_store is app["graph_store"]

    handler = _text_route_handler(app)
    status, body = await _post_text(handler, _text_request())
    assert status == 200
    assert body is not None
    assert body["type"] == "text_response"

    ws._recv_queue.put_nowait(None)
    await gateway_task
    assert app["graph_store"].snapshot is None


@pytest.mark.asyncio
async def test_missing_auth_returns_401() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller)
    status, body = await _post_text(handler, _text_request(), token=None)
    assert status == 401
    assert body is None
    assert controller.calls == []


@pytest.mark.asyncio
async def test_invalid_json_returns_400() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller)
    status, body = await _post_text(handler, "{not-json")
    assert status == 400
    assert body is not None
    assert body["type"] == "error"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_invalid_envelope_returns_400() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller)
    status, body = await _post_text(
        handler,
        {"version": 2, "type": "text", "correlation_id": "x", "payload": {}},
    )
    assert status == 400
    assert body is not None
    assert body["type"] == "error"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_unknown_satellite_never_reaches_controller() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller)
    status, body = await _post_text(handler, _text_request(satellite_id="unknown-sat"))
    assert status == 400
    assert body is not None
    assert body["type"] == "error"
    assert body["payload"]["code"] == "unknown_satellite"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_unknown_area_never_reaches_controller() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller, area_id="area_missing")
    status, body = await _post_text(handler, _text_request())
    assert status == 400
    assert body is not None
    assert body["type"] == "error"
    assert body["payload"]["code"] == "unknown_area"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_missing_graph_never_reaches_controller() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller, with_graph=False)
    status, body = await _post_text(handler, _text_request())
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "no_graph"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_text_api_reads_graph_after_gateway_ingest() -> None:
    import asyncio

    from sayso_server.gateway import handle_ha_connection
    from sayso_server.messages import MessageType

    class GatewayWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self._recv_queue: asyncio.Queue[str | None] = asyncio.Queue()
            self._recv_queue.put_nowait(
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.HELLO.value,
                        "correlation_id": "text-graph-1",
                        "payload": {},
                    },
                ),
            )
            self._recv_queue.put_nowait(
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.GRAPH_SNAPSHOT.value,
                        "correlation_id": "text-graph-2",
                        "payload": _load_graph().model_dump(mode="json"),
                    },
                ),
            )

        async def send_str(self, data: str) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def receive_str(self) -> str | None:
            return await self._recv_queue.get()

    from sayso_server.app import create_aiohttp_app

    controller = RecordingTextController()
    graph_store = HomeGraphStore()
    app = create_aiohttp_app(
        "secret-token",
        text_controller=controller,
        graph_store=graph_store,
    )
    registry = SatelliteRegistry()
    registry.register("macbook", "area_living_room")
    app["satellite_registry"] = registry
    ws = GatewayWebSocket()

    gateway_task = asyncio.create_task(
        handle_ha_connection(
            ws,
            authorization="Bearer secret-token",
            server_token="secret-token",
            graph_store=app["graph_store"],
        ),
    )
    for _ in range(200):
        if app["graph_store"].snapshot is not None:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("graph snapshot never arrived")

    handler = create_text_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=app["graph_store"],
        text_controller=controller,
    )
    status, body = await _post_text(handler, _text_request())
    assert status == 200
    assert body is not None
    assert body["type"] == "text_response"
    assert controller.calls

    ws._recv_queue.put_nowait(None)
    await gateway_task


@pytest.mark.asyncio
async def test_valid_request_returns_text_response_envelope() -> None:
    controller = RecordingTextController()
    handler = _build_handler(controller)
    status, body = await _post_text(handler, _text_request(correlation_id="corr-ok"))
    assert status == 200
    assert body is not None
    assert body["version"] == API_VERSION
    assert body["type"] == "text_response"
    assert body["correlation_id"] == "corr-ok"
    assert body["payload"]["category"] == "completed"
    assert controller.calls == [
        {
            "satellite_id": "macbook",
            "area_id": "area_living_room",
            "text": "turn off the floor lamp",
            "correlation_id": "corr-ok",
        },
    ]


def test_orchestrator_text_controller_composes_candidates_prompt_and_parse() -> None:
    from sayso_server.ha_client import FakeHaClient
    from sayso_server.runtime import ModelMetadata, ModelRuntime, RawGenerationResult
    from sayso_server.text_api import OrchestratorTextController

    graph = _load_graph()
    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(graph)

    class RecordingRuntime(ModelRuntime):
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self._loaded = False

        def load(self) -> None:
            self._loaded = True

        def generate(self, prompt: str) -> RawGenerationResult:
            self.prompts.append(prompt)
            payload = json.loads(prompt)
            user_text = payload["user_text"]
            return RawGenerationResult(
                text=json.dumps(
                    {
                        "outcome": "query",
                        "intent": user_text,
                        "domain": "light",
                    }
                ),
                prompt_tokens=10,
                completion_tokens=2,
                latency_ms=1.0,
                metadata=ModelMetadata(model_id="recording", runtime="fake"),
            )

    runtime = RecordingRuntime()
    runtime.load()
    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
    )

    result = controller.handle(
        satellite_id="macbook",
        area_id="area_living_room",
        text="turn off the floor lamp",
        correlation_id="compose-1",
    )

    assert runtime.prompts
    prompt = runtime.prompts[0]
    assert "turn off the floor lamp" in prompt
    assert "Floor Lamp" in prompt
    assert "category" in result
    assert "plan" in result


class _ActionPlanRuntime:
    """Runtime stub that emits a resolvable light action plan."""

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def generate(self, prompt: str) -> object:
        from sayso_server.runtime import ModelMetadata, RawGenerationResult

        if not self._loaded:
            msg = "model runtime must be loaded before generate"
            raise RuntimeError(msg)

        payload = json.loads(prompt)
        user_text = payload["user_text"]
        return RawGenerationResult(
            text=json.dumps(
                {
                    "outcome": "action",
                    "intent": user_text,
                    "domain": "light",
                    "targets": ["floor lamp"],
                    "state": "off",
                }
            ),
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
            metadata=ModelMetadata(model_id="action-stub", runtime="fake"),
        )


def _live_text_setup(*, attach: bool):
    from sayso_server.session import HaGatewayBinding, HaSession
    from sayso_server.test_gateway import FakeGatewayWebSocket
    from sayso_server.text_api import create_live_text_controller

    binding = HaGatewayBinding()
    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    ws = FakeGatewayWebSocket()
    session = HaSession(correlation_id="corr-live", graph=graph_store)
    if attach:
        binding.attach(session, ws)
    runtime = _ActionPlanRuntime()
    runtime.load()
    controller = create_live_text_controller(
        binding,
        runtime=runtime,
        graph_store=graph_store,
    )
    registry = SatelliteRegistry()
    registry.register("macbook", "area_living_room")
    return binding, session, ws, controller, registry, graph_store


@pytest.mark.asyncio
async def test_detached_ha_refuses_text_execution_without_action_request() -> None:
    binding, session, _ws, controller, registry, graph_store = _live_text_setup(attach=False)
    handler = create_text_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        text_controller=controller,
    )

    status, body = await _post_text(handler, _text_request(correlation_id="detached-1"))

    assert status == 200
    assert body is not None
    assert body["type"] == "text_response"
    assert body["payload"]["category"] == "no_action"
    assert body["payload"]["reason"] == "home assistant websocket is not connected"
    assert session.drain_outbound() == []


@pytest.mark.asyncio
async def test_detached_ha_handler_guard_returns_503_when_require_live_ha() -> None:
    binding, _session, _ws, controller, registry, graph_store = _live_text_setup(attach=False)
    handler = create_text_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        text_controller=controller,
        ha_gateway_binding=binding,
        require_live_ha=True,
    )

    status, body = await _post_text(handler, _text_request(correlation_id="detached-2"))

    assert status == 503
    assert body is not None
    assert body["type"] == "error"
    assert body["payload"]["code"] == "ha_disconnected"


@pytest.mark.asyncio
async def test_connected_ha_with_snapshot_executes_unique_name_command() -> None:
    import asyncio

    from sayso_server.gateway import _process_graph_messages
    from sayso_server.messages import MessageType
    from sayso_server.test_ha_ws_client import _action_result_envelope

    binding, session, ws, controller, registry, graph_store = _live_text_setup(attach=True)
    processor = asyncio.create_task(_process_graph_messages(ws, session))
    request_id = "corr-connected-1"

    async def respond() -> None:
        for _ in range(200):
            if ws.sent:
                break
            await asyncio.sleep(0)
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status="accepted",
                correlation_id="corr-live",
            ),
        )
        ws.push(
            _action_result_envelope(
                request_id=request_id,
                status="completed",
                reason="state_changed",
                correlation_id="corr-live",
            ),
        )

    responder = asyncio.create_task(respond())
    handler = create_text_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        text_controller=controller,
    )

    status, body = await _post_text(
        handler,
        _text_request(correlation_id=request_id),
    )

    assert status == 200
    assert body is not None
    assert body["payload"]["category"] == "completed"
    assert len(ws.sent) == 1
    sent = json.loads(ws.sent[0])
    assert sent["type"] == MessageType.ACTION_REQUEST.value
    assert sent["payload"]["entity_id"] == "light.floor_lamp"
    assert sent["payload"]["action"] == "off"

    ws.push(None)
    await responder
    await processor


@pytest.mark.asyncio
async def test_default_live_app_refuses_when_ha_gateway_detached() -> None:
    from sayso_server.app import create_aiohttp_app

    app = create_aiohttp_app("secret-token")
    app["satellite_registry"].register("macbook", "area_living_room")
    app["graph_store"].replace_snapshot(_load_graph())
    assert app["ha_gateway_binding"].is_attached is False

    handler = _text_route_handler(app)
    status, body = await _post_text(handler, _text_request(correlation_id="app-detached"))

    assert status == 200
    assert body is not None
    assert body["payload"]["category"] == "no_action"
    assert body["payload"]["reason"] == "home assistant websocket is not connected"

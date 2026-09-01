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
    from sayso_server.app import create_aiohttp_app
    from sayso_server.gateway import handle_ha_connection
    from sayso_server.messages import MessageType

    class GatewayWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self._messages = [
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.HELLO.value,
                        "correlation_id": "default-graph-1",
                        "payload": {},
                    },
                ),
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.GRAPH_SNAPSHOT.value,
                        "correlation_id": "default-graph-2",
                        "payload": _load_graph().model_dump(mode="json"),
                    },
                ),
                None,
            ]
            self._index = 0

        async def send_str(self, data: str) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def receive_str(self) -> str | None:
            message = self._messages[self._index]
            self._index += 1
            return message

    app = create_aiohttp_app("secret-token")
    registry = app["satellite_registry"]
    registry.register("macbook", "area_living_room")

    await handle_ha_connection(
        GatewayWebSocket(),
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=app["graph_store"],
    )

    assert app["graph_store"].snapshot is not None
    assert app["text_controller"]._graph_store is app["graph_store"]

    handler = _text_route_handler(app)
    status, body = await _post_text(handler, _text_request())
    assert status == 200
    assert body is not None
    assert body["type"] == "text_response"


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
    from sayso_server.gateway import handle_ha_connection
    from sayso_server.messages import MessageType

    class GatewayWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self._messages = [
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.HELLO.value,
                        "correlation_id": "text-graph-1",
                        "payload": {},
                    },
                ),
                json.dumps(
                    {
                        "version": API_VERSION,
                        "type": MessageType.GRAPH_SNAPSHOT.value,
                        "correlation_id": "text-graph-2",
                        "payload": _load_graph().model_dump(mode="json"),
                    },
                ),
                None,
            ]
            self._index = 0

        async def send_str(self, data: str) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def receive_str(self) -> str | None:
            message = self._messages[self._index]
            self._index += 1
            return message

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

    await handle_ha_connection(
        GatewayWebSocket(),
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=app["graph_store"],
    )

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

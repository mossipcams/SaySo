"""Minimal HTTP and WebSocket surface for SaySo server."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from aiohttp import web

from sayso_server.auth import bearer_token_valid
from sayso_server.const import READINESS_PATH, TEXT_PATH, WS_PATH
from sayso_server.graph_store import HomeGraphStore
from sayso_server.messages import MessageType
from sayso_server.satellites import SatelliteRegistry
from sayso_server.text_api import TextController, create_live_text_controller, create_text_handler
from sayso_server.gateway import handle_ha_connection
from sayso_server.health import HEALTH_PATH
from sayso_server.readiness import ReadinessSnapshot, ReadinessState, liveness_response, readiness_response
from sayso_server.session import HaGatewayBinding, HaSession


class SaySoHTTPRequestHandler(BaseHTTPRequestHandler):
    """Serve GET /api/v1/health and /api/v1/ready with Bearer token auth."""

    server_token: ClassVar[str] = ""
    readiness_state: ClassVar[ReadinessState | None] = None

    def do_GET(self) -> None:
        if self.path not in {HEALTH_PATH, READINESS_PATH}:
            self.send_error(404)
            return

        snapshot = _snapshot_from_handler(self)
        authorization = self.headers.get("Authorization")
        if self.path == HEALTH_PATH:
            status, body = liveness_response(
                authorization=authorization,
                token=self.server_token,
                snapshot=snapshot,
            )
        else:
            status, body = readiness_response(
                authorization=authorization,
                token=self.server_token,
                snapshot=snapshot,
            )

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if body is not None:
            self.wfile.write(json.dumps(body).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default access logs to avoid leaking request metadata."""


def create_server(
    host: str,
    port: int,
    token: str,
    *,
    readiness: ReadinessState | None = None,
) -> ThreadingHTTPServer:
    """Create a threaded HTTP server bound to host:port."""

    SaySoHTTPRequestHandler.server_token = token
    SaySoHTTPRequestHandler.readiness_state = readiness or ReadinessState()
    return ThreadingHTTPServer((host, port), SaySoHTTPRequestHandler)


def _snapshot_from_handler(handler: SaySoHTTPRequestHandler) -> ReadinessSnapshot:
    state = handler.readiness_state
    if state is None:
        return ReadinessSnapshot(model_ready=False, ha_connected=False)
    return state.snapshot()


class _ReadinessTrackingGatewayWebSocket:
    """Track HA session lifetime for readiness while proxying gateway I/O."""

    def __init__(self, ws: web.WebSocketResponse, readiness: ReadinessState | None) -> None:
        self._ws = ws
        self._readiness = readiness
        self._marked_connected = False

    @property
    def closed(self) -> bool:
        return self._ws.closed

    async def send_str(self, data: str) -> None:
        await self._ws.send_str(data)
        if self._readiness is not None and not self._marked_connected and _is_hello_ack(data):
            self._readiness.set_ha_connected(True)
            self._marked_connected = True

    async def close(self) -> None:
        await self._ws.close()
        self._clear_connected()

    async def receive_str(self) -> str | None:
        message = await self._ws.receive()
        if message.type in {web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED}:
            self._clear_connected()
            return None
        if message.type != web.WSMsgType.TEXT:
            return None
        return message.data

    def _clear_connected(self) -> None:
        if self._readiness is not None and self._marked_connected:
            self._readiness.set_ha_connected(False)
            self._marked_connected = False


def _is_hello_ack(payload: str) -> bool:
    try:
        envelope = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return envelope.get("type") == MessageType.HELLO_ACK.value


def create_aiohttp_app(
    token: str,
    *,
    text_controller: TextController | None = None,
    satellite_registry: SatelliteRegistry | None = None,
    graph_store: HomeGraphStore | None = None,
    readiness: ReadinessState | None = None,
    ha_gateway_binding: HaGatewayBinding | None = None,
) -> web.Application:
    """Create an aiohttp app exposing health, text, and HA WebSocket endpoints."""

    app = web.Application()
    registry = satellite_registry or SatelliteRegistry()
    store = graph_store or HomeGraphStore()
    readiness_state = readiness or ReadinessState()
    binding = ha_gateway_binding if ha_gateway_binding is not None else HaGatewayBinding()
    controller = text_controller
    if controller is None:
        controller = create_live_text_controller(binding, graph_store=store)
    app["satellite_registry"] = registry
    app["graph_store"] = store
    app["text_controller"] = controller
    app["readiness"] = readiness_state
    app["ha_gateway_binding"] = binding

    async def health(request: web.Request) -> web.Response:
        status, body = liveness_response(
            authorization=request.headers.get("Authorization"),
            token=token,
            snapshot=readiness_state.snapshot(),
        )
        if body is None:
            return web.Response(status=status)
        return web.json_response(body, status=status)

    async def ready(request: web.Request) -> web.Response:
        status, body = readiness_response(
            authorization=request.headers.get("Authorization"),
            token=token,
            snapshot=readiness_state.snapshot(),
        )
        if body is None:
            return web.Response(status=status)
        return web.json_response(body, status=status)

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        if not bearer_token_valid(
            authorization=request.headers.get("Authorization"),
            expected_token=token,
        ):
            raise web.HTTPUnauthorized()

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        gateway_ws = _ReadinessTrackingGatewayWebSocket(ws, readiness_state)

        def on_session_started(session: HaSession, bound_ws: object) -> None:
            binding.attach(session, bound_ws)  # type: ignore[arg-type]

        try:
            await handle_ha_connection(
                gateway_ws,
                authorization=request.headers.get("Authorization"),
                server_token=token,
                graph_store=store,
                on_session_started=on_session_started,
            )
        finally:
            binding.detach()
            gateway_ws._clear_connected()
        return ws

    app.router.add_get(HEALTH_PATH, health)
    app.router.add_get(READINESS_PATH, ready)
    app.router.add_post(
        TEXT_PATH,
        create_text_handler(
            token=token,
            satellite_registry=registry,
            graph_store=store,
            text_controller=controller,
        ),
    )
    app.router.add_get(WS_PATH, websocket)
    return app

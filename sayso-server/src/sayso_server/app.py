"""Minimal HTTP and WebSocket surface for SaySo server."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from aiohttp import web

from sayso_server.auth import bearer_token_valid
from sayso_server.audio_api import create_audio_handler
from sayso_server.const import (
    AUDIO_PATH,
    HA_WS_MAX_MSG_SIZE,
    READINESS_PATH,
    TEXT_PATH,
    TOKEN_ENV_VAR,
    WS_PATH,
)
from sayso_server.conversation import ConversationStore
from sayso_server.graph_store import HomeGraphStore
from sayso_server.mlx_stt import MlxWhisperSttRuntime
from sayso_server.satellites import SatelliteRegistry, register_default_satellites
from sayso_server.stt import SpeechToTextRuntime
from sayso_server.runtime import ModelRuntime
from sayso_server.telemetry import open_jsonl_telemetry_sink_from_env
from sayso_server.text_api import TextController, create_live_text_controller, create_text_handler
from sayso_server.gateway import handle_ha_connection
from sayso_server.health import HEALTH_PATH
from sayso_server.readiness import ReadinessSnapshot, ReadinessState, liveness_response, readiness_response
from sayso_server.session import HaGatewayBinding, HaSession


class MissingServerTokenError(RuntimeError):
    """Raised when the server bearer token is not configured."""


def load_server_token(*, environ: Mapping[str, str] | None = None) -> str:
    """Load the bearer token from ``SAYSO_TOKEN``."""

    import os

    source = os.environ if environ is None else environ
    token = source.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise MissingServerTokenError(
            f"{TOKEN_ENV_VAR} environment variable is required",
        )
    return token


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


class _GatewayWebSocketProxy:
    """Proxy aiohttp WebSocket I/O for the HA session gateway."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self._ws = ws

    @property
    def closed(self) -> bool:
        return self._ws.closed

    async def send_str(self, data: str) -> None:
        await self._ws.send_str(data)

    async def close(self) -> None:
        await self._ws.close()

    async def receive_str(self) -> str | None:
        while True:
            message = await self._ws.receive()
            if message.type in {web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED}:
                return None
            if message.type in {web.WSMsgType.PING, web.WSMsgType.PONG}:
                continue
            if message.type != web.WSMsgType.TEXT:
                continue
            return message.data


def _runtime_is_loaded(runtime: ModelRuntime) -> bool:
    loaded = getattr(runtime, "_loaded", None)
    if loaded is None:
        return False
    if isinstance(loaded, bool):
        return loaded
    return True


def create_aiohttp_app(
    token: str,
    *,
    text_controller: TextController | None = None,
    model_runtime: ModelRuntime | None = None,
    stt_runtime: SpeechToTextRuntime | None = None,
    satellite_registry: SatelliteRegistry | None = None,
    graph_store: HomeGraphStore | None = None,
    readiness: ReadinessState | None = None,
    ha_gateway_binding: HaGatewayBinding | None = None,
) -> web.Application:
    """Create an aiohttp app exposing health, text, and HA WebSocket endpoints."""

    app = web.Application()
    registry = satellite_registry or SatelliteRegistry()
    if satellite_registry is None:
        register_default_satellites(registry)
    store = graph_store or HomeGraphStore()
    readiness_state = readiness or ReadinessState()
    if model_runtime is not None and _runtime_is_loaded(model_runtime):
        readiness_state.set_model_ready(True)
    binding = ha_gateway_binding if ha_gateway_binding is not None else HaGatewayBinding()
    controller = text_controller
    env_telemetry_sink = None
    if controller is None:
        env_telemetry_sink = open_jsonl_telemetry_sink_from_env()
        controller = create_live_text_controller(
            binding,
            graph_store=store,
            runtime=model_runtime,
            conversation_store=ConversationStore(ttl_seconds=300.0),
            telemetry_sink=env_telemetry_sink,
        )
    stt = stt_runtime or MlxWhisperSttRuntime()
    app["satellite_registry"] = registry
    app["graph_store"] = store
    app["text_controller"] = controller
    app["stt_runtime"] = stt
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

        ws = web.WebSocketResponse(max_msg_size=HA_WS_MAX_MSG_SIZE)
        await ws.prepare(request)
        gateway_ws = _GatewayWebSocketProxy(ws)

        def on_session_started(session: HaSession, bound_ws: object) -> None:
            binding.attach(session, bound_ws)  # type: ignore[arg-type]

        try:
            await handle_ha_connection(
                gateway_ws,
                authorization=request.headers.get("Authorization"),
                server_token=token,
                graph_store=store,
                readiness=readiness_state,
                on_session_started=on_session_started,
            )
        finally:
            binding.detach()
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
    app.router.add_post(
        AUDIO_PATH,
        create_audio_handler(
            token=token,
            satellite_registry=registry,
            graph_store=store,
            stt_runtime=stt,
            text_controller=controller,
        ),
    )
    app.router.add_get(WS_PATH, websocket)
    if env_telemetry_sink is not None:

        async def _close_env_telemetry_sink(_app: web.Application) -> None:
            env_telemetry_sink.close()

        app.on_cleanup.append(_close_env_telemetry_sink)
    return app

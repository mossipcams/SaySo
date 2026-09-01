"""Minimal HTTP and WebSocket surface for SaySo server."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from aiohttp import web

from sayso_server.auth import bearer_token_valid
from sayso_server.const import WS_PATH
from sayso_server.gateway import handle_ha_connection
from sayso_server.health import HEALTH_PATH, health_status


class SaySoHTTPRequestHandler(BaseHTTPRequestHandler):
    """Serve GET /api/v1/health with Bearer token auth."""

    server_token: ClassVar[str] = ""

    def do_GET(self) -> None:
        if self.path != HEALTH_PATH:
            self.send_error(404)
            return

        status = health_status(
            authorization=self.headers.get("Authorization"),
            token=self.server_token,
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if status == 200:
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default access logs to avoid leaking request metadata."""


def create_server(host: str, port: int, token: str) -> ThreadingHTTPServer:
    """Create a threaded HTTP server bound to host:port."""

    SaySoHTTPRequestHandler.server_token = token
    return ThreadingHTTPServer((host, port), SaySoHTTPRequestHandler)


class _AiohttpGatewayWebSocket:
    """Adapt aiohttp WebSocketResponse to the gateway protocol."""

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
        message = await self._ws.receive()
        if message.type in {web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED}:
            return None
        if message.type != web.WSMsgType.TEXT:
            return None
        return message.data


def create_aiohttp_app(token: str) -> web.Application:
    """Create an aiohttp app exposing health and HA WebSocket endpoints."""

    app = web.Application()

    async def health(request: web.Request) -> web.Response:
        status = health_status(
            authorization=request.headers.get("Authorization"),
            token=token,
        )
        if status != 200:
            return web.Response(status=status)
        return web.json_response({"status": "ok"})

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        if not bearer_token_valid(
            authorization=request.headers.get("Authorization"),
            expected_token=token,
        ):
            raise web.HTTPUnauthorized()

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await handle_ha_connection(
            _AiohttpGatewayWebSocket(ws),
            authorization=request.headers.get("Authorization"),
            server_token=token,
        )
        return ws

    app.router.add_get(HEALTH_PATH, health)
    app.router.add_get(WS_PATH, websocket)
    return app

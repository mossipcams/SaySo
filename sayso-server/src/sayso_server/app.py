"""Minimal HTTP surface for SaySo server."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

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

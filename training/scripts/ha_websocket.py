"""Minimal Home Assistant websocket client for the two facts REST cannot give.

``/api/states`` and ``/api/template`` carry neither *which entities are exposed to
Assist* nor *their aliases*, and both decide what a training snapshot may contain:
an entity Assist cannot see must never appear in the corpus, and an alias is how a
household actually names a device. Both live behind ``/api/websocket``.

stdlib only -- a client-side RFC 6455 text channel is about eighty lines, which is
cheaper than adding a websocket dependency to the training venv.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import struct
from typing import Any
from urllib.parse import urlparse

_TEXT, _BINARY, _CLOSE, _PING, _PONG = 0x1, 0x2, 0x8, 0x9, 0xA


def encode_frame(payload: bytes, opcode: int = _TEXT) -> bytes:
    """One masked client frame. Clients must mask; servers must not."""
    mask = os.urandom(4)
    length = len(payload)
    header = bytes([0x80 | opcode])
    if length < 126:
        header += bytes([0x80 | length])
    elif length < 1 << 16:
        header += bytes([0x80 | 126]) + struct.pack("!H", length)
    else:
        header += bytes([0x80 | 127]) + struct.pack("!Q", length)
    return header + mask + bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))


def decode_frame(stream) -> tuple[bool, int, bytes]:
    """Read one frame from a buffered reader: (fin, opcode, payload)."""
    head = stream.read(2)
    if len(head) < 2:
        raise ConnectionError("websocket closed while reading a frame header")
    fin = bool(head[0] & 0x80)
    opcode = head[0] & 0x0F
    masked = bool(head[1] & 0x80)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", stream.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", stream.read(8))[0]
    mask = stream.read(4) if masked else b""
    payload = stream.read(length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return fin, opcode, payload


class HomeAssistantWebSocket:
    """Authenticated request/response channel. Not a subscription client."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0) -> None:
        parsed = urlparse(base_url.rstrip("/"))
        self._secure = parsed.scheme == "https"
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or (443 if self._secure else 8123)
        self._path = f"{parsed.path}/api/websocket" if parsed.path else "/api/websocket"
        self._token = token
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._stream = None
        self._next_id = 1

    def __enter__(self) -> HomeAssistantWebSocket:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self) -> None:
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        if self._secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self._host)
        self._sock = sock
        self._stream = sock.makefile("rb")
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall(
            (
                f"GET {self._path} HTTP/1.1\r\n"
                f"Host: {self._host}:{self._port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        status = self._stream.readline()
        if b"101" not in status:
            raise ConnectionError(f"websocket upgrade refused: {status!r}")
        while self._stream.readline() not in (b"\r\n", b"\n", b""):
            pass
        greeting = self._receive()
        if greeting.get("type") != "auth_required":
            raise ConnectionError(f"unexpected greeting: {greeting}")
        self._send({"type": "auth", "access_token": self._token})
        result = self._receive()
        if result.get("type") != "auth_ok":
            raise PermissionError(f"Home Assistant rejected the token: {result}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.sendall(encode_frame(b"", _CLOSE))
            except OSError:
                pass
            self._sock.close()
        self._sock, self._stream = None, None

    def _send(self, message: dict[str, Any]) -> None:
        assert self._sock is not None
        self._sock.sendall(encode_frame(json.dumps(message).encode()))

    def _receive(self) -> dict[str, Any]:
        buffer = bytearray()
        while True:
            fin, opcode, payload = decode_frame(self._stream)
            if opcode == _PING:
                assert self._sock is not None
                self._sock.sendall(encode_frame(payload, _PONG))
                continue
            if opcode == _PONG:
                continue
            if opcode == _CLOSE:
                raise ConnectionError("Home Assistant closed the websocket")
            buffer += payload
            if fin:
                return json.loads(buffer.decode())

    def command(self, message: dict[str, Any]) -> Any:
        """Send one command and return its ``result``, raising on failure."""
        request_id = self._next_id
        self._next_id += 1
        self._send({**message, "id": request_id})
        while True:
            reply = self._receive()
            if reply.get("id") != request_id or reply.get("type") != "result":
                continue
            if not reply.get("success", False):
                raise RuntimeError(f"{message['type']} failed: {reply.get('error')}")
            return reply.get("result")


def fetch_assist_registry(base_url: str, token: str) -> dict[str, Any]:
    """Assist exposure and aliases, keyed by entity id.

    ``expose_entity/list`` is authoritative for exposure; aliases live on the
    individual registry entry, so they are fetched only for exposed entities.
    """
    with HomeAssistantWebSocket(base_url, token) as client:
        exposure = client.command({"type": "homeassistant/expose_entity/list"}) or {}
        exposed_map = exposure.get("exposed_entities", exposure)
        exposed = {
            entity_id
            for entity_id, settings in exposed_map.items()
            if (settings or {}).get("conversation")
        }
        aliases: dict[str, list[str]] = {}
        for entity_id in sorted(exposed):
            try:
                entry = client.command(
                    {"type": "config/entity_registry/get", "entity_id": entity_id}
                )
            except RuntimeError:
                continue  # entity exists in state machine but not in the registry
            if entry and entry.get("aliases"):
                aliases[entity_id] = list(entry["aliases"])
    return {"exposed": exposed, "aliases": aliases}

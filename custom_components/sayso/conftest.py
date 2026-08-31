"""Pytest configuration for SaySo Home Assistant integration tests."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp import WSMsgType


class FakeWebSocket:
    """Minimal async WebSocket stand-in for integration tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._recv_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False

    async def send_str(self, data: str) -> None:
        if self.closed:
            msg = "WebSocket is closed"
            raise ConnectionError(msg)
        self.sent.append(data)
        payload = json.loads(data)
        if payload.get("type") == "hello":
            self.push(
                json.dumps(
                    {
                        "version": 1,
                        "type": "hello_ack",
                        "correlation_id": payload["correlation_id"],
                        "payload": {},
                    },
                ),
            )

    async def receive(self) -> "_FakeMessage":
        if self.closed:
            return _FakeMessage(type=WSMsgType.CLOSE)
        message = await self._recv_queue.get()
        if message is None:
            self.closed = True
            return _FakeMessage(type=WSMsgType.CLOSE)
        return _FakeMessage(type=WSMsgType.TEXT, data=message)

    async def close(self) -> None:
        self.closed = True

    def push(self, message: str) -> None:
        self._recv_queue.put_nowait(message)

    def disconnect(self) -> None:
        self.closed = True
        self._recv_queue.put_nowait(None)


class _FakeMessage:
    type: WSMsgType
    data: str = ""

    def __init__(self, *, type: WSMsgType, data: str = "") -> None:
        self.type = type
        self.data = data


async def _fake_default_ws_connect(*_args, **_kwargs) -> FakeWebSocket:
    return FakeWebSocket()


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Load custom integrations for every test in this package."""
    yield


@pytest.fixture(autouse=True)
def mock_sayso_default_ws_connect():
    """Prevent integration tests from opening real network sockets."""

    with patch(
        "custom_components.sayso.coordinator._default_ws_connect",
        side_effect=_fake_default_ws_connect,
    ):
        yield

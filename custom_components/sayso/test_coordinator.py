"""Connection coordinator tests for SaySo."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import (
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    RECONNECT_INITIAL_DELAY,
    RECONNECT_MAX_DELAY,
    WS_PATH,
)
from custom_components.sayso.conftest import FakeWebSocket
from custom_components.sayso.coordinator import (
    SaySoConnectionCoordinator,
    _default_ws_connect as _real_default_ws_connect,
    http_to_ws_url,
)


@pytest.fixture
def fast_timing():
    """Use short coordinator timing so tests stay fast."""

    patched_heartbeat = 0.02
    with (
        patch(
            "custom_components.sayso.coordinator.RECONNECT_INITIAL_DELAY",
            0.01,
        ),
        patch(
            "custom_components.sayso.coordinator.RECONNECT_MAX_DELAY",
            0.05,
        ),
        patch(
            "custom_components.sayso.coordinator.HEARTBEAT_INTERVAL",
            patched_heartbeat,
        ),
    ):
        yield {"heartbeat_interval": patched_heartbeat}


@pytest.mark.asyncio
async def test_http_to_ws_url_converts_scheme_and_path() -> None:
    assert (
        http_to_ws_url("http://127.0.0.1:8765")
        == f"ws://127.0.0.1:8765{WS_PATH}"
    )
    assert (
        http_to_ws_url("https://sayso.example.com/")
        == f"wss://sayso.example.com{WS_PATH}"
    )


@pytest.mark.asyncio
async def test_disconnect_clears_connected_immediately(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()

    async def connect(_url: str, token: str) -> FakeWebSocket:
        assert token == "secret-token"
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    async def wait_until_connected() -> None:
        for _ in range(50):
            if coordinator.connected:
                return
            await asyncio.sleep(0.01)
        pytest.fail("coordinator never connected")

    await wait_until_connected()
    assert coordinator.connected is True

    fake_ws.disconnect()

    for _ in range(20):
        if not coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("connected stayed true after disconnect")

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_reconnect_after_disconnect_restores_connected(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    sockets: list[FakeWebSocket] = []

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        ws = FakeWebSocket()
        sockets.append(ws)
        return ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    async def wait_for_socket_count(count: int) -> None:
        for _ in range(100):
            if len(sockets) >= count:
                return
            await asyncio.sleep(0.01)
        pytest.fail(f"expected {count} socket(s), got {len(sockets)}")

    await wait_for_socket_count(1)
    assert coordinator.connected is True

    sockets[0].disconnect()

    for _ in range(50):
        if not coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("connected stayed true after first disconnect")

    await wait_for_socket_count(2)
    assert coordinator.connected is True

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_connected_sends_heartbeats(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("coordinator never connected")

    await asyncio.sleep(fast_timing["heartbeat_interval"] + 0.05)

    ping_messages = [
        json.loads(message)
        for message in fake_ws.sent
        if json.loads(message)["type"] == "ping"
    ]
    assert ping_messages, "expected at least one heartbeat ping"

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_ws_connect_uses_bearer_auth_not_logged(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    captured: dict[str, Any] = {}

    async def connect(url: str, token: str) -> FakeWebSocket:
        captured["url"] = url
        captured["token"] = token
        return FakeWebSocket()

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    with caplog.at_level("DEBUG", logger="custom_components.sayso.coordinator"):
        await coordinator.async_start()
        for _ in range(50):
            if coordinator.connected:
                break
            await asyncio.sleep(0.01)

    assert captured["url"] == f"ws://127.0.0.1:8765{WS_PATH}"
    assert captured["token"] == "secret-token"
    assert "secret-token" not in caplog.text

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_reconnect_backoff_delay_is_bounded(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    connect = AsyncMock(side_effect=ConnectionError("offline"))

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    delays: list[float] = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay: float) -> None:
        delays.append(delay)
        await original_sleep(0)

    with patch(
        "custom_components.sayso.coordinator.asyncio.sleep",
        side_effect=record_sleep,
    ):
        await coordinator.async_start()
        await original_sleep(0.2)
        await coordinator.async_stop()

    reconnect_delays = [delay for delay in delays if delay > 0]
    assert reconnect_delays
    assert max(reconnect_delays) <= RECONNECT_MAX_DELAY


@pytest.mark.asyncio
async def test_hello_ack_requires_wsmsgtype_not_string_type(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Fail if coordinator compares message.type to string literals."""

    from aiohttp import WSMsgType

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    hello_ack = json.dumps({"version": 1, "type": "hello_ack", "payload": {}})

    class StrictAiohttpMessageWebSocket(FakeWebSocket):
        async def receive(self) -> Any:
            if self.closed:
                return type(
                    "WSMessage",
                    (),
                    {"type": WSMsgType.CLOSE, "data": None},
                )()
            message = await self._recv_queue.get()
            if message is None:
                self.closed = True
                return type(
                    "WSMessage",
                    (),
                    {"type": WSMsgType.CLOSE, "data": None},
                )()
            # String comparison would never match real aiohttp enum values.
            assert WSMsgType.TEXT != "TEXT"
            return type(
                "WSMessage",
                (),
                {"type": WSMsgType.TEXT, "data": message},
            )()

    fake_ws = StrictAiohttpMessageWebSocket()

    async def connect(_url: str, _token: str) -> StrictAiohttpMessageWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("hello_ack not recognized — message.type may be compared to strings")

    await coordinator.async_stop()


@pytest.mark.asyncio
async def test_default_ws_connect_uses_hass_shared_session(
    hass: HomeAssistant,
) -> None:
    """Production connect must not allocate a private ClientSession per attempt."""

    from unittest.mock import MagicMock

    shared_session = MagicMock()
    shared_session.ws_connect = AsyncMock(return_value=FakeWebSocket())

    with (
        patch("aiohttp.ClientSession") as session_ctor,
        patch(
            "custom_components.sayso.coordinator.async_get_clientsession",
            return_value=shared_session,
        ),
        patch(
            "custom_components.sayso.coordinator._default_ws_connect",
            _real_default_ws_connect,
        ),
    ):
        await _real_default_ws_connect(hass, f"ws://127.0.0.1:8765{WS_PATH}", "token")

    session_ctor.assert_not_called()
    shared_session.ws_connect.assert_called_once()


@pytest.mark.asyncio
async def test_private_connect_session_closed_when_socket_ends(
    hass: HomeAssistant,
    fast_timing,
) -> None:
    """Per-connect sessions attached to a socket are closed when the socket ends."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    fake_ws = FakeWebSocket()
    fake_session = AsyncMock()
    fake_ws._sayso_session = fake_session

    async def connect(_url: str, _token: str) -> FakeWebSocket:
        return fake_ws

    coordinator = SaySoConnectionCoordinator(hass, entry, ws_connect=connect)
    await coordinator.async_start()

    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("coordinator never connected")

    fake_ws.disconnect()

    for _ in range(20):
        if fake_session.close.await_count:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("private connect session was not closed when socket ended")

    await coordinator.async_stop()


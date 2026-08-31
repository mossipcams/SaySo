"""SaySo outbound WebSocket connection coordinator."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

import aiohttp
from aiohttp import WSMsgType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_registry import (
    EVENT_ENTITY_REGISTRY_UPDATED,
    EventEntityRegistryUpdatedData,
    async_get as er_async_get,
)

from .action_mapping import map_action_to_ha_service
from .const import (
    API_VERSION,
    CONF_TOKEN,
    CONF_URL,
    HEARTBEAT_INTERVAL,
    MSG_ACTION_REQUEST,
    MSG_ACTION_RESULT,
    MSG_GRAPH_SNAPSHOT,
    MSG_REGISTRY_DELTA,
    MSG_STATE_DELTA,
    RECONNECT_BACKOFF_FACTOR,
    RECONNECT_INITIAL_DELAY,
    RECONNECT_MAX_DELAY,
    WS_PATH,
    get_entry_options,
)
from .deltas import build_registry_delta, build_state_delta
from .exposure import is_entity_id_exposed
from .permissions import entity_domain_from_id, validate_action_permission
from .snapshot import build_home_graph_snapshot

_LOGGER = logging.getLogger(__name__)


class WebSocketLike(Protocol):
    """Minimal WebSocket surface used by the coordinator."""

    closed: bool

    async def send_str(self, data: str) -> None: ...

    async def receive(self) -> Any: ...

    async def close(self) -> None: ...


WsConnect = Any  # Callable[[str, str], Awaitable[WebSocketLike]]
ServiceCaller = Callable[[str, str, dict[str, Any]], Any]


def http_to_ws_url(url: str) -> str:
    """Convert a configured HTTP(S) base URL to the SaySo WebSocket URL."""

    normalized = url.rstrip("/")
    if normalized.startswith("https://"):
        base = "wss://" + normalized.removeprefix("https://")
    elif normalized.startswith("http://"):
        base = "ws://" + normalized.removeprefix("http://")
    else:
        base = normalized
    return f"{base}{WS_PATH}"


def _envelope(message_type: str, *, payload: dict[str, Any] | None = None) -> str:
    return json.dumps(
        {
            "version": API_VERSION,
            "type": message_type,
            "correlation_id": uuid4().hex,
            "payload": payload or {},
        },
    )


class SaySoConnectionCoordinator:
    """Maintain an authenticated outbound WebSocket to the SaySo server."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        *,
        ws_connect: WsConnect | None = None,
        service_caller: ServiceCaller | None = None,
    ) -> None:
        self.hass = hass
        self.entry = entry
        self.connected = False
        self._ws_connect = ws_connect or (
            lambda url, token: _default_ws_connect(hass, url, token)
        )
        self._service_caller = service_caller or (
            lambda domain, service, data: _default_service_caller(
                hass,
                domain,
                service,
                data,
            )
        )
        self._stop_event = asyncio.Event()
        self._runner_task: asyncio.Task[None] | None = None
        self._ws: WebSocketLike | None = None
        self._sequence = 0
        self._home_id = entry.entry_id
        self._options = get_entry_options(entry)
        self._unsubscribers: list[Callable[[], None]] = []

    @property
    def url(self) -> str:
        return self.entry.data[CONF_URL]

    @property
    def token(self) -> str:
        return self.entry.data[CONF_TOKEN]

    async def async_start(self) -> None:
        """Start the connection loop."""

        if self._runner_task is not None:
            return
        self._register_listeners()
        self._stop_event.clear()
        self._runner_task = asyncio.create_task(self._run())

    async def async_stop(self) -> None:
        """Stop the connection loop and clear connected state."""

        self._stop_event.set()
        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()
        self._ws = None
        self.connected = False

    def _register_listeners(self) -> None:
        if self._unsubscribers:
            return
        self._unsubscribers.append(
            self.hass.bus.async_listen("state_changed", self._handle_state_changed),
        )
        self._unsubscribers.append(
            self.hass.bus.async_listen(
                EVENT_ENTITY_REGISTRY_UPDATED,
                self._handle_entity_registry_updated,
            ),
        )

    @callback
    def _handle_state_changed(self, event: Event) -> None:
        self.hass.async_create_task(self._async_send_state_delta(event))

    @callback
    def _handle_entity_registry_updated(
        self,
        event: Event[EventEntityRegistryUpdatedData],
    ) -> None:
        self.hass.async_create_task(self._async_send_registry_delta(event))

    async def _async_send_state_delta(self, event: Event) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return

        new_state = event.data.get("new_state")
        if new_state is None:
            return
        if not is_entity_id_exposed(
            self.hass,
            new_state.entity_id,
            self._options,
        ):
            return

        self._sequence += 1
        payload = build_state_delta(
            home_id=self._home_id,
            sequence=self._sequence,
            entity_id=new_state.entity_id,
            state=new_state,
        )
        await ws.send_str(_envelope(MSG_STATE_DELTA, payload=payload))

    async def _async_send_registry_delta(
        self,
        event: Event[EventEntityRegistryUpdatedData],
    ) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return

        data = event.data
        action = data["action"]
        entity_id = data["entity_id"]
        entry = None
        if action != "remove":
            entry = er_async_get(self.hass).async_get(entity_id)
            if entry is None:
                return
            if not is_entity_id_exposed(
                self.hass,
                entity_id,
                self._options,
                entry=entry,
            ):
                return

        self._sequence += 1
        payload = build_registry_delta(
            self.hass,
            home_id=self._home_id,
            sequence=self._sequence,
            change=action,
            entry=entry,
            entity_id=entity_id,
        )
        await ws.send_str(_envelope(MSG_REGISTRY_DELTA, payload=payload))

    async def _send_graph_snapshot(self, ws: WebSocketLike) -> None:
        self._sequence += 1
        payload = build_home_graph_snapshot(
            self.hass,
            home_id=self._home_id,
            sequence=self._sequence,
            options=self._options,
        )
        await ws.send_str(_envelope(MSG_GRAPH_SNAPSHOT, payload=payload))

    async def _run(self) -> None:
        delay = RECONNECT_INITIAL_DELAY
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                delay = RECONNECT_INITIAL_DELAY
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — reconnect on any connection failure
                _LOGGER.debug("SaySo connection failed; retrying in %.1fs", delay)

            if self._stop_event.is_set():
                break

            self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * RECONNECT_BACKOFF_FACTOR, RECONNECT_MAX_DELAY)

    async def _connect_once(self) -> None:
        ws_url = http_to_ws_url(self.url)
        ws = await self._ws_connect(ws_url, self.token)
        try:
            await ws.send_str(_envelope("hello"))
            if not await self._wait_for_hello_ack(ws):
                await ws.close()
                self.connected = False
                msg = "SaySo server did not acknowledge hello"
                raise ConnectionError(msg)

            self.connected = True
            self._ws = ws
            await self._send_graph_snapshot(ws)
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                await self._receive_loop(ws)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
        finally:
            self.connected = False
            self._ws = None
            if not ws.closed:
                await ws.close()
            await _close_ws_session(ws)

    async def _wait_for_hello_ack(self, ws: WebSocketLike) -> bool:
        while not ws.closed and not self._stop_event.is_set():
            message = await ws.receive()
            if message.type in {
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
            }:
                return False
            if message.type != WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "hello_ack":
                return True
        return False

    async def _receive_loop(self, ws: WebSocketLike) -> None:
        while not self._stop_event.is_set() and not ws.closed:
            message = await ws.receive()
            if message.type in {
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
            }:
                break
            if message.type != WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == MSG_ACTION_REQUEST:
                await self._handle_action_request(ws, payload)

    async def _handle_action_request(
        self,
        ws: WebSocketLike,
        envelope: dict[str, Any],
    ) -> None:
        request = envelope.get("payload") or {}
        request_id = request.get("request_id") or envelope.get("correlation_id")
        entity_id = request.get("entity_id")
        domain = request.get("domain")
        action = request.get("action")

        if not isinstance(request_id, str) or not request_id:
            return
        if not isinstance(entity_id, str) or not isinstance(domain, str) or not isinstance(action, str):
            await self._send_action_result(
                ws,
                request_id=request_id if isinstance(request_id, str) else "unknown",
                status="rejected",
                reason="invalid_request",
            )
            return

        permission = validate_action_permission(
            self.hass,
            self._options,
            entity_id=entity_id,
            domain=domain,
            action=action,
        )
        if not permission.allowed:
            await self._send_action_result(
                ws,
                request_id=request_id,
                status="rejected",
                reason=permission.reason or "rejected",
            )
            return

        entity_domain = entity_domain_from_id(entity_id)
        if entity_domain is None:
            await self._send_action_result(
                ws,
                request_id=request_id,
                status="rejected",
                reason="invalid_request",
            )
            return

        await self._dispatch_action(
            entity_id=entity_id,
            domain=entity_domain,
            action=action,
            request=request,
        )
        await self._send_action_result(ws, request_id=request_id, status="accepted")

    async def _dispatch_action(
        self,
        *,
        entity_id: str,
        domain: str,
        action: str,
        request: dict[str, Any],
    ) -> None:
        ha_domain, service, data = map_action_to_ha_service(
            entity_id=entity_id,
            entity_domain=domain,
            action=action,
            request=request,
        )
        await self._service_caller(ha_domain, service, data)

    async def _send_action_result(
        self,
        ws: WebSocketLike,
        *,
        request_id: str,
        status: str,
        reason: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "request_id": request_id,
            "status": status,
        }
        if reason is not None:
            payload["reason"] = reason
        await ws.send_str(_envelope(MSG_ACTION_RESULT, payload=payload))

    async def _heartbeat_loop(self, ws: WebSocketLike) -> None:
        while not self._stop_event.is_set() and not ws.closed:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=HEARTBEAT_INTERVAL,
                )
                break
            except TimeoutError:
                if ws.closed:
                    break
                await ws.send_str(_envelope("ping"))


async def _default_service_caller(
    hass: HomeAssistant,
    domain: str,
    service: str,
    data: dict[str, Any],
) -> None:
    await hass.services.async_call(domain, service, data)


async def _close_ws_session(ws: WebSocketLike) -> None:
    """Close a per-connect ClientSession when the socket ends."""

    session = getattr(ws, "_sayso_session", None)
    if session is not None:
        await session.close()


async def _default_ws_connect(
    hass: HomeAssistant,
    url: str,
    token: str,
) -> WebSocketLike:
    session = async_get_clientsession(hass)
    return await session.ws_connect(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )

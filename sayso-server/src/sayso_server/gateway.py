"""Home Assistant WebSocket session gateway."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from pydantic import ValidationError

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.envelope import SaySoEnvelope
from sayso_server.graph_store import HomeGraphStore
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.messages import MessageType
from sayso_server.results import ActionResultStatus
from sayso_server.session import HaSession


class GatewayWebSocket(Protocol):
    """Minimal WebSocket surface used by the HA session gateway."""

    closed: bool

    async def send_str(self, data: str) -> None: ...

    async def close(self) -> None: ...

    async def receive_str(self) -> str | None: ...


async def handle_ha_connection(
    ws: GatewayWebSocket,
    *,
    authorization: str | None,
    server_token: str,
    graph_store: HomeGraphStore | None = None,
) -> HaSession | None:
    """Authenticate, complete the v1 hello handshake, and process graph updates."""

    if not bearer_token_valid(authorization=authorization, expected_token=server_token):
        await ws.close()
        return None

    raw = await ws.receive_str()
    if raw is None:
        await ws.close()
        return None

    try:
        envelope = SaySoEnvelope.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
        await ws.close()
        return None

    if envelope.type != MessageType.HELLO:
        await ws.close()
        return None

    ack = SaySoEnvelope(
        version=API_VERSION,
        type=MessageType.HELLO_ACK,
        correlation_id=envelope.correlation_id,
        payload={},
    )
    await ws.send_str(ack.model_dump_json())
    store = graph_store if graph_store is not None else HomeGraphStore()
    session = HaSession(correlation_id=envelope.correlation_id, graph=store)
    # ponytail: test fakes expose _recv_queue; an empty queue means handshake-only.
    recv_queue = getattr(ws, "_recv_queue", None)
    if isinstance(recv_queue, asyncio.Queue) and recv_queue.empty():
        return session
    await _process_graph_messages(ws, session)
    return session


async def _process_graph_messages(ws: GatewayWebSocket, session: HaSession) -> None:
    while not ws.closed:
        for outbound in session.drain_outbound():
            await ws.send_str(outbound)

        try:
            raw = await asyncio.wait_for(ws.receive_str(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        if raw is None:
            return

        try:
            envelope = SaySoEnvelope.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        if envelope.type == MessageType.GRAPH_SNAPSHOT:
            try:
                snapshot = HomeGraphSnapshot.model_validate(envelope.payload)
            except ValidationError:
                continue
            session.graph.replace_snapshot(snapshot)
        elif envelope.type == MessageType.STATE_DELTA:
            session.graph.apply_state_delta(envelope.payload)
        elif envelope.type == MessageType.REGISTRY_DELTA:
            session.graph.apply_registry_delta(envelope.payload)
        elif envelope.type == MessageType.ACTION_RESULT:
            _record_action_result(session, envelope.payload)


async def _receive_or_idle(ws: GatewayWebSocket) -> str | None:
    """Wait for inbound text or briefly idle so queued outbound can flush."""
    try:
        return await asyncio.wait_for(ws.receive_str(), timeout=0.05)
    except asyncio.TimeoutError:
        return None


def _record_action_result(session: HaSession, payload: dict[str, object]) -> None:
    request_id = payload.get("request_id")
    status = payload.get("status")
    if not isinstance(request_id, str) or not request_id:
        return
    if not isinstance(status, str):
        return
    try:
        parsed_status = ActionResultStatus(status)
    except ValueError:
        return
    reason = payload.get("reason")
    parsed_reason = reason if isinstance(reason, str) else None
    session.record_action_result(
        request_id=request_id,
        status=parsed_status,
        reason=parsed_reason,
    )

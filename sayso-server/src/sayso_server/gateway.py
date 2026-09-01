"""Home Assistant WebSocket session gateway."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.envelope import SaySoEnvelope
from sayso_server.messages import MessageType
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
) -> HaSession | None:
    """Authenticate and complete the v1 hello handshake."""

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
    return HaSession(correlation_id=envelope.correlation_id)

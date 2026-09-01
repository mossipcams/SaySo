"""Live Home Assistant WebSocket action request client."""

from __future__ import annotations

from typing import Protocol

from sayso_server.api import API_VERSION
from sayso_server.envelope import SaySoEnvelope
from sayso_server.gateway import GatewayWebSocket
from sayso_server.messages import MessageType
from sayso_server.results import ActionResult
from sayso_server.session import HaSession


class HaWsActionClient:
    """Send action_request envelopes and read correlated action_result payloads."""

    def __init__(
        self,
        ws: GatewayWebSocket,
        session: HaSession,
        *,
        correlation_id: str | None = None,
    ) -> None:
        self._ws = ws
        self._session = session
        self._correlation_id = correlation_id or session.correlation_id

    def send_action_request(
        self,
        *,
        request_id: str,
        entity_id: str,
        domain: str,
        action: str,
        data: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "request_id": request_id,
            "entity_id": entity_id,
            "domain": domain,
            "action": action,
            "data": data or {},
        }
        envelope = SaySoEnvelope(
            version=API_VERSION,
            type=MessageType.ACTION_REQUEST,
            correlation_id=self._correlation_id,
            payload=payload,
        )
        self._session.queue_outbound(envelope.model_dump_json())

    def take_action_results(self, request_id: str) -> list[ActionResult]:
        return self._session.take_action_results(request_id)


class SupportsHaWsActionClient(Protocol):
    """Structural alias for tests and wiring."""

    def send_action_request(
        self,
        *,
        request_id: str,
        entity_id: str,
        domain: str,
        action: str,
        data: dict[str, object] | None = None,
    ) -> None: ...

    def take_action_results(self, request_id: str) -> list[ActionResult]: ...

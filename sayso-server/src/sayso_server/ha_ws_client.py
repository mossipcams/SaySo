"""Live Home Assistant WebSocket action request client."""

from __future__ import annotations

import asyncio
from typing import Protocol

from sayso_server.api import API_VERSION
from sayso_server.envelope import SaySoEnvelope
from sayso_server.gateway import GatewayWebSocket
from sayso_server.messages import MessageType
from sayso_server.results import ActionResult, ExecutionCategory
from sayso_server.session import HaGatewayBinding, HaSession


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

    async def collect_action_results(
        self,
        request_id: str,
        *,
        timeout: float = 30.0,
    ) -> list[ActionResult]:
        """Wait for correlated action_result payloads without blocking the event loop."""
        from sayso_server.orchestrator import classify_action_results

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            results = self._session.peek_action_results(request_id)
            if results:
                category, _ = classify_action_results(request_id, results)
                if category is not ExecutionCategory.INCOMPLETE_RESULTS:
                    return self._session.take_action_results(request_id)
            if loop.time() >= deadline:
                return self._session.take_action_results(request_id)
            await asyncio.sleep(0)


class BoundHaWsActionClient:
    """ActionRequestClient backed by the live HA gateway session when connected."""

    def __init__(self, binding: HaGatewayBinding) -> None:
        self._binding = binding

    def _client(self) -> HaWsActionClient:
        client = self._binding.action_client()
        if client is None:
            msg = "home assistant websocket session is not connected"
            raise RuntimeError(msg)
        return client

    def send_action_request(
        self,
        *,
        request_id: str,
        entity_id: str,
        domain: str,
        action: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self._client().send_action_request(
            request_id=request_id,
            entity_id=entity_id,
            domain=domain,
            action=action,
            data=data,
        )

    def take_action_results(self, request_id: str) -> list[ActionResult]:
        return self._client().take_action_results(request_id)

    async def collect_action_results(
        self,
        request_id: str,
        *,
        timeout: float = 30.0,
    ) -> list[ActionResult]:
        return await self._client().collect_action_results(
            request_id,
            timeout=timeout,
        )


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

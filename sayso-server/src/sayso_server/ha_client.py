"""Home Assistant service-call client for server-side execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sayso_server.api import API_VERSION
from sayso_server.envelope import SaySoEnvelope
from sayso_server.messages import MessageType
from sayso_server.results import ActionResult, ActionResultStatus

if TYPE_CHECKING:
    from sayso_server.gateway import GatewayWebSocket
    from sayso_server.session import HaSession


@dataclass(frozen=True)
class ServiceCall:
    domain: str
    service: str
    data: dict[str, object]
    entity_ids: frozenset[str]


@dataclass(frozen=True)
class ActionRequest:
    request_id: str
    entity_id: str
    domain: str
    action: str
    data: dict[str, object]


class HaClient(Protocol):
    """Minimal HA service-call surface used by safety-gated execution."""

    def call_service(
        self,
        *,
        domain: str,
        service: str,
        data: dict[str, object],
        entity_ids: frozenset[str],
    ) -> None: ...


class ActionRequestClient(Protocol):
    """Send action requests and read correlated action results."""

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


class FakeHaClient:
    """Records service calls and simulates action request/result exchange."""

    def __init__(self) -> None:
        self.calls: list[ServiceCall] = []
        self.action_requests: list[ActionRequest] = []
        self._queued_results: deque[tuple[str, ActionResultStatus, str | None]] = deque()
        self._delivered_results: dict[str, list[ActionResult]] = {}

    def queue_results(
        self,
        results: list[tuple[str, ActionResultStatus, str | None]],
    ) -> None:
        self._queued_results.extend(results)

    def call_service(
        self,
        *,
        domain: str,
        service: str,
        data: dict[str, object],
        entity_ids: frozenset[str],
    ) -> None:
        self.calls.append(
            ServiceCall(
                domain=domain,
                service=service,
                data=data,
                entity_ids=entity_ids,
            )
        )

    def send_action_request(
        self,
        *,
        request_id: str,
        entity_id: str,
        domain: str,
        action: str,
        data: dict[str, object] | None = None,
    ) -> None:
        payload = data or {}
        self.action_requests.append(
            ActionRequest(
                request_id=request_id,
                entity_id=entity_id,
                domain=domain,
                action=action,
                data=payload,
            )
        )
        delivered: list[ActionResult] = []
        while self._queued_results and self._queued_results[0][0] == request_id:
            queued_id, status, reason = self._queued_results.popleft()
            delivered.append(
                ActionResult(request_id=queued_id, status=status, reason=reason),
            )
        if delivered:
            self._delivered_results.setdefault(request_id, []).extend(delivered)

    def take_action_results(self, request_id: str) -> list[ActionResult]:
        return list(self._delivered_results.pop(request_id, []))


class HaSessionActionClient:
    """Send action_request envelopes and await correlated action_result futures."""

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
        return await self._session.collect_action_results(
            request_id,
            timeout=timeout,
        )

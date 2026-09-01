"""Home Assistant service-call client for server-side execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from sayso_server.results import ActionResult, ActionResultStatus


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

"""Home Assistant service-call client for server-side execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ServiceCall:
    domain: str
    service: str
    data: dict[str, object]
    entity_ids: frozenset[str]


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


class FakeHaClient:
    """Records service calls for safety and orchestration tests."""

    def __init__(self) -> None:
        self.calls: list[ServiceCall] = []

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

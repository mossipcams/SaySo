"""Established SaySo server sessions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sayso_server.graph_store import HomeGraphStore
from sayso_server.results import ActionResult, ActionResultStatus

if TYPE_CHECKING:
    from sayso_server.gateway import GatewayWebSocket
    from sayso_server.ha_ws_client import HaWsActionClient


@dataclass(slots=True)
class HaGatewayBinding:
    """Live HA WebSocket session handle exposed to the text execution path."""

    session: HaSession | None = field(default=None, repr=False)
    ws: GatewayWebSocket | None = field(default=None, repr=False)

    def attach(self, session: HaSession, ws: GatewayWebSocket) -> None:
        self.session = session
        self.ws = ws

    def detach(self) -> None:
        self.session = None
        self.ws = None

    @property
    def is_attached(self) -> bool:
        return self.session is not None and self.ws is not None

    def action_client(self) -> HaWsActionClient | None:
        if self.session is None or self.ws is None:
            return None
        from sayso_server.ha_ws_client import HaWsActionClient

        return HaWsActionClient(self.ws, self.session)


@dataclass(slots=True)
class HaSession:
    """An authenticated Home Assistant integration WebSocket session."""

    correlation_id: str
    graph: HomeGraphStore
    graph_ready: bool = False
    _outbound: deque[str] = field(default_factory=deque, repr=False)
    _action_results: dict[str, list[ActionResult]] = field(default_factory=dict, repr=False)

    def mark_graph_ready(self) -> None:
        self.graph_ready = True

    def queue_outbound(self, payload: str) -> None:
        self._outbound.append(payload)

    def drain_outbound(self) -> list[str]:
        if not self._outbound:
            return []
        pending = list(self._outbound)
        self._outbound.clear()
        return pending

    def record_action_result(
        self,
        *,
        request_id: str,
        status: ActionResultStatus,
        reason: str | None = None,
    ) -> None:
        self._action_results.setdefault(request_id, []).append(
            ActionResult(request_id=request_id, status=status, reason=reason),
        )

    def peek_action_results(self, request_id: str) -> list[ActionResult]:
        return list(self._action_results.get(request_id, []))

    def take_action_results(self, request_id: str) -> list[ActionResult]:
        return list(self._action_results.pop(request_id, []))

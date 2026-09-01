"""Established SaySo server sessions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sayso_server.graph_store import HomeGraphStore
from sayso_server.results import ActionResult, ActionResultStatus


@dataclass(slots=True)
class HaSession:
    """An authenticated Home Assistant integration WebSocket session."""

    correlation_id: str
    graph: HomeGraphStore
    _outbound: deque[str] = field(default_factory=deque, repr=False)
    _action_results: dict[str, list[ActionResult]] = field(default_factory=dict, repr=False)

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

    def take_action_results(self, request_id: str) -> list[ActionResult]:
        return list(self._action_results.pop(request_id, []))

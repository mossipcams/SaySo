"""Established SaySo server sessions."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sayso_server.graph_store import HomeGraphStore
from sayso_server.results import ActionResult, ActionResultStatus, ExecutionCategory

if TYPE_CHECKING:
    from sayso_server.gateway import GatewayWebSocket
    from sayso_server.ha_client import HaSessionActionClient


@dataclass(slots=True)
class HaGatewayBinding:
    """Live HA WebSocket session handle exposed to the text execution path."""

    session: HaSession | None = field(default=None, repr=False)
    ws: GatewayWebSocket | None = field(default=None, repr=False)

    def attach(self, session: HaSession, ws: GatewayWebSocket) -> None:
        self.session = session
        self.ws = ws

    def detach(self) -> None:
        if self.session is not None:
            self.session.clear_pending_action_waits()
        self.session = None
        self.ws = None

    @property
    def is_attached(self) -> bool:
        return self.session is not None and self.ws is not None

    def action_client(self) -> HaSessionActionClient | None:
        if self.session is None or self.ws is None:
            return None
        from sayso_server.ha_client import HaSessionActionClient

        return HaSessionActionClient(self.ws, self.session)


@dataclass(slots=True)
class HaSession:
    """An authenticated Home Assistant integration WebSocket session."""

    correlation_id: str
    graph: HomeGraphStore
    graph_ready: bool = False
    _outbound: deque[str] = field(default_factory=deque, repr=False)
    _outbound_ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _action_results: dict[str, list[ActionResult]] = field(default_factory=dict, repr=False)
    _action_futures: dict[str, asyncio.Future[None]] = field(default_factory=dict, repr=False)

    def mark_graph_ready(self) -> None:
        self.graph_ready = True

    def queue_outbound(self, payload: str) -> None:
        self._outbound.append(payload)
        self._outbound_ready.set()

    def drain_outbound(self) -> list[str]:
        if not self._outbound:
            self._outbound_ready.clear()
            return []
        pending = list(self._outbound)
        self._outbound.clear()
        self._outbound_ready.clear()
        return pending

    async def wait_for_outbound(self) -> None:
        if self._outbound:
            return
        await self._outbound_ready.wait()

    def clear_pending_action_waits(self, *, request_id: str | None = None) -> None:
        """Cancel and remove pending action-result futures."""

        if request_id is not None:
            self._cancel_action_future(request_id)
            return
        for pending_id in list(self._action_futures):
            self._cancel_action_future(pending_id)

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
        self._maybe_resolve_action_future(request_id)

    def peek_action_results(self, request_id: str) -> list[ActionResult]:
        return list(self._action_results.get(request_id, []))

    def take_action_results(self, request_id: str) -> list[ActionResult]:
        return list(self._action_results.pop(request_id, []))

    async def collect_action_results(
        self,
        request_id: str,
        *,
        timeout: float = 30.0,
    ) -> list[ActionResult]:
        """Wait for terminal correlated action_result payloads for one request."""

        if self._has_terminal_action_results(request_id):
            return self.take_action_results(request_id)

        future = self._ensure_action_future(request_id)
        try:
            await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._cancel_action_future(request_id)
        except asyncio.CancelledError:
            self._cancel_action_future(request_id)
            raise
        finally:
            self._release_action_future(request_id)
        return self.take_action_results(request_id)

    def _cancel_action_future(self, request_id: str) -> None:
        future = self._action_futures.pop(request_id, None)
        if future is not None and not future.done():
            future.cancel()

    def _release_action_future(self, request_id: str) -> None:
        self._action_futures.pop(request_id, None)

    def _ensure_action_future(self, request_id: str) -> asyncio.Future[None]:
        future = self._action_futures.get(request_id)
        if future is None or future.done():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._action_futures[request_id] = future
        return future

    def _has_terminal_action_results(self, request_id: str) -> bool:
        results = self._action_results.get(request_id)
        if not results:
            return False
        category, _ = _classify_action_results(request_id, results)
        return category is not ExecutionCategory.INCOMPLETE_RESULTS

    def _maybe_resolve_action_future(self, request_id: str) -> None:
        if not self._has_terminal_action_results(request_id):
            return
        future = self._action_futures.get(request_id)
        if future is not None and not future.done():
            future.set_result(None)


def _classify_action_results(
    request_id: str,
    results: list[ActionResult],
) -> tuple[ExecutionCategory, str | None]:
    from sayso_server.orchestrator import classify_action_results

    return classify_action_results(request_id, results)

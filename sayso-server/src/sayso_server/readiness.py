"""Readiness probes separate from process liveness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sayso_server.const import READINESS_PATH
from sayso_server.health import health_status

if TYPE_CHECKING:
    from sayso_server.home_graph import HomeGraphSnapshot
    from sayso_server.session import HaGatewayBinding


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    """Point-in-time dependency status for readiness evaluation."""

    model_ready: bool
    ha_connected: bool

    @property
    def ready(self) -> bool:
        return self.model_ready and self.ha_connected


class ReadinessState:
    """Mutable readiness tracker updated during startup and reconnects."""

    def __init__(self) -> None:
        self._model_ready = False
        self._ha_connected = False

    def set_model_ready(self, ready: bool) -> None:
        self._model_ready = ready

    def set_ha_connected(self, connected: bool) -> None:
        self._ha_connected = connected

    def snapshot(self) -> ReadinessSnapshot:
        return ReadinessSnapshot(
            model_ready=self._model_ready,
            ha_connected=self._ha_connected,
        )


def readiness_http_status(snapshot: ReadinessSnapshot) -> int:
    """Return HTTP status for GET /api/v1/ready."""

    return 200 if snapshot.ready else 503


def readiness_body(snapshot: ReadinessSnapshot) -> dict[str, bool]:
    """JSON body exposing aggregate and per-dependency readiness."""

    return {
        "ready": snapshot.ready,
        "model_ready": snapshot.model_ready,
        "ha_connected": snapshot.ha_connected,
    }


def liveness_body(snapshot: ReadinessSnapshot) -> dict[str, object]:
    """JSON body for GET /api/v1/health."""

    return {
        "status": "ok",
        "liveness": "ok",
        "model_ready": snapshot.model_ready,
        "ha_connected": snapshot.ha_connected,
    }


def liveness_response(
    *,
    authorization: str | None,
    token: str,
    snapshot: ReadinessSnapshot,
) -> tuple[int, dict[str, object] | None]:
    """Return HTTP status and JSON body for the liveness probe."""

    status = health_status(authorization=authorization, token=token)
    if status != 200:
        return status, None
    return status, liveness_body(snapshot)


def readiness_response(
    *,
    authorization: str | None,
    token: str,
    snapshot: ReadinessSnapshot,
) -> tuple[int, dict[str, bool] | None]:
    """Return HTTP status and JSON body for the readiness probe."""

    status = health_status(authorization=authorization, token=token)
    if status != 200:
        return status, None
    readiness_status = readiness_http_status(snapshot)
    return readiness_status, readiness_body(snapshot)


def text_execution_refusal(
    *,
    graph_snapshot: HomeGraphSnapshot | None,
    ha_gateway_binding: HaGatewayBinding | None = None,
) -> tuple[str, str] | None:
    """Return ``(code, message)`` when live text execution must be refused."""

    if graph_snapshot is None:
        return ("no_graph", "home graph snapshot is required")
    if ha_gateway_binding is not None and not ha_gateway_binding.is_attached:
        return ("ha_disconnected", "home assistant websocket is not connected")
    return None

"""Readiness and restart-state matrix tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from sayso_server.const import READINESS_PATH
from sayso_server.graph_store import HomeGraphStore
from sayso_server.readiness import (
    ReadinessSnapshot,
    ReadinessState,
    liveness_response,
    prepare_response_payload,
    readiness_http_status,
    readiness_response,
)
from sayso_server.session import HaSession


@pytest.mark.parametrize(
    ("model_ready", "ha_connected", "expected_ready", "expected_http"),
    [
        (False, False, False, 503),
        (True, False, False, 503),
        (False, True, False, 503),
        (True, True, True, 200),
    ],
    ids=[
        "cold_start",
        "model_only",
        "ha_only",
        "fully_ready",
    ],
)
def test_restart_state_matrix(
    model_ready: bool,
    ha_connected: bool,
    expected_ready: bool,
    expected_http: int,
) -> None:
    snapshot = ReadinessSnapshot(model_ready=model_ready, ha_connected=ha_connected)
    assert snapshot.ready is expected_ready
    assert readiness_http_status(snapshot) == expected_http


def test_readiness_path_constant() -> None:
    assert READINESS_PATH == "/api/v1/ready"


def test_readiness_response_separates_model_and_ha() -> None:
    snapshot = ReadinessSnapshot(model_ready=True, ha_connected=False)
    status, body = readiness_response(
        authorization="Bearer secret",
        token="secret",
        snapshot=snapshot,
    )
    assert status == 503
    assert body == {
        "ready": False,
        "model_ready": True,
        "ha_connected": False,
    }


def test_liveness_response_ok_while_dependencies_missing() -> None:
    snapshot = ReadinessSnapshot(model_ready=False, ha_connected=False)
    status, body = liveness_response(
        authorization="Bearer secret",
        token="secret",
        snapshot=snapshot,
    )
    assert status == 200
    assert body["status"] == "ok"
    assert body["liveness"] == "ok"
    assert body["model_ready"] is False
    assert body["ha_connected"] is False
    assert "ready" not in body


def test_prepare_response_payload_reports_session_and_readiness() -> None:
    readiness = ReadinessState()
    readiness.set_model_ready(True)
    session = HaSession(correlation_id="prep-corr", graph=HomeGraphStore())

    assert prepare_response_payload(session=session, readiness=readiness) == {
        "connected": False,
        "graph_ready": False,
        "model_ready": True,
    }

    session.mark_graph_ready()
    readiness.set_ha_connected(True)
    assert prepare_response_payload(session=session, readiness=readiness) == {
        "connected": True,
        "graph_ready": True,
        "model_ready": True,
    }


def test_readiness_state_tracks_restart_transitions() -> None:
    state = ReadinessState()
    assert state.snapshot().ready is False

    state.set_model_ready(True)
    assert state.snapshot().ready is False

    state.set_ha_connected(True)
    assert state.snapshot().ready is True

    state.set_ha_connected(False)
    assert state.snapshot().ready is False

    state.set_model_ready(False)
    assert state.snapshot().model_ready is False


def test_create_aiohttp_app_registers_readiness_route() -> None:
    from sayso_server.app import create_aiohttp_app
    from sayso_server.health import HEALTH_PATH

    app = create_aiohttp_app("secret-token", readiness=ReadinessState())
    paths = {route.resource.canonical for route in app.router.routes()}
    assert READINESS_PATH in paths
    assert HEALTH_PATH in paths


@pytest.mark.asyncio
async def test_aiohttp_ready_returns_503_until_model_and_ha_ready() -> None:
    from sayso_server.app import create_aiohttp_app

    readiness = ReadinessState()
    app = create_aiohttp_app("secret-token", readiness=readiness)
    handler = _route_handler(app, READINESS_PATH)

    status, body = await _get_json(handler, app=app, token="secret-token")
    assert status == 503
    assert body == {
        "ready": False,
        "model_ready": False,
        "ha_connected": False,
    }

    readiness.set_model_ready(True)
    status, body = await _get_json(handler, app=app, token="secret-token")
    assert status == 503
    assert body["ready"] is False

    readiness.set_ha_connected(True)
    status, body = await _get_json(handler, app=app, token="secret-token")
    assert status == 200
    assert body == {
        "ready": True,
        "model_ready": True,
        "ha_connected": True,
    }


@pytest.mark.asyncio
async def test_aiohttp_health_stays_alive_while_not_ready() -> None:
    from sayso_server.app import create_aiohttp_app
    from sayso_server.health import HEALTH_PATH

    readiness = ReadinessState()
    app = create_aiohttp_app("secret-token", readiness=readiness)
    handler = _route_handler(app, HEALTH_PATH)

    status, body = await _get_json(handler, app=app, token="secret-token")
    assert status == 200
    assert body["status"] == "ok"
    assert body["liveness"] == "ok"
    assert body["model_ready"] is False
    assert body["ha_connected"] is False


def _route_handler(app, path: str):
    for route in app.router.routes():
        if route.resource.canonical == path:
            return route.handler
    msg = f"missing route {path}"
    raise AssertionError(msg)


async def _get_json(handler, *, app, token: str | None) -> tuple[int, dict[str, object]]:
    request = MagicMock()
    header_values: dict[str, str] = {}
    if token is not None:
        header_values["Authorization"] = f"Bearer {token}"
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: header_values.get(key, default)
    request.app = app
    response = await handler(request)
    body = json.loads(response.text) if response.text else {}
    return response.status, body

"""Tests for server app construction and token loading."""

from __future__ import annotations

import pytest

from sayso_server.app import MissingServerTokenError, create_aiohttp_app, load_server_token
from sayso_server.const import (
    AUDIO_PATH,
    DEFAULT_SATELLITE_AREA_ID,
    DEFAULT_SATELLITE_ID,
    READINESS_PATH,
    TEXT_PATH,
    TOKEN_ENV_VAR,
    WS_PATH,
)
from sayso_server.health import HEALTH_PATH
from sayso_server.text_api import OrchestratorTextController


def test_load_server_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "  secret-token  ")
    assert load_server_token() == "secret-token"


def test_load_server_token_missing_raises() -> None:
    with pytest.raises(MissingServerTokenError, match=TOKEN_ENV_VAR):
        load_server_token(environ={})


def test_load_server_token_blank_raises() -> None:
    with pytest.raises(MissingServerTokenError, match=TOKEN_ENV_VAR):
        load_server_token(environ={TOKEN_ENV_VAR: "   "})


def test_create_aiohttp_app_uses_default_live_wiring() -> None:
    app = create_aiohttp_app("secret-token")
    paths = {route.resource.canonical for route in app.router.routes()}

    assert paths == {HEALTH_PATH, READINESS_PATH, TEXT_PATH, AUDIO_PATH, WS_PATH}
    assert isinstance(app["text_controller"], OrchestratorTextController)
    assert app["satellite_registry"] is not None
    assert app["graph_store"] is not None
    assert app["readiness"] is not None
    assert app["ha_gateway_binding"] is not None
    registration = app["satellite_registry"].get(DEFAULT_SATELLITE_ID)
    assert registration is not None
    assert registration.area_id == DEFAULT_SATELLITE_AREA_ID

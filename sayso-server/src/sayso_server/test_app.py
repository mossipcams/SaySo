"""Tests for server app construction and token loading."""

from __future__ import annotations

from unittest.mock import patch

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
from sayso_server.mlx_runtime import MODEL_ID_ENV_VAR, build_mlx_runtime_for_server, ensure_mlx_lm_available
from sayso_server.runtime import FakeModelRuntime
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
    controller = app["text_controller"]
    assert isinstance(controller, OrchestratorTextController)
    assert isinstance(controller._runtime, FakeModelRuntime)


def test_create_aiohttp_app_accepts_injected_model_runtime() -> None:
    runtime = FakeModelRuntime(model_id="injected")
    runtime.load()
    app = create_aiohttp_app("secret-token", model_runtime=runtime)
    controller = app["text_controller"]
    assert isinstance(controller, OrchestratorTextController)
    assert controller._runtime is runtime


def test_ensure_mlx_lm_available_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mlx_lm" or name.startswith("mlx_lm."):
            msg = "No module named 'mlx_lm'"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="mlx-lm"):
        ensure_mlx_lm_available()


def test_build_mlx_runtime_for_server_uses_env_model_id() -> None:
    from sayso_server.mlx_runtime import MlxLoadedModel

    load_calls: list[str] = []

    def fake_loader(model_id: str) -> MlxLoadedModel:
        load_calls.append(model_id)
        return MlxLoadedModel(model=object(), tokenizer=object())

    with patch("sayso_server.mlx_runtime.ensure_mlx_lm_available"):
        runtime = build_mlx_runtime_for_server(
            environ={MODEL_ID_ENV_VAR: "custom/model"},
            loader=fake_loader,
        )

    assert runtime._model_id == "custom/model"
    assert load_calls == ["custom/model"]


def test_main_wires_mlx_runtime_for_live_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")

    runtime = FakeModelRuntime(model_id="mlx-wired")
    runtime.load()

    from sayso_server.__main__ import main

    with patch("sayso_server.__main__.build_mlx_runtime_for_server", return_value=runtime):
        with patch("sayso_server.__main__.web.run_app") as run_app:
            main()

    controller = run_app.call_args.args[0]["text_controller"]
    assert isinstance(controller, OrchestratorTextController)
    assert controller._runtime is runtime


@pytest.mark.asyncio
async def test_gateway_ws_proxy_skips_protocol_ping_and_pong() -> None:
    from aiohttp import web
    from unittest.mock import AsyncMock, MagicMock

    from sayso_server.app import _GatewayWebSocketProxy

    ws = MagicMock()
    ws.closed = False
    ping = MagicMock(type=web.WSMsgType.PING, data=b"")
    pong = MagicMock(type=web.WSMsgType.PONG, data=b"")
    text = MagicMock(type=web.WSMsgType.TEXT, data='{"type":"hello"}')
    ws.receive = AsyncMock(side_effect=[ping, pong, text])

    proxy = _GatewayWebSocketProxy(ws)
    result = await proxy.receive_str()

    assert result == '{"type":"hello"}'
    assert ws.receive.await_count == 3


@pytest.mark.asyncio
async def test_gateway_ws_proxy_ends_only_on_close() -> None:
    from aiohttp import web
    from unittest.mock import AsyncMock, MagicMock

    from sayso_server.app import _GatewayWebSocketProxy

    ws = MagicMock()
    ws.closed = False
    ping = MagicMock(type=web.WSMsgType.PING, data=b"")
    close = MagicMock(type=web.WSMsgType.CLOSE, data=None)
    ws.receive = AsyncMock(side_effect=[ping, close])

    proxy = _GatewayWebSocketProxy(ws)
    result = await proxy.receive_str()

    assert result is None
    assert ws.receive.await_count == 2


@pytest.mark.asyncio
async def test_aiohttp_ws_handler_prepares_large_max_msg_size() -> None:
    from aiohttp import web
    from unittest.mock import AsyncMock, MagicMock, patch

    from sayso_server.app import create_aiohttp_app
    from sayso_server.const import HA_WS_MAX_MSG_SIZE

    app = create_aiohttp_app("secret-token")
    handler = None
    for route in app.router.routes():
        if route.resource.canonical == WS_PATH:
            handler = route.handler
            break
    assert handler is not None

    request = MagicMock()
    request.headers = {"Authorization": "Bearer secret-token"}

    ws = MagicMock()
    ws.prepare = AsyncMock()
    ws.closed = False
    ws.receive = AsyncMock(
        return_value=MagicMock(type=web.WSMsgType.CLOSE, data=None),
    )

    with patch("sayso_server.app.web.WebSocketResponse", return_value=ws) as mock_ws_class:
        with patch("sayso_server.app.handle_ha_connection", new_callable=AsyncMock):
            await handler(request)

    mock_ws_class.assert_called_once_with(max_msg_size=HA_WS_MAX_MSG_SIZE)

"""Tests for the sayso_server process entrypoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sayso_server.const import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SATELLITE_AREA_ID,
    DEFAULT_SATELLITE_ID,
    TOKEN_ENV_VAR,
)
from sayso_server.runtime import FakeModelRuntime
from sayso_server.text_api import OrchestratorTextController


def test_main_exits_when_token_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    from sayso_server.__main__ import main

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert TOKEN_ENV_VAR in captured.err


def test_main_exits_when_mlx_lm_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")

    from sayso_server.__main__ import main

    with patch(
        "sayso_server.__main__.build_mlx_runtime_for_server",
        side_effect=RuntimeError("mlx-lm is required to run sayso_server but is not installed"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "mlx-lm" in captured.err


def test_main_starts_default_live_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    monkeypatch.delenv("SAYSO_HOST", raising=False)
    monkeypatch.delenv("SAYSO_PORT", raising=False)

    runtime = FakeModelRuntime(model_id="mlx-wired")
    runtime.load()

    from sayso_server.__main__ import main

    with patch("sayso_server.__main__.build_mlx_runtime_for_server", return_value=runtime):
        with patch("sayso_server.__main__.web.run_app") as run_app:
            main()

    run_app.assert_called_once()
    app = run_app.call_args.args[0]
    controller = app["text_controller"]
    assert isinstance(controller, OrchestratorTextController)
    assert controller._runtime is runtime
    assert run_app.call_args.kwargs["host"] == DEFAULT_HOST
    assert run_app.call_args.kwargs["port"] == DEFAULT_PORT
    registration = app["satellite_registry"].get(DEFAULT_SATELLITE_ID)
    assert registration is not None
    assert registration.area_id == DEFAULT_SATELLITE_AREA_ID
    snapshot = app["readiness"].snapshot()
    assert snapshot.model_ready is True
    assert snapshot.ha_connected is False

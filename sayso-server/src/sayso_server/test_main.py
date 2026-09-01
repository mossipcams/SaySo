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


def test_main_starts_default_live_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    monkeypatch.delenv("SAYSO_HOST", raising=False)
    monkeypatch.delenv("SAYSO_PORT", raising=False)

    from sayso_server.__main__ import main

    with patch("sayso_server.__main__.web.run_app") as run_app:
        main()

    run_app.assert_called_once()
    app = run_app.call_args.args[0]
    assert app["text_controller"] is not None
    assert run_app.call_args.kwargs["host"] == DEFAULT_HOST
    assert run_app.call_args.kwargs["port"] == DEFAULT_PORT
    registration = app["satellite_registry"].get(DEFAULT_SATELLITE_ID)
    assert registration is not None
    assert registration.area_id == DEFAULT_SATELLITE_AREA_ID

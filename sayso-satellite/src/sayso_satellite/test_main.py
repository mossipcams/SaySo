"""Tests for the satellite CLI entrypoint."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest

from sayso_satellite.__main__ import json_dumps, print_response_body
from sayso_satellite.response import EARCON_TOKEN, ResponseMode


def test_print_response_body_renders_text_response_earcon() -> None:
    buffer = StringIO()
    body = {
        "version": 1,
        "type": "text_response",
        "correlation_id": "c1",
        "payload": {
            "category": "completed",
            "reason": "state_changed",
            "response_mode": ResponseMode.EARCON.value,
            "response_content": EARCON_TOKEN,
        },
    }

    print_response_body(body, sink=buffer.write)

    assert buffer.getvalue() == EARCON_TOKEN


def test_print_response_body_renders_text_response_short_text() -> None:
    lines: list[str] = []
    body = {
        "version": 1,
        "type": "text_response",
        "correlation_id": "c1",
        "payload": {
            "category": "no_action",
            "reason": "which lights?",
            "response_mode": ResponseMode.TEXT.value,
            "response_content": "which lights?",
        },
    }

    print_response_body(body, sink=lines.append)

    assert lines == ["which lights?"]


def test_print_response_body_falls_back_to_json_for_other_envelopes() -> None:
    buffer = StringIO()
    body = {"version": 1, "type": "error", "correlation_id": "c1", "payload": {"code": "invalid_request"}}

    print_response_body(body, sink=buffer.write)

    assert json.loads(buffer.getvalue()) == body


def test_main_exits_non_zero_on_http_error(capsys: pytest.CaptureFixture[str]) -> None:
    from sayso_satellite.__main__ import main

    with patch("sys.argv", ["sayso_satellite", "hello"]):
        with patch(
            "sayso_satellite.__main__.send_text",
            return_value=(401, {"error": "unauthorized"}),
        ):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "unauthorized" in captured.out


def test_main_prints_usage_when_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    from sayso_satellite.__main__ import main

    with patch("sys.argv", ["sayso_satellite"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_json_dumps_is_stable() -> None:
    assert json_dumps({"b": 2, "a": 1}) == '{\n  "a": 1,\n  "b": 2\n}'

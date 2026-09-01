"""Tests for satellite response rendering."""

from __future__ import annotations

from io import StringIO

import pytest

from sayso_satellite.response import EARCON_TOKEN, ResponseMode, render_response, render_text_response_payload


def test_render_completed_action_as_earcon() -> None:
    buffer = StringIO()

    result = render_response(
        mode=ResponseMode.EARCON.value,
        content=EARCON_TOKEN,
        sink=buffer.write,
    )

    assert result is None
    assert buffer.getvalue() == EARCON_TOKEN


def test_render_query_as_short_text() -> None:
    lines: list[str] = []

    result = render_response(
        mode=ResponseMode.TEXT.value,
        content="on",
        sink=lines.append,
    )

    assert result == "on"
    assert lines == ["on"]


def test_render_text_response_payload_earcon() -> None:
    buffer = StringIO()

    result = render_text_response_payload(
        {
            "category": "completed",
            "reason": "state_changed",
            "response_mode": ResponseMode.EARCON.value,
            "response_content": EARCON_TOKEN,
        },
        sink=buffer.write,
    )

    assert result is None
    assert buffer.getvalue() == EARCON_TOKEN


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(
            {
                "category": "no_action",
                "reason": "clarification required: which lights?",
                "response_mode": ResponseMode.TEXT.value,
                "response_content": "which lights?",
            },
            "which lights?",
            id="clarification",
        ),
        pytest.param(
            {
                "category": "failed",
                "reason": "entity unavailable",
                "response_mode": ResponseMode.TEXT.value,
                "response_content": "entity unavailable",
            },
            "entity unavailable",
            id="error",
        ),
    ],
)
def test_render_text_response_payload_matrix(
    payload: dict[str, str],
    expected: str,
) -> None:
    lines: list[str] = []
    result = render_text_response_payload(payload, sink=lines.append)
    assert result == expected
    assert lines == [expected]

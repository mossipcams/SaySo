"""Tests for satellite response rendering."""

from __future__ import annotations

import pytest

from sayso_satellite.response import render_assist_response


def test_render_assist_speech() -> None:
    lines: list[str] = []

    result = render_assist_response("on", sink=lines.append)

    assert result == "on"
    assert lines == ["on"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("which lights?", "which lights?", id="clarification"),
        pytest.param("entity unavailable", "entity unavailable", id="error"),
    ],
)
def test_render_assist_response_matrix(
    content: str,
    expected: str,
) -> None:
    lines: list[str] = []
    result = render_assist_response(content, sink=lines.append)
    assert result == expected
    assert lines == [expected]

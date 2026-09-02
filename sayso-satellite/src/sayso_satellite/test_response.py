"""Tests for satellite response rendering and playback routing."""

from __future__ import annotations

from typing import Any

import pytest

from sayso_satellite.response import (
    EARCON_SPEECH_PLACEHOLDER,
    EARCON_TOKEN,
    ResponsePlaybackMode,
    extract_assist_response_speech,
    extract_assist_speech,
    render_assist_response,
    resolve_playback_mode,
)


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
        pytest.param("on", "on", id="query_answer"),
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


def test_render_assist_response_suppresses_earcon_placeholder() -> None:
    lines: list[str] = []

    assert render_assist_response(EARCON_TOKEN, sink=lines.append) is None
    assert render_assist_response(EARCON_SPEECH_PLACEHOLDER, sink=lines.append) is None
    assert lines == []


@pytest.mark.parametrize(
    ("speech", "expected"),
    [
        pytest.param(EARCON_TOKEN, ResponsePlaybackMode.EARCON, id="bel_token"),
        pytest.param(EARCON_SPEECH_PLACEHOLDER, ResponsePlaybackMode.EARCON, id="done_placeholder"),
        pytest.param("Done", ResponsePlaybackMode.EARCON, id="done_without_period"),
        pytest.param("which lights?", ResponsePlaybackMode.SPEECH, id="clarification"),
        pytest.param("entity unavailable", ResponsePlaybackMode.SPEECH, id="error"),
        pytest.param("on", ResponsePlaybackMode.SPEECH, id="query_answer"),
    ],
)
def test_resolve_playback_mode_matrix(
    speech: str,
    expected: ResponsePlaybackMode,
) -> None:
    assert resolve_playback_mode(speech) == expected


def test_extract_assist_speech_from_plain_response() -> None:
    result: dict[str, Any] = {"intent": {"response": "on"}}

    assert extract_assist_response_speech(result) == "on"
    assert extract_assist_speech(result) == "on"


def test_extract_assist_speech_from_ha_speech_dict() -> None:
    result: dict[str, Any] = {
        "intent": {
            "response": {
                "speech": {"plain": {"speech": "which lights?"}},
            },
        },
    }

    assert extract_assist_response_speech(result) == "which lights?"
    assert extract_assist_speech(result) == "which lights?"


def test_extract_assist_speech_falls_back_to_transcript() -> None:
    result: dict[str, Any] = {"text": "turn on the lamp", "intent": {}}

    assert extract_assist_response_speech(result) is None
    assert extract_assist_speech(result) == "turn on the lamp"

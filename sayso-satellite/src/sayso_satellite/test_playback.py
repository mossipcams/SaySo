"""Tests for HA TTS fetch and Mac audio playback."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from sayso_satellite.assist import TtsOutput
from sayso_satellite.playback import (
    PlaybackError,
    afplay_audio_player,
    fetch_tts_audio,
    generate_earcon_wav,
    ha_base_url_from_websocket,
    play_earcon,
    play_tts_response,
)


def _tts_output() -> TtsOutput:
    return {
        "media_id": "media-source://tts/-stream-/abc.mp3",
        "token": "abc.mp3",
        "url": "/api/tts_proxy/abc.mp3",
        "mime_type": "audio/mpeg",
    }


class RecordingPlayer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[bytes, str]] = []

    def play(self, audio: bytes, *, mime_type: str) -> None:
        self.calls.append((audio, mime_type))
        if self.fail:
            raise PlaybackError("playback failed")


def test_ha_base_url_from_websocket_maps_ws_and_wss() -> None:
    assert (
        ha_base_url_from_websocket("ws://127.0.0.1:8123/api/websocket")
        == "http://127.0.0.1:8123"
    )
    assert (
        ha_base_url_from_websocket("wss://ha.example/api/websocket")
        == "https://ha.example"
    )


def test_fetch_tts_audio_uses_ha_bearer_auth() -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: object) -> object:
        captured["request"] = request
        return _FakeResponse(b"audio-bytes")

    audio = fetch_tts_audio(
        _tts_output(),
        token="secret-token",
        base_url="http://127.0.0.1:8123",
        urlopen=fake_urlopen,
    )

    assert audio == b"audio-bytes"
    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:8123/api/tts_proxy/abc.mp3"
    assert request.headers["Authorization"] == "Bearer secret-token"


def test_play_tts_response_fetches_once_and_plays_once() -> None:
    player = RecordingPlayer()
    fetch_calls = 0

    def fake_fetch(
        tts: TtsOutput,
        *,
        token: str,
        base_url: str,
        urlopen: object | None = None,
    ) -> bytes:
        nonlocal fetch_calls
        fetch_calls += 1
        assert token == "secret-token"
        assert base_url == "http://127.0.0.1:8123"
        assert tts == _tts_output()
        return b"audio-bytes"

    play_tts_response(
        _tts_output(),
        token="secret-token",
        base_url="http://127.0.0.1:8123",
        player=player,
        fetch=fake_fetch,
    )

    assert fetch_calls == 1
    assert player.calls == [(b"audio-bytes", "audio/mpeg")]


def test_play_tts_response_reports_fetch_failure() -> None:
    player = RecordingPlayer()

    def failing_fetch(*_args: object, **_kwargs: object) -> bytes:
        raise PlaybackError("fetch failed")

    with pytest.raises(PlaybackError, match="fetch failed"):
        play_tts_response(
            _tts_output(),
            token="secret-token",
            base_url="http://127.0.0.1:8123",
            player=player,
            fetch=failing_fetch,
        )

    assert player.calls == []


def test_play_tts_response_reports_playback_failure() -> None:
    player = RecordingPlayer(fail=True)

    with pytest.raises(PlaybackError, match="playback failed"):
        play_tts_response(
            _tts_output(),
            token="secret-token",
            base_url="http://127.0.0.1:8123",
            player=player,
            fetch=lambda *_args, **_kwargs: b"audio-bytes",
        )

    assert player.calls == [(b"audio-bytes", "audio/mpeg")]


def test_afplay_audio_player_invokes_native_player() -> None:
    player = afplay_audio_player()
    with patch("sayso_satellite.playback.subprocess.run") as mock_run:
        player.play(b"audio-bytes", mime_type="audio/mpeg")

    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == "afplay"
    assert args[1].endswith(".mp3")


def test_generate_earcon_wav_returns_non_empty_wav() -> None:
    payload = generate_earcon_wav()

    assert payload.startswith(b"RIFF")
    assert b"WAVE" in payload[:16]
    assert len(payload) > 44


def test_play_earcon_invokes_player_once() -> None:
    player = RecordingPlayer()

    play_earcon(player)

    assert len(player.calls) == 1
    audio, mime_type = player.calls[0]
    assert audio.startswith(b"RIFF")
    assert mime_type == "audio/wav"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

"""Tests for the text and audio POST client request shapes."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from sayso_satellite.capture import CHANNELS, SAMPLE_RATE_HZ, pcm_duration_ms
from sayso_satellite.client import (
    DEFAULT_AUDIO_TIMEOUT_SECONDS,
    DEFAULT_SATELLITE_ID,
    DEFAULT_TEXT_TIMEOUT_SECONDS,
    PCM_ENCODING,
    SERVER_URL_ENV_VAR,
    TIMEOUT_ENV_VAR,
    TOKEN_ENV_VAR,
    send_audio,
    send_text,
)

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"
CORNER_LAMP_PCM = FIXTURES / "turn_off_the_corner_lamp.pcm"


def test_send_text_posts_envelope_with_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    monkeypatch.setenv(SERVER_URL_ENV_VAR, "http://127.0.0.1:8765")

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        payload = b'{"version":1,"type":"text_response","correlation_id":"c1","payload":{"category":"completed"}}'

        class FakeResponse(BytesIO):
            status = 200

        return FakeResponse(payload)

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        status, body = send_text("turn off the floor lamp")

    assert status == 200
    assert body is not None
    assert captured["url"] == "http://127.0.0.1:8765/api/v1/text"
    assert captured["method"] == "POST"
    headers = {key.title(): value for key, value in captured["headers"].items()}
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Content-Type"] == "application/json"
    envelope = captured["body"]
    assert envelope["version"] == 1
    assert envelope["type"] == "text"
    assert envelope["payload"]["satellite_id"] == DEFAULT_SATELLITE_ID
    assert envelope["payload"]["text"] == "turn off the floor lamp"
    assert isinstance(envelope["correlation_id"], str)
    assert envelope["correlation_id"]


def test_send_text_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=TOKEN_ENV_VAR):
        send_text("hello")


def test_send_text_surfaces_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=BytesIO(b'{"error":"unauthorized"}'),
        )

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        status, body = send_text("hello")

    assert status == 401
    assert body == {"error": "unauthorized"}


def test_send_audio_posts_envelope_with_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    monkeypatch.setenv(SERVER_URL_ENV_VAR, "http://127.0.0.1:8765")

    pcm = b"\x00\x01" * 800  # 50 ms at 16 kHz mono PCM16
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        payload = b'{"version":1,"type":"audio_response","correlation_id":"c1","payload":{"sequence":0}}'

        class FakeResponse(BytesIO):
            status = 200

        return FakeResponse(payload)

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        status, body = send_audio(pcm, sequence=0)

    assert status == 200
    assert body is not None
    assert captured["url"] == "http://127.0.0.1:8765/api/v1/audio"
    assert captured["method"] == "POST"
    headers = {key.title(): value for key, value in captured["headers"].items()}
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Content-Type"] == "application/json"
    envelope = captured["body"]
    assert envelope["version"] == 1
    assert envelope["type"] == "audio"
    assert isinstance(envelope["correlation_id"], str)
    assert envelope["correlation_id"]
    payload = envelope["payload"]
    assert payload["satellite_id"] == DEFAULT_SATELLITE_ID
    assert payload["sequence"] == 0
    assert payload["duration_ms"] == pcm_duration_ms(byte_length=len(pcm))
    assert payload["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert payload["channels"] == CHANNELS
    assert payload["encoding"] == PCM_ENCODING
    assert base64.b64decode(payload["pcm_base64"]) == pcm


def test_send_audio_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=TOKEN_ENV_VAR):
        send_audio(b"\x00\x00" * 100)


def test_send_audio_surfaces_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=BytesIO(b'{"error":"unauthorized"}'),
        )

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        status, body = send_audio(b"\x00\x00" * 100)

    assert status == 401
    assert body == {"error": "unauthorized"}


def test_default_audio_timeout_is_extended() -> None:
    assert DEFAULT_AUDIO_TIMEOUT_SECONDS == 180
    assert DEFAULT_TEXT_TIMEOUT_SECONDS == 180


def test_send_audio_uses_extended_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    captured: dict[str, float] = {}

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        captured["timeout"] = timeout
        payload = b'{"version":1,"type":"text_response","correlation_id":"c1","payload":{}}'

        class FakeResponse(BytesIO):
            status = 200

        return FakeResponse(payload)

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        send_audio(b"\x00\x00" * 100)

    assert captured["timeout"] == DEFAULT_AUDIO_TIMEOUT_SECONDS


def test_send_audio_timeout_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    monkeypatch.setenv(TIMEOUT_ENV_VAR, "240")
    captured: dict[str, float] = {}

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        captured["timeout"] = timeout
        payload = b'{"version":1,"type":"text_response","correlation_id":"c1","payload":{}}'

        class FakeResponse(BytesIO):
            status = 200

        return FakeResponse(payload)

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        send_audio(b"\x00\x00" * 100)

    assert captured["timeout"] == 240.0


def test_send_audio_posts_corner_lamp_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOKEN_ENV_VAR, "secret-token")
    pcm = CORNER_LAMP_PCM.read_bytes()
    assert len(pcm) % 2 == 0
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=30):  # noqa: ANN001, ARG001
        captured["body"] = json.loads(request.data.decode("utf-8"))
        payload = b'{"version":1,"type":"text_response","correlation_id":"c1","payload":{"category":"completed"}}'

        class FakeResponse(BytesIO):
            status = 200

        return FakeResponse(payload)

    with patch("sayso_satellite.client.urlopen", side_effect=fake_urlopen):
        status, body = send_audio(pcm)

    assert status == 200
    assert body is not None
    envelope = captured["body"]
    assert envelope["type"] == "audio"
    payload = envelope["payload"]
    assert base64.b64decode(payload["pcm_base64"]) == pcm
    assert payload["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert payload["channels"] == CHANNELS

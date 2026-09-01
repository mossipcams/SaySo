"""Tests for the text POST client request shape."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from sayso_satellite.client import (
    DEFAULT_SATELLITE_ID,
    SERVER_URL_ENV_VAR,
    TOKEN_ENV_VAR,
    send_text,
)


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

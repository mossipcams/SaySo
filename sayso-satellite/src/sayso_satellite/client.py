"""Minimal HTTP client for POST /api/v1/text and /api/v1/audio."""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sayso_satellite.capture import CHANNELS, SAMPLE_RATE_HZ, pcm_duration_ms

API_VERSION = 1
TEXT_PATH = "/api/v1/text"
AUDIO_PATH = "/api/v1/audio"
PCM_ENCODING = "pcm_s16le"
TOKEN_ENV_VAR = "SAYSO_TOKEN"
SERVER_URL_ENV_VAR = "SAYSO_SERVER_URL"
DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
DEFAULT_SATELLITE_ID = "macbook"
DEFAULT_TIMEOUT_SECONDS = 30


def _load_token(*, environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    token = source.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise RuntimeError(f"{TOKEN_ENV_VAR} environment variable is required")
    return token


def _server_url(*, environ: dict[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return source.get(SERVER_URL_ENV_VAR, DEFAULT_SERVER_URL).rstrip("/")


def build_text_request(
    text: str,
    *,
    satellite_id: str = DEFAULT_SATELLITE_ID,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build the versioned text request envelope."""

    return {
        "version": API_VERSION,
        "type": "text",
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "payload": {
            "satellite_id": satellite_id,
            "text": text,
        },
    }


def build_audio_request(
    pcm: bytes,
    *,
    satellite_id: str = DEFAULT_SATELLITE_ID,
    sequence: int = 0,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build the versioned audio request envelope."""

    return {
        "version": API_VERSION,
        "type": "audio",
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "payload": {
            "satellite_id": satellite_id,
            "sequence": sequence,
            "duration_ms": pcm_duration_ms(byte_length=len(pcm)),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
            "encoding": PCM_ENCODING,
            "pcm_base64": base64.b64encode(pcm).decode("ascii"),
        },
    }


def _post_json_envelope(
    path: str,
    body: dict[str, Any],
    *,
    environ: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any] | None]:
    token = _load_token(environ=environ)
    url = f"{_server_url(environ=environ)}{path}"
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        parsed = json.loads(raw) if raw else None
        return exc.code, parsed


def send_text(
    text: str,
    *,
    satellite_id: str = DEFAULT_SATELLITE_ID,
    correlation_id: str | None = None,
    environ: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any] | None]:
    """POST text to the SaySo server and return `(status, json_body)`."""

    body = build_text_request(
        text,
        satellite_id=satellite_id,
        correlation_id=correlation_id,
    )
    return _post_json_envelope(
        TEXT_PATH,
        body,
        environ=environ,
        timeout=timeout,
    )


def send_audio(
    pcm: bytes,
    *,
    satellite_id: str = DEFAULT_SATELLITE_ID,
    sequence: int = 0,
    correlation_id: str | None = None,
    environ: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any] | None]:
    """POST 16 kHz mono PCM16 to the SaySo server and return `(status, json_body)`."""

    body = build_audio_request(
        pcm,
        satellite_id=satellite_id,
        sequence=sequence,
        correlation_id=correlation_id,
    )
    return _post_json_envelope(
        AUDIO_PATH,
        body,
        environ=environ,
        timeout=timeout,
    )

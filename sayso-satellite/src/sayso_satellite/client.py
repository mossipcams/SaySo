"""Minimal HTTP client for POST /api/v1/text."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_VERSION = 1
TEXT_PATH = "/api/v1/text"
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


def send_text(
    text: str,
    *,
    satellite_id: str = DEFAULT_SATELLITE_ID,
    correlation_id: str | None = None,
    environ: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any] | None]:
    """POST text to the SaySo server and return `(status, json_body)`."""

    token = _load_token(environ=environ)
    url = f"{_server_url(environ=environ)}{TEXT_PATH}"
    body = build_text_request(
        text,
        satellite_id=satellite_id,
        correlation_id=correlation_id,
    )
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

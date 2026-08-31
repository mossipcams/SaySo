"""Protocol parsing for SaySo API v1 envelopes."""

import json

import pytest
from pydantic import ValidationError

from sayso_server.protocol import parse_envelope, parse_envelope_json


def test_parse_envelope_json_accepts_valid_message() -> None:
    raw = json.dumps(
        {
            "version": 1,
            "type": "pong",
            "correlation_id": "heartbeat-42",
        },
    )
    envelope = parse_envelope_json(raw)
    assert envelope.type.value == "pong"
    assert envelope.correlation_id == "heartbeat-42"


def test_parse_envelope_rejects_unknown_version() -> None:
    with pytest.raises(ValidationError):
        parse_envelope({"version": 99, "type": "hello", "correlation_id": "x"})


def test_parse_envelope_rejects_missing_correlation_id() -> None:
    with pytest.raises(ValidationError):
        parse_envelope({"version": 1, "type": "error", "payload": {"code": "bad"}})

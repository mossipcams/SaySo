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


def test_parse_envelope_json_accepts_conversation_request() -> None:
    correlation_id = "turn-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    raw = json.dumps(
        {
            "version": 1,
            "type": "conversation_request",
            "correlation_id": correlation_id,
            "payload": {"transcript": "turn off the lights"},
        },
    )
    envelope = parse_envelope_json(raw)
    assert envelope.type.value == "conversation_request"
    assert envelope.correlation_id == correlation_id


def test_parse_envelope_accepts_conversation_response() -> None:
    correlation_id = "turn-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    envelope = parse_envelope(
        {
            "version": 1,
            "type": "conversation_response",
            "correlation_id": correlation_id,
            "payload": {"speech": "Okay.", "response_type": "action_done"},
        },
    )
    assert envelope.type.value == "conversation_response"
    assert envelope.correlation_id == correlation_id


def test_parse_envelope_json_accepts_prepare() -> None:
    correlation_id = "prep-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    raw = json.dumps(
        {
            "version": 1,
            "type": "prepare",
            "correlation_id": correlation_id,
            "payload": {},
        },
    )
    envelope = parse_envelope_json(raw)
    assert envelope.type.value == "prepare"
    assert envelope.correlation_id == correlation_id


def test_parse_envelope_accepts_prepare_response() -> None:
    correlation_id = "prep-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    envelope = parse_envelope(
        {
            "version": 1,
            "type": "prepare_response",
            "correlation_id": correlation_id,
            "payload": {
                "connected": True,
                "graph_ready": True,
                "model_ready": True,
            },
        },
    )
    assert envelope.type.value == "prepare_response"
    assert envelope.correlation_id == correlation_id


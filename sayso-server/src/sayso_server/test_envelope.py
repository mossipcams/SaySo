"""SaySo API v1 envelope validation tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sayso_server.envelope import SaySoEnvelope

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def test_valid_envelope_round_trips() -> None:
    payload = {
        "version": 1,
        "type": "hello",
        "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
        "payload": {"server_id": "sayso-local"},
    }
    envelope = SaySoEnvelope.model_validate(payload)
    dumped = envelope.model_dump(mode="json")
    assert SaySoEnvelope.model_validate(dumped) == envelope
    assert json.loads(json.dumps(dumped)) == dumped


def test_missing_correlation_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SaySoEnvelope.model_validate({"version": 1, "type": "ping"})


def test_empty_correlation_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SaySoEnvelope.model_validate({"version": 1, "type": "ping", "correlation_id": ""})


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SaySoEnvelope.model_validate(
            {"version": 2, "type": "hello", "correlation_id": "req-1"},
        )


def test_graph_snapshot_envelope_round_trips() -> None:
    envelope = SaySoEnvelope.model_validate(
        {"version": 1, "type": "graph_snapshot", "correlation_id": "req-1"},
    )
    assert envelope.type.value == "graph_snapshot"


def test_action_request_envelope_round_trips() -> None:
    envelope = SaySoEnvelope.model_validate(
        {"version": 1, "type": "action_request", "correlation_id": "req-1"},
    )
    assert envelope.type.value == "action_request"


def test_action_result_envelope_round_trips() -> None:
    envelope = SaySoEnvelope.model_validate(
        {
            "version": 1,
            "type": "action_result",
            "correlation_id": "req-1",
            "payload": {"request_id": "req-1", "status": "accepted"},
        },
    )
    assert envelope.type.value == "action_result"


def test_conversation_request_envelope_preserves_correlation_id() -> None:
    correlation_id = "turn-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    envelope = SaySoEnvelope.model_validate(
        {
            "version": 1,
            "type": "conversation_request",
            "correlation_id": correlation_id,
            "payload": {"transcript": "turn off the lights"},
        },
    )
    assert envelope.type.value == "conversation_request"
    assert envelope.correlation_id == correlation_id


def test_conversation_response_envelope_preserves_correlation_id() -> None:
    correlation_id = "turn-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    envelope = SaySoEnvelope.model_validate(
        {
            "version": 1,
            "type": "conversation_response",
            "correlation_id": correlation_id,
            "payload": {"speech": "Okay.", "response_type": "action_done"},
        },
    )
    assert envelope.type.value == "conversation_response"
    assert envelope.correlation_id == correlation_id


def test_prepare_envelope_preserves_correlation_id() -> None:
    correlation_id = "prep-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    envelope = SaySoEnvelope.model_validate(
        {
            "version": 1,
            "type": "prepare",
            "correlation_id": correlation_id,
            "payload": {},
        },
    )
    assert envelope.type.value == "prepare"
    assert envelope.correlation_id == correlation_id


def test_prepare_response_envelope_preserves_correlation_id() -> None:
    correlation_id = "prep-9f3c2a1b-4d5e-6789-abcd-ef0123456789"
    envelope = SaySoEnvelope.model_validate(
        {
            "version": 1,
            "type": "prepare_response",
            "correlation_id": correlation_id,
            "payload": {
                "connected": False,
                "graph_ready": False,
                "model_ready": True,
            },
        },
    )
    assert envelope.type.value == "prepare_response"
    assert envelope.correlation_id == correlation_id


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SaySoEnvelope.model_validate(
            {"version": 1, "type": "not_a_sayso_type", "correlation_id": "req-1"},
        )


def test_envelope_valid_fixture_round_trips() -> None:
    data = json.loads((FIXTURES / "envelope.valid.json").read_text())
    envelope = SaySoEnvelope.model_validate(data)
    assert envelope.type.value == "hello_ack"


def test_envelope_invalid_fixture_cases_are_rejected() -> None:
    cases = json.loads((FIXTURES / "envelope.invalid.json").read_text())
    assert isinstance(cases, list)
    for case in cases:
        with pytest.raises(ValidationError):
            SaySoEnvelope.model_validate(case)

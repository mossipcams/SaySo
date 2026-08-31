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


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SaySoEnvelope.model_validate(
            {"version": 1, "type": "action_request", "correlation_id": "req-1"},
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

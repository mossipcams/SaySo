"""JSON Schema generation for ControlPlan and SaySo API v1."""

import json
from pathlib import Path

from sayso_server.envelope import SaySoEnvelope
from sayso_server.schema import control_plan_json_schema, sayso_api_v1_json_schema

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def test_control_plan_json_schema_has_outcome_discriminator() -> None:
    schema = control_plan_json_schema()
    assert schema["title"] == "ControlPlan"
    assert "oneOf" in schema or "anyOf" in schema or "discriminator" in schema
    schema_text = str(schema)
    for outcome in ("action", "query", "clarification", "unsupported", "no-action"):
        assert outcome in schema_text


def test_sayso_api_v1_json_schema_matches_committed_fixture() -> None:
    schema = sayso_api_v1_json_schema()
    assert schema["title"] == "SaySoEnvelope"
    committed = json.loads((FIXTURES / "sayso_api_v1.schema.json").read_text())
    assert schema == committed


def test_contract_fixtures_validate_against_generated_schema() -> None:
    schema = sayso_api_v1_json_schema()
    required = set(schema.get("required", []))
    assert {"version", "type", "correlation_id"}.issubset(required)

    for name in ("envelope.valid.json", "api_v1_envelope.json"):
        data = json.loads((FIXTURES / name).read_text())
        envelope = SaySoEnvelope.model_validate(data)
        assert envelope.correlation_id

    catalog = json.loads((FIXTURES / "sayso_api_v1.json").read_text())
    SaySoEnvelope.model_validate(catalog["envelope"])


def test_api_v1_fixture_catalog_lists_known_types() -> None:
    catalog = json.loads((FIXTURES / "api_v1.json").read_text())
    assert catalog["version"] == 1
    assert "hello" in catalog["message_types"]

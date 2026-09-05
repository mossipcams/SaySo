"""Ensure training fixtures stay locked to schemas/sayso-tool-schema-v2.json."""

from __future__ import annotations

import json
from pathlib import Path

from adapters.schema import ALLOWED_HASS_TOOLS, load_v2_schema, v2_tool_names

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def test_locked_fingerprint_is_stable_in_fixture_copy() -> None:
    locked = load_v2_schema()
    fixture = json.loads((FIXTURES / "sayso_tool_schema_v2.json").read_text(encoding="utf-8"))
    assert fixture["contract_version"] == "sayso-tool-schema/v2"
    assert fixture["schema_fingerprint"] == locked["schema_fingerprint"]


def test_allowlist_derived_from_locked_artifact_not_hardcoded_drift() -> None:
    locked_names = v2_tool_names()
    assert ALLOWED_HASS_TOOLS == frozenset(locked_names)

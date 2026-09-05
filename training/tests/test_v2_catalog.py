"""Regression tests for device-type-tiered v2 schema alignment."""

from __future__ import annotations

import json
from pathlib import Path

from adapters.schema import (
    ALLOWED_HASS_TOOLS,
    assert_v2_tiers_cover_catalog,
    load_v1_schema,
    load_v2_schema,
    v2_tool_catalog_by_device_type,
    v2_tool_device_type_tiers,
    v2_tool_names,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def test_v2_fixture_matches_locked_artifact() -> None:
    fixture = json.loads((FIXTURES / "sayso_tool_schema_v2.json").read_text(encoding="utf-8"))
    locked = load_v2_schema()
    assert fixture["schema_fingerprint"] == locked["schema_fingerprint"]
    assert fixture["tools"] == locked["tools"]
    assert fixture["tool_catalog_by_device_type"] == locked["tool_catalog_by_device_type"]


def test_v2_flat_tools_match_v1_contract() -> None:
    v1 = load_v1_schema()
    v2 = load_v2_schema()
    assert v1["schema_fingerprint"] == v2["schema_fingerprint"]
    assert v1["tools"] == v2["tools"]


def test_v2_catalog_groups_match_device_type_tiers() -> None:
    catalog = v2_tool_catalog_by_device_type()
    tiers = v2_tool_device_type_tiers()
    assert set(catalog) == set(tiers)
    for device_type, names in tiers.items():
        grouped = {tool["function"]["name"] for tool in catalog[device_type]}
        assert grouped == set(names)
    assert_v2_tiers_cover_catalog()
    assert ALLOWED_HASS_TOOLS == v2_tool_names()

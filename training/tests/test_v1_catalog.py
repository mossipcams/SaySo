"""Regression tests for locked v1 tool catalog alignment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.schema import (
    ALLOWED_HASS_TOOLS,
    assert_openai_tool_envelope,
    assert_v1_tiers_cover_catalog,
    load_v1_schema,
    load_v1_tools,
    v1_tool_device_type_tiers,
    v1_tool_names,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
LOCKED_ARTIFACT = ROOT.parent / "schemas" / "sayso-tool-schema-v1.json"


def test_allowed_tools_match_locked_v1_names() -> None:
    assert ALLOWED_HASS_TOOLS == v1_tool_names()
    assert "HassClimateSetTemperature" in ALLOWED_HASS_TOOLS
    assert "HassMediaPause" in ALLOWED_HASS_TOOLS
    assert "HassStartTimer" in ALLOWED_HASS_TOOLS
    assert "HassVacuumStart" in ALLOWED_HASS_TOOLS
    assert_v1_tiers_cover_catalog()


def test_v1_device_type_tiers_partition_catalog() -> None:
    tiers = v1_tool_device_type_tiers()
    assert set(tiers) == {
        "query",
        "generic",
        "light",
        "fan",
        "climate",
        "media_player",
        "vacuum",
        "timer",
    }
    assert_v1_tiers_cover_catalog()


def test_ha_assist_tools_match_locked_v1_tools() -> None:
    fixture_tools = json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))
    locked_tools = list(load_v1_tools())
    assert fixture_tools == locked_tools


def test_training_fixture_schema_matches_locked_artifact() -> None:
    fixture = json.loads((FIXTURES / "sayso_tool_schema_v1.json").read_text(encoding="utf-8"))
    locked = load_v1_schema()
    assert fixture["schema_fingerprint"] == locked["schema_fingerprint"]
    assert fixture["tools"] == locked["tools"]


def test_every_v1_tool_uses_openai_function_envelope() -> None:
    for tool in load_v1_tools():
        assert_openai_tool_envelope(tool)


@pytest.mark.parametrize(
    "path",
    [
        FIXTURES / "sayso_lfm_smoke.jsonl",
        FIXTURES / "sayso_axolotl_smoke.jsonl",
    ],
)
def test_smoke_datasets_tool_names_are_v1_subset(path: Path) -> None:
    allowed = ALLOWED_HASS_TOOLS
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        names = {tool["function"]["name"] for tool in record["tools"]}
        assert names.issubset(allowed)
        for tool in record["tools"]:
            assert tool["type"] == "function"

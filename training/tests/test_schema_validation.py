"""JSON Schema validation for tool arguments."""

from __future__ import annotations

import json
from pathlib import Path

from adapters.schema import ALLOWED_HASS_TOOLS, validate_tool_arguments, tool_schema_map

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_enum_or_type_failure_via_adapter() -> None:
    tools = json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))
    schemas = tool_schema_map(tools)
    err = validate_tool_arguments(
        "HassLightSet",
        {"name": "Lamp", "brightness": "fifty"},
        schemas,
    )
    assert err == "schema_validation_failed"


def test_brightness_out_of_range_rejected() -> None:
    tools = json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))
    schemas = tool_schema_map(tools)
    err = validate_tool_arguments(
        "HassLightSet",
        {"name": "Lamp", "brightness": 150},
        schemas,
    )
    assert err == "schema_validation_failed"


def test_climate_tool_in_pinned_catalog_and_validates() -> None:
    assert "HassClimateSetTemperature" in ALLOWED_HASS_TOOLS
    tools = json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))
    schemas = tool_schema_map(tools)
    err = validate_tool_arguments(
        "HassClimateSetTemperature",
        {"name": "Thermostat", "temperature": 72},
        schemas,
    )
    assert err is None

"""JSON Schema validation for tool arguments."""

from __future__ import annotations

import json
from pathlib import Path

from adapters.home_llm_v2 import convert_entry
from adapters.schema import RejectionStats, validate_tool_arguments, tool_schema_map

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


def test_non_v1_climate_tool_rejected() -> None:
    entry = {
        "messages": [
            {"role": "user", "content": "Set thermostat"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "HassClimateSetTemperature",
                            "arguments": '{"name": "Thermostat"}',
                        }
                    }
                ],
            },
        ],
        "tools": [
            {
                "function": {
                    "name": "HassClimateSetTemperature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "temperature": {"type": "number"},
                        },
                        "required": ["temperature"],
                    },
                }
            }
        ],
    }
    stats = RejectionStats()
    assert convert_entry(entry, stats=stats) is None
    assert stats.counts.get("unknown_tool", 0) >= 1

"""The inlined tool list must match the bytes the model trained on.

ajax-optimization sayso-training/training/adapters/test_lfm.py asserts the same
COMPILED_TOOL -> TRAINED_TOOL_TEXT pair for the training render.
"""

from __future__ import annotations

from custom_components.sayso.area_context import tools_before_area_messages
from custom_components.sayso.tool_schema import compile_source_tools

COMPILED_TOOL = {
    "function": {
        "description": "Turns on a device",
        "name": "intent__HassTurnOn",
        "parameters": {
            "properties": {"area": {"type": "string"}, "name": {"type": "string"}},
            "type": "object",
        },
    },
    "type": "function",
}
TRAINED_TOOL_TEXT = (
    '{"type": "function", "function": {"name": "intent__HassTurnOn", '
    '"description": "Turns on a device", "parameters": {"properties": '
    '{"area": {"type": "string"}, "name": {"type": "string"}}, "type": "object"}}}'
)


def test_compiled_tools_arrive_with_sorted_keys() -> None:
    compiled = compile_source_tools(
        [
            {
                "name": "intent__HassTurnOn",
                "description": "Turns on a device",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "area": {"type": "string"},
                    },
                },
            }
        ]
    ).tools[0]
    assert compiled == COMPILED_TOOL
    assert list(compiled) == ["function", "type"]


def test_inlined_tool_uses_training_key_order() -> None:
    messages = [
        {"role": "system", "content": "Head\nArea context:\nsatellite_area: kitchen"}
    ]

    content = tools_before_area_messages(messages, [COMPILED_TOOL])[0]["content"]

    assert content == (
        f"Head\nList of tools: [{TRAINED_TOOL_TEXT}]\nArea context:\nsatellite_area: kitchen"
    )

from __future__ import annotations

import voluptuous as vol
from homeassistant.helpers import intent, llm

from .boundary import validate_arguments
from .client import ToolCall
from .schema import validate_tool_arguments
from .tracing import TraceContext


class _SetVolumeRelativeHandler(intent.IntentHandler):
    name = "HassSetVolumeRelative"
    platform = "media_player"
    intent_type = "HassSetVolumeRelative"
    slot_schema = {
        vol.Required("name"): str,
        vol.Required("volume_step"): vol.All(
            vol.Coerce(int),
            vol.Range(min=-100, max=100),
            lambda val: val / 100,
        ),
    }


class _SetVolumeHandler(intent.IntentHandler):
    name = "HassSetVolume"
    platform = "media_player"
    intent_type = "HassSetVolume"
    slot_schema = {
        vol.Required("name"): str,
        vol.Required("volume_level"): vol.All(
            vol.Coerce(int),
            vol.Range(min=0, max=100),
            lambda val: val / 100,
        ),
    }


def _executed_tool_args(tool: llm.IntentTool, arguments: dict) -> dict:
    call = ToolCall(id="call-1", name=tool.name, arguments=arguments)
    tool_map = {tool.name: tool}
    validated, failures = validate_arguments([call], tool_map, TraceContext("test"))
    assert not failures
    tool_inputs = [
        llm.ToolInput(id=c.id, tool_name=tool_map[c.name].name, tool_args=args)
        for c, args in validated
    ]
    return dict(tool_inputs[0].tool_args)


def test_volume_step_reaches_ha_unconverted():
    tool = llm.IntentTool(
        "media_player__HassSetVolumeRelative", _SetVolumeRelativeHandler()
    )
    executed = _executed_tool_args(tool, {"name": "TV", "volume_step": 10})
    assert executed == {"name": "TV", "volume_step": 10}
    converted_once = tool.parameters({"name": "TV", "volume_step": 10})
    assert converted_once["volume_step"] == 0.1


def test_volume_level_reaches_ha_unconverted():
    tool = llm.IntentTool("media_player__HassSetVolume", _SetVolumeHandler())
    executed = _executed_tool_args(tool, {"name": "TV", "volume_level": 40})
    assert executed == {"name": "TV", "volume_level": 40}
    converted_once = tool.parameters({"name": "TV", "volume_level": 40})
    assert converted_once["volume_level"] == 0.4


def test_out_of_range_volume_step_still_fails_closed():
    tool = llm.IntentTool(
        "media_player__HassSetVolumeRelative", _SetVolumeRelativeHandler()
    )
    validated, failures = validate_arguments(
        [
            ToolCall(
                id="call-1",
                name=tool.name,
                arguments={"name": "TV", "volume_step": 150},
            )
        ],
        {tool.name: tool},
        TraceContext("test"),
    )
    assert not validated
    assert len(failures) == 1


def test_unexpected_argument_still_fails_closed():
    tool = llm.IntentTool("media_player__HassSetVolume", _SetVolumeHandler())
    _, error = validate_tool_arguments(
        tool, {"name": "TV", "volume_level": 40, "area": "Kitchen"}
    )
    assert error is not None

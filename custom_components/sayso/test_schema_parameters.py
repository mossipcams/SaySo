"""Regressions for issue #52: tool parameters must survive schema compilation.

Home Assistant 2026.9 swapped voluptuous for ``probatio`` and installs it under
the ``voluptuous`` name, so ``tool.parameters`` is a ``probatio.schema.Schema``.
``voluptuous_openapi.convert`` returns its unsupported marker for those, and the
old code answered with an empty ``{"type": "object", "properties": {}}``. Every
tool then reached the model with no arguments at all, making
``HassTurnOn(name="TV", domain=["media_player"])`` unexpressible: the model
emitted no tool call and spoke prose instead.
"""

from __future__ import annotations

from typing import Any

import pytest
import voluptuous as vol
from homeassistant.helpers import config_validation as cv, llm

from custom_components.sayso import schema as schema_module
from custom_components.sayso.exceptions import SaySoInvalidToolEnvelopeError
from custom_components.sayso.schema import compile_parameters, compile_tool


class _TurnOnTool(llm.Tool):
    """A tool shaped like Home Assistant's HassTurnOn."""

    name = "HassTurnOn"
    description = "Turns on/opens a device or entity"

    def __init__(self) -> None:
        self.parameters = vol.Schema(
            {
                vol.Any("name", "area", "floor"): cv.string,
                vol.Optional("domain"): vol.All(cv.ensure_list, [cv.string]),
                vol.Optional("device_class"): vol.All(
                    cv.ensure_list, [vol.In(["tv", "speaker"])]
                ),
            }
        )

    async def async_call(
        self,
        hass: Any,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        return {"ok": True}


class _NoArgTool(llm.Tool):
    """A tool that genuinely takes no arguments, such as GetDateTime."""

    name = "GetDateTime"
    description = "Returns the current date and time"

    def __init__(self) -> None:
        self.parameters = vol.Schema({})

    async def async_call(
        self,
        hass: Any,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        return {"ok": True}


def test_targeting_parameters_reach_the_model() -> None:
    """A control tool keeps the slots needed to name a target."""
    compiled = compile_tool(_TurnOnTool(), custom_serializer=llm.selector_serializer)

    properties = compiled["function"]["parameters"]["properties"]
    assert "name" in properties, "no name slot: the model cannot target an entity"
    assert "area" in properties
    assert "domain" in properties


def test_argument_free_tool_still_compiles() -> None:
    """An genuinely empty schema is not a conversion failure."""
    compiled = compile_tool(_NoArgTool(), custom_serializer=llm.selector_serializer)

    assert compiled["function"]["parameters"]["properties"] == {}


def test_failed_conversion_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the tool rather than offering it stripped of its arguments.

    This is the deployed configuration from issue #52: probatio's converter
    unavailable, so nothing can read a probatio schema.
    """
    monkeypatch.setattr(schema_module, "_to_openapi", None)
    monkeypatch.setattr(schema_module, "convert", None)

    with pytest.raises(SaySoInvalidToolEnvelopeError):
        compile_parameters(
            _TurnOnTool().parameters,
            custom_serializer=llm.selector_serializer,
        )

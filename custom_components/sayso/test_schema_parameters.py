
from __future__ import annotations

from typing import Any

import pytest
import voluptuous as vol
from homeassistant.helpers import config_validation as cv, llm

from custom_components.sayso import schema as schema_module
from custom_components.sayso.exceptions import SaySoInvalidToolEnvelopeError
from custom_components.sayso.schema import compile_parameters, compile_tool


class _TurnOnTool(llm.Tool):

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
    compiled = compile_tool(_TurnOnTool(), custom_serializer=llm.selector_serializer)

    properties = compiled["function"]["parameters"]["properties"]
    assert "name" in properties, "no name slot: the model cannot target an entity"
    assert "area" in properties
    assert "domain" in properties


def test_argument_free_tool_still_compiles() -> None:
    compiled = compile_tool(_NoArgTool(), custom_serializer=llm.selector_serializer)

    assert compiled["function"]["parameters"]["properties"] == {}


def test_failed_conversion_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema_module, "_to_openapi", None)
    monkeypatch.setattr(schema_module, "convert", None)

    with pytest.raises(SaySoInvalidToolEnvelopeError):
        compile_parameters(
            _TurnOnTool().parameters,
            custom_serializer=llm.selector_serializer,
        )

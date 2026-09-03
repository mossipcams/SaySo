"""Focused checks for compiled tool envelope validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import custom_components
import pytest
from homeassistant.components import conversation
from homeassistant.components.fan import FanEntity
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import intent, llm
from homeassistant.helpers.llm import LLM_API_ASSIST
from homeassistant.loader import DATA_CUSTOM_COMPONENTS
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.client import LlamaCppClient
from custom_components.sayso.const import DOMAIN, ERROR_ACTION_FAILED, ERROR_MODEL_UNAVAILABLE
from custom_components.sayso.exceptions import SaySoInvalidToolEnvelopeError
from custom_components.sayso.schema import (
    clear_compile_cache,
    compile_llm_tools,
    compile_tools,
    validate_compiled_tool_envelope,
)
from tests.test_config_flow import (
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_COMPONENTS_PATH = str(REPO_ROOT / "custom_components")


def _valid_tool(name: str = "AlphaTool") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
            },
        },
    }


@pytest.fixture(autouse=True)
def enable_custom_integrations(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow Home Assistant to discover this repository's custom components."""
    monkeypatch.setattr(custom_components, "__path__", [CUSTOM_COMPONENTS_PATH])
    hass.data.pop(DATA_CUSTOM_COMPONENTS, None)


@pytest.fixture(autouse=True)
async def setup_required_integrations(hass: HomeAssistant) -> None:
    """Load Home Assistant integrations required by SaySo tests."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "llm", {})
    assert await async_setup_component(hass, "conversation", {})


@pytest.fixture(autouse=True)
def _reset_compile_cache() -> None:
    clear_compile_cache()
    yield
    clear_compile_cache()


@pytest.fixture
def mock_llama_client() -> Any:
    """Patch llama.cpp connectivity checks during SaySo setup."""
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(return_value=[MODEL_ID]),
    ), patch.object(
        LlamaCppClient,
        "validate_model",
        new=AsyncMock(return_value=None),
    ):
        yield


async def _create_entry(hass: HomeAssistant) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return hass.config_entries.async_entries(DOMAIN)[0]


def _speech(result: conversation.ConversationResult) -> str:
    return result.response.speech["plain"]["speech"]


def test_validate_rejects_duplicate_function_names() -> None:
    """Duplicate function.name values fail before transport."""
    tools = [_valid_tool("SameName"), _valid_tool("SameName")]

    with pytest.raises(SaySoInvalidToolEnvelopeError, match="Duplicate function name"):
        validate_compiled_tool_envelope(tools)


@pytest.mark.parametrize(
    ("name", "pattern"),
    [
        ("", "function name"),
        ("a" * 65, "function name"),
        ("bad name", "function name"),
        ("bad.name", "function name"),
    ],
)
def test_validate_rejects_invalid_function_names(name: str, pattern: str) -> None:
    """Names outside the OpenAI-compatible rule are rejected."""
    with pytest.raises(SaySoInvalidToolEnvelopeError, match=pattern):
        validate_compiled_tool_envelope([_valid_tool(name)])


@pytest.mark.parametrize(
    "parameters",
    [
        "not-an-object",
        {"type": "string"},
        {"type": "array"},
    ],
)
def test_validate_rejects_non_object_parameter_roots(parameters: Any) -> None:
    """Parameter roots must be JSON object schemas."""
    tool = _valid_tool()
    tool["function"]["parameters"] = parameters

    with pytest.raises(SaySoInvalidToolEnvelopeError, match="parameters"):
        validate_compiled_tool_envelope([tool])


@pytest.mark.parametrize(
    "tool",
    [
        {"function": {"name": "AlphaTool", "parameters": {"type": "object"}}},
        {"type": "tool", "function": {"name": "AlphaTool", "parameters": {"type": "object"}}},
        {"type": "function", "function": "not-a-mapping"},
        {"type": "function", "function": {"parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "AlphaTool"}},
    ],
)
def test_validate_rejects_malformed_function_wrappers(tool: dict[str, Any]) -> None:
    """Malformed outer function wrappers are rejected."""
    with pytest.raises(SaySoInvalidToolEnvelopeError):
        validate_compiled_tool_envelope([tool])


def test_validate_rejects_non_json_serializable_values() -> None:
    """Non-JSON-serializable compiled values fail before transport."""
    tool = _valid_tool()
    tool["function"]["parameters"]["properties"]["mode"] = {1, 2, 3}

    with pytest.raises(SaySoInvalidToolEnvelopeError, match="JSON-serializable"):
        validate_compiled_tool_envelope([tool])


class _RoutingTestLight(LightEntity):
    _attr_name = "Kitchen Light"
    _attr_unique_id = "kitchen-light"
    _attr_color_mode = ColorMode.BRIGHTNESS
    _attr_supported_color_modes = {ColorMode.BRIGHTNESS}
    _attr_brightness = 128


class _RoutingTestFan(FanEntity):
    _attr_name = "Bedroom Fan"
    _attr_unique_id = "bedroom-fan"
    _attr_percentage = 50


@pytest.fixture
async def representative_ha_tools(hass: HomeAssistant) -> list[llm.Tool]:
    """Load representative Home Assistant 2026.8.3 assist tools."""
    assert await async_setup_component(hass, "intent", {})
    setup_test_component_platform(hass, "light", [_RoutingTestLight()])
    setup_test_component_platform(hass, "fan", [_RoutingTestFan()])
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    assert await async_setup_component(hass, "fan", {"fan": {"platform": "test"}})
    await hass.async_block_till_done()

    llm_context = llm.LLMContext(
        platform="conversation",
        context=None,
        language="en",
        assistant="conversation",
        device_id=None,
    )
    api = await llm.async_get_api(hass, LLM_API_ASSIST, llm_context)
    return list(api.tools)


def test_representative_ha_tools_compile_with_valid_envelope(
    representative_ha_tools: list[llm.Tool],
) -> None:
    """Representative Home Assistant 2026.8.3 tools still compile unchanged."""
    compiled = compile_tools(representative_ha_tools)

    validate_compiled_tool_envelope(compiled)

    assert compiled
    assert all(tool["type"] == "function" for tool in compiled)
    assert all(tool["function"]["parameters"]["type"] == "object" for tool in compiled)


async def test_compile_llm_tools_validates_envelope(
    hass: HomeAssistant,
    representative_ha_tools: list[llm.Tool],
) -> None:
    """compile_llm_tools returns a validated compiled schema."""
    llm_context = llm.LLMContext(
        platform="conversation",
        context=None,
        language="en",
        assistant="conversation",
        device_id=None,
    )
    api = await llm.async_get_api(hass, LLM_API_ASSIST, llm_context)

    compiled_schema = compile_llm_tools(api)

    assert compiled_schema is not None
    validate_compiled_tool_envelope(compiled_schema.tools)


async def test_invalid_tool_envelope_returns_action_failed(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Envelope compile failures return ERROR_ACTION_FAILED, not model unavailable."""
    entry = await _create_entry(hass)

    with patch(
        "custom_components.sayso.conversation.compile_llm_tools",
        side_effect=SaySoInvalidToolEnvelopeError("invalid compiled tool envelope"),
    ), patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(),
    ) as mock_chat:
        result = await conversation.async_converse(
            hass,
            "Turn on the living room light",
            None,
            Context(),
            agent_id=entry.entry_id,
        )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED
    assert _speech(result) != ERROR_MODEL_UNAVAILABLE
    mock_chat.assert_not_called()

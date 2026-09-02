"""Focused checks that each request phase sends and records its schema identity."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import custom_components
import pytest
from homeassistant.components.fan import FanEntity
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import intent, llm
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient, ToolCall
from custom_components.sayso.const import DOMAIN
from custom_components.sayso.diagnostics import (
    async_get_config_entry_diagnostics,
    clear_boundary_diagnostics,
)
from custom_components.sayso.routing import select_schema_for_domain
from custom_components.sayso.schema import (
    ToolArgumentFailureCode,
    clear_compile_cache,
    compile_llm_tools,
    schema_fingerprint,
)
from homeassistant.components import conversation
from homeassistant.loader import DATA_CUSTOM_COMPONENTS
from homeassistant.setup import async_setup_component

from tests.test_config_flow import (
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)


class _TestLight(LightEntity):
    _attr_name = "Living Room"
    _attr_unique_id = "living_room"
    _attr_supported_color_modes = {ColorMode.ONOFF}
    _attr_color_mode = ColorMode.ONOFF

    def __init__(self) -> None:
        self._is_on = False

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._is_on = True
        self.async_write_ha_state()


class _BedroomFan(FanEntity):
    _attr_name = "Bedroom Fan"
    _attr_unique_id = "bedroom_fan"

    def __init__(self) -> None:
        self._is_on = False
        self._percentage = 0

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def percentage(self) -> int:
        return self._percentage


REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_COMPONENTS_PATH = str(REPO_ROOT / "custom_components")


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
def _reset_schema_state() -> None:
    clear_compile_cache()
    clear_boundary_diagnostics()
    yield
    clear_compile_cache()
    clear_boundary_diagnostics()


@pytest.fixture
def mock_llama_client() -> Any:
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


@pytest.fixture
async def assist_light_and_fan(hass: HomeAssistant) -> None:
    setup_test_component_platform(hass, "light", [_TestLight()])
    setup_test_component_platform(hass, "fan", [_BedroomFan()])
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    assert await async_setup_component(hass, "fan", {"fan": {"platform": "test"}})
    assert await async_setup_component(hass, "intent", {})
    await hass.async_block_till_done()


async def _create_entry(hass: HomeAssistant) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return hass.config_entries.async_entries(DOMAIN)[0]


async def _converse(hass: HomeAssistant, entry: Any, text: str) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass,
        text,
        None,
        Context(),
        agent_id=entry.entry_id,
    )


async def _schemas_for_command(
    hass: HomeAssistant,
    entry: Any,
    command: str,
) -> tuple[Any, Any]:
    llm_context = llm.LLMContext(
        platform=DOMAIN,
        context=None,
        language="en",
        assistant="conversation",
        device_id=None,
    )
    llm_api = await llm.async_get_api(hass, entry.runtime_data.llm_api, llm_context)
    complete_schema = compile_llm_tools(llm_api)
    assert complete_schema is not None
    from custom_components.sayso.routing import (
        build_routing_catalog,
        build_routing_preferences,
        build_routing_registries,
        identify_command_domain,
    )

    domain_hint = identify_command_domain(
        command,
        build_routing_catalog(hass, assistant=llm_context.assistant),
        registries=build_routing_registries(hass),
        preferences=build_routing_preferences(hass, llm_context),
    )
    active_schema = select_schema_for_domain(
        complete_schema,
        llm_api.tools,
        domain_hint,
    )
    return complete_schema, active_schema


def _tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str):
            import json

            results[message["tool_call_id"]] = json.loads(content)
    return results


async def test_initial_request_sends_active_schema_identity(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light_and_fan: None,
) -> None:
    """Confident routing must send the active schema tools and fingerprint on the first call."""
    entry = await _create_entry(hass)
    complete_schema, active_schema = await _schemas_for_command(
        hass,
        entry,
        "Turn on the living room light",
    )
    assert active_schema.fingerprint != complete_schema.fingerprint

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content="Done.", tool_calls=[]),
        ),
    ) as mock_chat:
        await _converse(hass, entry, "Turn on the living room light")

    initial_tools = mock_chat.await_args_list[0].kwargs["tools"]
    assert schema_fingerprint(initial_tools) == active_schema.fingerprint
    assert schema_fingerprint(initial_tools) != complete_schema.fingerprint


async def test_follow_up_request_sends_active_schema_identity(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light_and_fan: None,
) -> None:
    """Follow-up model calls reuse the active schema identity, not the complete schema."""
    entry = await _create_entry(hass)
    complete_schema, active_schema = await _schemas_for_command(
        hass,
        entry,
        "Turn on the living room light",
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="HassTurnOn",
                            arguments={"name": "Living Room"},
                        )
                    ],
                ),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ) as mock_chat:
        await _converse(hass, entry, "Turn on the living room light")

    for call in mock_chat.await_args_list:
        tools = call.kwargs["tools"]
        assert schema_fingerprint(tools) == active_schema.fingerprint
        assert schema_fingerprint(tools) != complete_schema.fingerprint


async def test_argument_correction_sends_complete_schema_identity(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light_and_fan: None,
) -> None:
    """Argument correction must send and cite the complete schema fingerprint."""
    entry = await _create_entry(hass)
    complete_schema, active_schema = await _schemas_for_command(
        hass,
        entry,
        "Turn on the living room light",
    )
    invalid_call = ToolCall(
        id="call_bad",
        name="HassTurnOn",
        arguments={"name": "Living Room", "unexpected": True},
    )
    corrected_call = ToolCall(
        id="call_fixed",
        name="HassTurnOn",
        arguments={"name": "Living Room"},
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[invalid_call]),
                ChatCompletionResult(content=None, tool_calls=[corrected_call]),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ) as mock_chat:
        await _converse(hass, entry, "Turn on the living room light")

    initial_tools = mock_chat.await_args_list[0].kwargs["tools"]
    correction_tools = mock_chat.await_args_list[1].kwargs["tools"]
    assert schema_fingerprint(initial_tools) == active_schema.fingerprint
    assert schema_fingerprint(correction_tools) == complete_schema.fingerprint

    correction_messages = mock_chat.await_args_list[1].args[0]
    tool_results = _tool_results_by_id(correction_messages)
    assert (
        tool_results["call_bad"]["error"]["schema_fingerprint"]
        == complete_schema.fingerprint
    )


async def test_filtered_miss_correction_sends_complete_schema_identity(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light_and_fan: None,
) -> None:
    """Filtered-schema miss correction must send the complete schema identity."""
    entry = await _create_entry(hass)
    complete_schema, active_schema = await _schemas_for_command(
        hass,
        entry,
        "Turn on the living room light",
    )
    filtered_call = ToolCall(
        id="call_fan",
        name="HassFanSetSpeed",
        arguments={"name": "Bedroom Fan", "percentage": 50},
    )
    corrected_call = ToolCall(
        id="call_light",
        name="HassTurnOn",
        arguments={"name": "Living Room"},
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[filtered_call]),
                ChatCompletionResult(content=None, tool_calls=[corrected_call]),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ) as mock_chat:
        await _converse(hass, entry, "Turn on the living room light")

    initial_tools = mock_chat.await_args_list[0].kwargs["tools"]
    correction_tools = mock_chat.await_args_list[1].kwargs["tools"]
    assert schema_fingerprint(initial_tools) == active_schema.fingerprint
    assert schema_fingerprint(correction_tools) == complete_schema.fingerprint

    correction_messages = mock_chat.await_args_list[1].args[0]
    tool_results = _tool_results_by_id(correction_messages)
    assert tool_results["call_fan"]["error"]["code"] == (
        ToolArgumentFailureCode.SCHEMA_MISMATCH
    )
    assert (
        tool_results["call_fan"]["error"]["schema_fingerprint"]
        == complete_schema.fingerprint
    )


async def test_initial_boundary_diagnostic_uses_active_schema_fingerprint(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light_and_fan: None,
) -> None:
    """Boundary diagnostics on the initial phase must record the active schema fingerprint."""
    entry = await _create_entry(hass)
    complete_schema, active_schema = await _schemas_for_command(
        hass,
        entry,
        "Turn on the living room light",
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="NotARealTool",
                        arguments={},
                    )
                ],
            )
        ),
    ):
        await _converse(hass, entry, "Turn on the living room light")

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    assert diagnostics["boundary"]["last"]["fingerprint"] == active_schema.fingerprint
    assert diagnostics["boundary"]["last"]["fingerprint"] != complete_schema.fingerprint

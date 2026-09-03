"""Tests for SaySo diagnostics."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.diagnostics import REDACTED
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient, ToolCall
from custom_components.sayso.const import CONF_MAX_TOOL_ITERATIONS, DOMAIN
from custom_components.sayso.diagnostics import (
    async_get_config_entry_diagnostics,
    clear_boundary_diagnostics,
)
from custom_components.sayso.exceptions import SaySoTimeoutError
from tests.test_config_flow import (
    BASE_URL,
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)


class _TestLight(LightEntity):
    """Light used to exercise boundary diagnostics through real tool execution."""

    _attr_name = "Living Room"
    _attr_unique_id = "living_room"
    _attr_supported_color_modes = {ColorMode.ONOFF}
    _attr_color_mode = ColorMode.ONOFF

    def __init__(self) -> None:
        """Initialize the test light."""
        self._is_on = False

    @property
    def is_on(self) -> bool:
        """Return if the light is on."""
        return self._is_on

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the light on."""
        self._is_on = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the light off."""
        self._is_on = False
        self.async_write_ha_state()


@pytest.fixture(autouse=True)
def _reset_boundary_diagnostics() -> None:
    """Isolate boundary diagnostics between tests."""
    clear_boundary_diagnostics()
    yield
    clear_boundary_diagnostics()


@pytest.fixture
async def assist_light(hass: HomeAssistant) -> None:
    """Register a test light for Assist tool execution."""
    setup_test_component_platform(hass, "light", [_TestLight()])
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    assert await async_setup_component(hass, "intent", {})
    await hass.async_block_till_done()


@pytest.fixture
def mock_llama_client() -> Any:
    """Patch llama.cpp connectivity checks during setup."""
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


async def _create_entry_with_api_key(
    hass: HomeAssistant,
    api_key: str,
) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(
        hass, result["flow_id"], api_key=api_key
    )
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    await hass.async_block_till_done()
    return entry


async def test_api_key_redacted_from_diagnostics(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test API keys are redacted from config entry diagnostics."""
    secret = "super-secret-api-key"
    entry = await _create_entry_with_api_key(hass, secret)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    serialized = json.dumps(diagnostics)

    assert diagnostics["entry"]["data"][CONF_API_KEY] == REDACTED
    assert secret not in serialized
    assert diagnostics["connectivity"]["api_key_configured"] is True
    assert diagnostics["connectivity"]["reachable"] is True
    assert diagnostics["connectivity"]["models"] == [MODEL_ID]
    assert diagnostics["runtime"]["loaded"] is True
    assert diagnostics["runtime"]["model"] == MODEL_ID


async def test_diagnostics_without_api_key(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test diagnostics when no API key is configured."""
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    await hass.async_block_till_done()

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert CONF_API_KEY not in diagnostics["entry"]["data"]
    assert diagnostics["connectivity"]["api_key_configured"] is False
    assert diagnostics["entry"]["data"][CONF_URL] == BASE_URL


async def _converse(hass: HomeAssistant, entry: Any, text: str) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass,
        text,
        None,
        Context(),
        agent_id=entry.entry_id,
    )


def _assert_safe_boundary_diagnostics(
    diagnostics: dict[str, Any],
    *,
    expected_code: str,
    expected_phase: str | None = None,
) -> None:
    """Assert boundary diagnostics expose counts and safe last-failure metadata."""
    boundary = diagnostics["boundary"]
    assert boundary["counts"][expected_code] >= 1
    last = boundary["last"]
    assert last is not None
    assert last["code"] == expected_code
    assert isinstance(last["phase"], str) and last["phase"]
    if expected_phase is not None:
        assert last["phase"] == expected_phase
    assert isinstance(last["timestamp"], str) and last["timestamp"]
    if last.get("fingerprint") is not None:
        assert isinstance(last["fingerprint"], str)
    if expected_code == "tool_execution_failed":
        assert isinstance(last.get("ha_error"), str) and last["ha_error"]
        assert "error_text" not in last

    serialized = json.dumps(boundary).lower()
    for forbidden in (
        "living room",
        "nonexistent room",
        "kitchen",
        "hassturnon",
        "notarealtool",
        "unexpected",
        '"name"',
        "super-secret",
        "http://",
        "https://",
        "turn on",
        "system prompt",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("expected_code", "side_effect"),
    [
        pytest.param(
            "request_timeout",
            SaySoTimeoutError("llama.cpp request timed out"),
            id="request_timeout",
        ),
        pytest.param(
            "unavailable_tool",
            ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="NotARealTool", arguments={}),
                ],
            ),
            id="unavailable_tool",
        ),
        pytest.param(
            "schema_mismatch",
            ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="HassTurnOn",
                        arguments={"name": "Living Room", "unexpected": True},
                    ),
                ],
            ),
            id="schema_mismatch",
        ),
        pytest.param(
            "invalid_arguments",
            ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="HassTurnOn",
                        arguments={"name": ["Living Room"]},
                    ),
                ],
            ),
            id="invalid_arguments",
        ),
        pytest.param(
            "tool_execution_failed",
            ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="HassTurnOn",
                        arguments={"name": "Nonexistent Room"},
                    ),
                ],
            ),
            id="tool_execution_failed",
        ),
    ],
)
async def test_boundary_diagnostics_record_failure_codes(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
    expected_code: str,
    side_effect: Any,
) -> None:
    """Test each boundary code is counted with safe last-failure metadata."""
    entry = await _create_entry_with_api_key(hass, "super-secret-api-key")

    mock_side_effect: Any
    if isinstance(side_effect, SaySoTimeoutError):
        mock_side_effect = side_effect
    elif expected_code in {"schema_mismatch", "invalid_arguments"}:
        mock_side_effect = [side_effect, side_effect]
    else:
        mock_side_effect = [side_effect]

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=mock_side_effect),
    ):
        await _converse(hass, entry, "Turn on the living room light")

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    _assert_safe_boundary_diagnostics(diagnostics, expected_code=expected_code)


async def test_boundary_diagnostics_record_iteration_limit(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test iteration-limit failures appear in boundary diagnostics."""
    entry = await _create_entry_with_api_key(hass, "super-secret-api-key")
    hass.config_entries.async_update_entry(
        entry,
        options={**dict(entry.options), CONF_MAX_TOOL_ITERATIONS: 1},
    )
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    entry = hass.config_entries.async_get_entry(entry.entry_id)

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
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="HassTurnOff",
                            arguments={"name": "Living Room"},
                        )
                    ],
                ),
            ]
        ),
    ):
        result = await _converse(hass, entry, "Turn on the living room light")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    _assert_safe_boundary_diagnostics(
        diagnostics,
        expected_code="iteration_limit",
        expected_phase="follow_up",
    )


@pytest.mark.parametrize(
    ("expected_phase", "side_effect"),
    [
        pytest.param(
            "initial",
            SaySoTimeoutError("llama.cpp request timed out"),
            id="initial_timeout",
        ),
        pytest.param(
            "correction",
            [
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_bad",
                            name="HassTurnOn",
                            arguments={"name": "Living Room", "unexpected": True},
                        )
                    ],
                ),
                SaySoTimeoutError("llama.cpp request timed out"),
            ],
            id="correction_timeout",
        ),
        pytest.param(
            "follow_up",
            [
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
                SaySoTimeoutError("llama.cpp request timed out"),
            ],
            id="follow_up_timeout",
        ),
    ],
)
async def test_boundary_diagnostics_record_timeout_phases(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
    expected_phase: str,
    side_effect: Any,
) -> None:
    """Test request_timeout diagnostics record the correct phase."""
    entry = await _create_entry_with_api_key(hass, "super-secret-api-key")

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=side_effect),
    ):
        result = await _converse(hass, entry, "Turn on the living room light")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    _assert_safe_boundary_diagnostics(
        diagnostics,
        expected_code="request_timeout",
        expected_phase=expected_phase,
    )

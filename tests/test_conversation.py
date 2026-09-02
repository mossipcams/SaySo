"""Tests for the SaySo conversation entity."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient, ToolCall
from custom_components.sayso.const import (
    CONF_MAX_TOOL_ITERATIONS,
    DOMAIN,
    ERROR_ACTION_FAILED,
    ERROR_EMPTY_RESPONSE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_REQUEST_TIMEOUT,
    ERROR_TOOL_ITERATION_LIMIT,
)
from custom_components.sayso.exceptions import (
    SaySoConnectionError,
    SaySoHttpError,
    SaySoInvalidResponseError,
    SaySoTimeoutError,
)
from tests.test_config_flow import (
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)


class _TestLight(LightEntity):
    """Light used to exercise real HassTurnOn tool execution."""

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


@pytest.fixture
async def assist_light(hass: HomeAssistant) -> None:
    """Register a test light for Assist tool execution."""
    setup_test_component_platform(hass, "light", [_TestLight()])
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    assert await async_setup_component(hass, "intent", {})
    await hass.async_block_till_done()


async def _create_entry(hass: HomeAssistant) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return hass.config_entries.async_entries(DOMAIN)[0]


def _speech(result: conversation.ConversationResult) -> str:
    return result.response.speech["plain"]["speech"]


async def _converse(
    hass: HomeAssistant,
    entry: Any,
    text: str,
    *,
    conversation_id: str | None = None,
    context: Context | None = None,
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass,
        text,
        conversation_id,
        context or Context(),
        agent_id=entry.entry_id,
    )


async def test_plain_conversational_response(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test a plain assistant response from llama.cpp."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content="The living room light is on.",
                tool_calls=[],
            )
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Is the living room light on?")

    mock_chat.assert_awaited_once()
    messages = mock_chat.await_args.args[0]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Is the living room light on?"
    assert messages[0]["role"] == "system"
    assert "SaySo" in messages[0]["content"]
    assert mock_chat.await_args.kwargs.get("tools") is not None
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert _speech(result) == "The living room light is on."


async def test_conversation_id_preservation(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test conversation_id is preserved across turns."""
    entry = await _create_entry(hass)
    conversation_id = "sayso-test-conversation"

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content="Hello.", tool_calls=[])
        ),
    ):
        result = await _converse(
            hass,
            entry,
            "Hello",
            conversation_id=conversation_id,
        )

    assert result.conversation_id == conversation_id


async def test_llama_cpp_timeout(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test timeout errors return a short spoken failure."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=SaySoTimeoutError("llama.cpp request timed out")),
    ):
        result = await _converse(hass, entry, "Turn on the light")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_REQUEST_TIMEOUT


async def test_llama_cpp_http_error(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test HTTP errors return a short spoken failure."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=SaySoHttpError(503)),
    ):
        result = await _converse(hass, entry, "Turn on the light")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_MODEL_UNAVAILABLE


async def test_empty_model_response(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test empty assistant content fails closed."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="   ", tool_calls=[])),
    ):
        result = await _converse(hass, entry, "Hello")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_EMPTY_RESPONSE


async def test_successful_light_tool_call(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test a successful HassTurnOn tool call through real HA LLM plumbing."""
    entry = await _create_entry(hass)

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
                    content="The living room light is on.",
                    tool_calls=[],
                ),
            ]
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    assert hass.states.get("light.living_room").state == "on"
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert _speech(result) == "The living room light is on."


async def test_entity_state_query_tool_call(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test an entity state query via GetLiveContext."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_ctx",
                            name="GetLiveContext",
                            arguments={},
                        )
                    ],
                ),
                ChatCompletionResult(
                    content="The living room light is off.",
                    tool_calls=[],
                ),
            ]
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Is the living room light on?")

    assert mock_chat.await_count == 2
    follow_up_messages = mock_chat.await_args_list[1].args[0]
    assert any(message.get("role") == "tool" for message in follow_up_messages)
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert _speech(result) == "The living room light is off."


async def test_tool_result_returned_to_llama_cpp(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test assistant tool calls and tool results are sent back to llama.cpp."""
    entry = await _create_entry(hass)

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

    follow_up_messages = mock_chat.await_args_list[1].args[0]
    assistant_tool_message = next(
        message
        for message in follow_up_messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    tool_result_message = next(
        message for message in follow_up_messages if message.get("role") == "tool"
    )
    assert assistant_tool_message["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert tool_result_message["tool_call_id"] == "call_1"
    assert "error" not in tool_result_message["content"]


async def test_home_assistant_context_preservation(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test the original Home Assistant Context is preserved for tool execution."""
    entry = await _create_entry(hass)
    request_context = Context(id="sayso-context-test")

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
    ), patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        await _converse(
            hass,
            entry,
            "Turn on the living room light",
            context=request_context,
        )

    assert mock_handle.await_count >= 1
    assert mock_handle.await_args.kwargs["context"] is request_context


async def test_unknown_tool_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test unknown tool names fail closed without claiming success."""
    entry = await _create_entry(hass)

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
    ) as mock_chat:
        result = await _converse(hass, entry, "Do something impossible")

    mock_chat.assert_awaited_once()
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_sequential_tool_calls_succeed(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test multiple tool-call iterations succeed until final text."""
    entry = await _create_entry(hass)

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
                ChatCompletionResult(
                    content="The living room light is off.",
                    tool_calls=[],
                ),
            ]
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Toggle the living room light")

    assert mock_chat.await_count == 3
    assert hass.states.get("light.living_room").state == "off"
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert _speech(result) == "The living room light is off."


async def test_max_tool_iterations_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test exceeding max tool iterations fails closed without claiming success."""
    entry = await _create_entry(hass)
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
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_TOOL_ITERATION_LIMIT


async def test_malformed_tool_arguments_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test malformed tool arguments fail closed before execution."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="",
                        name="HassTurnOn",
                        arguments={"name": "Living Room"},
                    )
                ],
            )
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the living room light")

    mock_chat.assert_awaited_once()
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_non_object_tool_arguments_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test non-object tool arguments fail closed before execution."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="HassTurnOn",
                        arguments=[],  # type: ignore[arg-type]
                    )
                ],
            )
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the living room light")

    mock_chat.assert_awaited_once()
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_tool_validation_failure_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test Home Assistant tool validation errors fail closed."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="HassTurnOn",
                        arguments={"name": 123},
                    )
                ],
            )
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the living room light")

    mock_chat.assert_awaited_once()
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_tool_execution_failure_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test Home Assistant tool execution errors fail closed."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="HassTurnOn",
                        arguments={"name": "Nonexistent Room"},
                    )
                ],
            )
        ),
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the nonexistent room light")

    mock_chat.assert_awaited_once()
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_connection_error(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test unreachable llama.cpp returns a short spoken failure."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=SaySoConnectionError("llama.cpp is unreachable")),
    ):
        result = await _converse(hass, entry, "Hello")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_MODEL_UNAVAILABLE


async def test_invalid_llama_response_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test invalid llama.cpp responses fail closed."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=SaySoInvalidResponseError(
                "llama.cpp returned neither content nor tool calls"
            )
        ),
    ):
        result = await _converse(hass, entry, "Turn on the light")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_MODEL_UNAVAILABLE

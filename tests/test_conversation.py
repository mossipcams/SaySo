"""Tests for the SaySo conversation entity."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import intent, llm
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient, ToolCall
from custom_components.sayso.conversation import _chat_log_to_messages
from custom_components.sayso.schema import (
    ToolArgumentFailureCode,
    _build_compiled_tools_from_source,
    clear_compile_cache,
    compile_llm_tools,
    schema_fingerprint,
)
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


@pytest.fixture(autouse=True)
def _reset_schema_compile_cache() -> None:
    """Isolate compile cache between conversation tests."""
    clear_compile_cache()
    yield
    clear_compile_cache()


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


def test_chat_log_serializes_batched_tool_calls_deterministically() -> None:
    """Test one assistant batch and its ordered mixed tool results."""
    content: list[conversation.Content] = [
        conversation.AssistantContent(
            agent_id="conversation.sayso",
            tool_calls=[
                llm.ToolInput(
                    id="call_1",
                    tool_name="HassTurnOn",
                    tool_args={"name": "Living Room", "area": "downstairs"},
                ),
                llm.ToolInput(
                    id="call_2",
                    tool_name="HassTurnOff",
                    tool_args={"name": "Kitchen", "area": "upstairs"},
                ),
            ],
        ),
        conversation.ToolResultContent(
            agent_id="conversation.sayso",
            tool_call_id="call_1",
            tool_name="HassTurnOn",
            tool_result={"success": True},
        ),
        conversation.ToolResultContent(
            agent_id="conversation.sayso",
            tool_call_id="call_2",
            tool_name="HassTurnOff",
            tool_result={"error": "Kitchen is unavailable"},
        ),
    ]

    assert _chat_log_to_messages(content) == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": '{"area": "downstairs", "name": "Living Room"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "HassTurnOff",
                        "arguments": '{"area": "upstairs", "name": "Kitchen"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"success": true}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"error": "Kitchen is unavailable"}',
        },
    ]


def _assistant_tool_batches(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return each assistant message's tool_calls payload in transcript order."""
    return [
        message["tool_calls"]
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]


def _tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool_call_id to parsed tool result payloads."""
    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        results[message["tool_call_id"]] = json.loads(message["content"])
    return results


def _assert_batch_transcript(
    messages: list[dict[str, Any]],
    *,
    expected_call_ids: list[str],
    expected_errors: dict[str, bool],
) -> None:
    """Assert one assistant turn holds every call, then ordered tool results."""
    batches = _assistant_tool_batches(messages)
    assert len(batches) == 1
    assert [call["id"] for call in batches[0]] == expected_call_ids

    assistant_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    tool_messages = [
        message
        for message in messages[assistant_index + 1 :]
        if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == expected_call_ids

    results_by_id = _tool_results_by_id(messages)
    for call_id, should_error in expected_errors.items():
        assert ("error" in results_by_id[call_id]) is should_error


@pytest.mark.parametrize(
    ("tool_calls", "follow_up", "expect_success", "expect_follow_up"),
    [
        pytest.param(
            [
                ToolCall(
                    id="call_single",
                    name="HassTurnOn",
                    arguments={"name": "Living Room"},
                )
            ],
            ChatCompletionResult(content="Single call done.", tool_calls=[]),
            True,
            True,
            id="single_success",
        ),
        pytest.param(
            [
                ToolCall(
                    id="call_a",
                    name="HassTurnOn",
                    arguments={"name": "Living Room"},
                ),
                ToolCall(
                    id="call_b",
                    name="HassTurnOff",
                    arguments={"name": "Living Room"},
                ),
            ],
            ChatCompletionResult(content="Both calls done.", tool_calls=[]),
            True,
            True,
            id="multiple_success",
        ),
        pytest.param(
            [
                ToolCall(
                    id="call_fail",
                    name="HassTurnOn",
                    arguments={"name": "Nonexistent Room"},
                )
            ],
            None,
            False,
            False,
            id="failed_batch",
        ),
        pytest.param(
            [
                ToolCall(
                    id="call_ok",
                    name="HassTurnOn",
                    arguments={"name": "Living Room"},
                ),
                ToolCall(
                    id="call_bad",
                    name="HassTurnOn",
                    arguments={"name": "Nonexistent Room"},
                ),
            ],
            None,
            False,
            False,
            id="partial_success_batch",
        ),
    ],
)
async def test_batched_tool_calls_transcript_and_follow_up(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
    tool_calls: list[ToolCall],
    follow_up: ChatCompletionResult | None,
    expect_success: bool,
    expect_follow_up: bool,
) -> None:
    """Test one assistant message batches calls and follow-up only on full success."""
    entry = await _create_entry(hass)
    captured_chat_logs: list[list[Any]] = []

    def _capture_error(
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        speech: str,
    ) -> conversation.ConversationResult:
        captured_chat_logs.append(list(chat_log.content))
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            speech,
        )
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
        )

    side_effect: list[ChatCompletionResult] = [
        ChatCompletionResult(content=None, tool_calls=tool_calls)
    ]
    if follow_up is not None:
        side_effect.append(follow_up)

    expected_call_ids = [tool_call.id for tool_call in tool_calls]
    expected_errors = {
        "call_fail": True,
        "call_ok": False,
        "call_bad": True,
        "call_single": False,
        "call_a": False,
        "call_b": False,
    }

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=side_effect),
    ) as mock_chat, patch(
        "custom_components.sayso.conversation._error_result",
        side_effect=_capture_error,
    ):
        result = await _converse(hass, entry, "Run the batched tool calls")

    if expect_follow_up:
        assert mock_chat.await_count == 2
        transcript = mock_chat.await_args_list[1].args[0]
    else:
        mock_chat.assert_awaited_once()
        assert captured_chat_logs
        transcript = _chat_log_to_messages(captured_chat_logs[-1])

    _assert_batch_transcript(
        transcript,
        expected_call_ids=expected_call_ids,
        expected_errors={
            call_id: expected_errors[call_id]
            for call_id in expected_call_ids
        },
    )

    if expect_success:
        assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
        assert _speech(result) == follow_up.content
    else:
        assert result.response.response_type == intent.IntentResponseType.ERROR
        assert _speech(result) == ERROR_ACTION_FAILED


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


async def test_model_turn_compiled_schema_and_fingerprint(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Initial and every follow-up llama.cpp call share one compiled schema and fingerprint."""
    entry = await _create_entry(hass)
    compile_calls: list[Any] = []

    def _spy_compile(llm_api: Any) -> Any:
        compiled = compile_llm_tools(llm_api)
        compile_calls.append(compiled)
        return compiled

    with patch(
        "custom_components.sayso.conversation.compile_llm_tools",
        side_effect=_spy_compile,
    ), patch.object(
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
        await _converse(hass, entry, "Toggle the living room light")

    assert len(compile_calls) == 1
    compiled = compile_calls[0]
    assert compiled is not None
    assert compiled.fingerprint == schema_fingerprint(compiled.tools)

    tools_payloads = [call.kwargs.get("tools") for call in mock_chat.await_args_list]
    assert mock_chat.await_count == len(tools_payloads) == 3
    assert all(tools is compiled.tools for tools in tools_payloads)
    assert len({schema_fingerprint(tools) for tools in tools_payloads}) == 1


async def test_tool_follow_up_reuses_compiled_schema_without_reconversion(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Follow-up model iterations reuse one compiled schema for the whole turn."""
    entry = await _create_entry(hass)
    build_calls = 0

    def counting_build(source_json: str) -> tuple[dict[str, Any], ...]:
        nonlocal build_calls
        build_calls += 1
        return _build_compiled_tools_from_source(source_json)

    with patch(
        "custom_components.sayso.schema._build_compiled_tools_from_source",
        side_effect=counting_build,
    ), patch.object(
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
        await _converse(hass, entry, "Toggle the living room light")

    assert build_calls == 1
    assert mock_chat.await_count == 3
    tools_payloads = [
        call.kwargs.get("tools") for call in mock_chat.await_args_list
    ]
    assert tools_payloads[0] is not None
    assert all(tools is tools_payloads[0] for tools in tools_payloads)


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


@pytest.mark.parametrize(
    ("second_call",),
    [
        pytest.param(
            ToolCall(
                id="",
                name="HassTurnOn",
                arguments={"name": "Living Room"},
            ),
            id="missing_id",
        ),
        pytest.param(
            ToolCall(
                id="call_ok",
                name="HassTurnOn",
                arguments={"name": "Living Room"},
            ),
            id="duplicate_id",
        ),
        pytest.param(
            ToolCall(
                id="call_bad",
                name="NotARealTool",
                arguments={},
            ),
            id="unknown_name",
        ),
    ],
)
async def test_invalid_batch_prevalidation_executes_no_tools(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
    second_call: ToolCall,
) -> None:
    """Test a structurally invalid second call prevents any HA tool execution."""
    entry = await _create_entry(hass)

    tool_calls = [
        ToolCall(
            id="call_ok",
            name="HassTurnOn",
            arguments={"name": "Living Room"},
        ),
        second_call,
    ]

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content=None, tool_calls=tool_calls)
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Run the batched tool calls")

    mock_chat.assert_awaited_once()
    mock_handle.assert_not_called()
    assert hass.states.get("light.living_room").state == "off"
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


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


async def test_pre_execution_correction_repairs_invalid_call(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test one pre-execution correction repairs a repairable invalid call."""
    entry = await _create_entry(hass)
    compiled_schemas: list[Any] = []

    def _capture_compile(llm_api: Any) -> Any:
        compiled = compile_llm_tools(llm_api)
        if compiled is not None:
            compiled_schemas.append(compiled)
        return compiled

    invalid_call = ToolCall(
        id="call_bad",
        name="HassTurnOn",
        arguments={
            "name": "Living Room",
            "unexpected": True,
        },
    )
    corrected_call = ToolCall(
        id="call_fixed",
        name="HassTurnOn",
        arguments={"name": "Living Room"},
    )

    with patch(
        "custom_components.sayso.conversation.compile_llm_tools",
        side_effect=_capture_compile,
    ), patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[invalid_call]),
                ChatCompletionResult(content=None, tool_calls=[corrected_call]),
                ChatCompletionResult(
                    content="The living room light is on.",
                    tool_calls=[],
                ),
            ]
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 3
    assert mock_handle.await_count == 1
    assert hass.states.get("light.living_room").state == "on"
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert _speech(result) == "The living room light is on."

    compiled = compiled_schemas[0]
    correction_call = mock_chat.await_args_list[1]
    correction_messages = correction_call.args[0]
    correction_tools = correction_call.kwargs.get("tools")
    assert correction_tools is compiled.tools
    assert schema_fingerprint(correction_tools) == compiled.fingerprint

    assistant_batches = _assistant_tool_batches(correction_messages)
    assert len(assistant_batches) == 1
    assert assistant_batches[0] == [
        {
            "id": "call_bad",
            "type": "function",
            "function": {
                "name": "HassTurnOn",
                "arguments": '{"name": "Living Room", "unexpected": true}',
            },
        }
    ]

    tool_results = _tool_results_by_id(correction_messages)
    assert set(tool_results) == {"call_bad"}
    synthetic_error = tool_results["call_bad"]["error"]
    assert synthetic_error["code"] == ToolArgumentFailureCode.SCHEMA_MISMATCH
    assert "unexpected" in synthetic_error["message"].lower()
    assert "HassTurnOn" in synthetic_error["allowed_tools"]
    assert synthetic_error["schema_fingerprint"] == compiled.fingerprint

    executed_transcript = mock_chat.await_args_list[2].args[0]
    executed_batches = _assistant_tool_batches(executed_transcript)
    assert len(executed_batches) == 1
    assert executed_batches[0][0]["id"] == "call_fixed"
    assert "call_bad" not in _tool_results_by_id(executed_transcript)


async def test_tool_validation_failure_fails_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test schema validation errors fail closed before HA tool execution."""
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
                        arguments={"name": ["Living Room"]},
                    )
                ],
            )
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    mock_handle.assert_not_called()
    assert hass.states.get("light.living_room").state == "off"
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_tool_schema_mismatch_fails_closed_before_execution(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test unexpected tool arguments fail closed before HA tool execution."""
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
                        arguments={
                            "name": "Living Room",
                            "unexpected": True,
                        },
                    )
                ],
            )
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    mock_handle.assert_not_called()
    assert hass.states.get("light.living_room").state == "off"
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


async def test_second_invalid_response_fails_after_one_correction(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test a second invalid model response fails after one pre-execution correction."""
    entry = await _create_entry(hass)
    invalid_call = ToolCall(
        id="call_bad",
        name="HassTurnOn",
        arguments={"name": "Living Room", "unexpected": True},
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[invalid_call]),
                ChatCompletionResult(content=None, tool_calls=[invalid_call]),
            ]
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    mock_handle.assert_not_called()
    assert hass.states.get("light.living_room").state == "off"
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED


async def test_correction_timeout_never_retries(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test a correction-request timeout fails closed without retrying."""
    entry = await _create_entry(hass)
    invalid_call = ToolCall(
        id="call_bad",
        name="HassTurnOn",
        arguments={"name": "Living Room", "unexpected": True},
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[invalid_call]),
                SaySoTimeoutError("llama.cpp request timed out"),
            ]
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    mock_handle.assert_not_called()
    assert hass.states.get("light.living_room").state == "off"
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_REQUEST_TIMEOUT


async def test_initial_timeout_never_retries(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test an initial request timeout makes exactly one client call."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=SaySoTimeoutError("llama.cpp request timed out")),
    ) as mock_chat:
        result = await _converse(hass, entry, "Turn on the light")

    mock_chat.assert_awaited_once()
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_REQUEST_TIMEOUT


async def test_follow_up_timeout_never_retries_after_tool_execution(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test a follow-up timeout after tool execution never retries or re-executes."""
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
                SaySoTimeoutError("llama.cpp request timed out"),
            ]
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    assert mock_handle.await_count == 1
    assert hass.states.get("light.living_room").state == "on"
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_REQUEST_TIMEOUT


async def test_invalid_follow_up_after_tool_execution_never_retries(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    """Test invalid follow-up tool calls after execution never retry or re-execute."""
    entry = await _create_entry(hass)
    invalid_follow_up = ToolCall(
        id="call_bad",
        name="HassTurnOn",
        arguments={"name": "Living Room", "unexpected": True},
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
                ChatCompletionResult(
                    content=None,
                    tool_calls=[invalid_follow_up],
                ),
            ]
        ),
    ) as mock_chat, patch.object(
        intent,
        "async_handle",
        wraps=intent.async_handle,
    ) as mock_handle:
        result = await _converse(hass, entry, "Turn on the living room light")

    assert mock_chat.await_count == 2
    assert mock_handle.await_count == 1
    assert hass.states.get("light.living_room").state == "on"
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(result) == ERROR_ACTION_FAILED

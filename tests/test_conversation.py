"""Tests for the SaySo conversation entity."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import intent

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from custom_components.sayso.const import (
    DOMAIN,
    ERROR_EMPTY_RESPONSE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_REQUEST_TIMEOUT,
    ERROR_TOOL_CALLS_UNSUPPORTED,
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
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass,
        text,
        conversation_id,
        Context(),
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


async def test_tool_calls_fail_closed(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test tool calls are rejected in the plain-text milestone."""
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
        invalid_result = await _converse(hass, entry, "Turn on the light")

    assert invalid_result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(invalid_result) == ERROR_MODEL_UNAVAILABLE

    from custom_components.sayso.client import ToolCall

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="HassTurnOn", arguments={"name": "light"})
                ],
            )
        ),
    ):
        tool_result = await _converse(hass, entry, "Turn on the light")

    assert tool_result.response.response_type == intent.IntentResponseType.ERROR
    assert _speech(tool_result) == ERROR_TOOL_CALLS_UNSUPPORTED


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

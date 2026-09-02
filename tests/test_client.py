"""Tests for the llama.cpp HTTP client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import ClientTimeout
from homeassistant.core import HomeAssistant

from custom_components.sayso.client import (
    ChatCompletionResult,
    LlamaCppClient,
    ToolCall,
    normalize_base_url,
)
from custom_components.sayso.const import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from custom_components.sayso.exceptions import (
    SaySoAuthError,
    SaySoConnectionError,
    SaySoHttpError,
    SaySoInvalidResponseError,
    SaySoTimeoutError,
)


def test_normalize_base_url_appends_v1() -> None:
    assert normalize_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080/v1"


def test_normalize_base_url_strips_trailing_slash() -> None:
    assert normalize_base_url("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080/v1"


def test_normalize_base_url_deduplicates_v1() -> None:
    assert (
        normalize_base_url("http://127.0.0.1:8080/v1/v1")
        == "http://127.0.0.1:8080/v1"
    )


def test_default_constants() -> None:
    assert DEFAULT_TEMPERATURE == 0
    assert DEFAULT_MAX_OUTPUT_TOKENS == 160
    assert DEFAULT_MAX_TOOL_ITERATIONS == 3
    assert DEFAULT_TIMEOUT == 30


async def test_successful_text_response(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(
        json_body={
            "choices": [{"message": {"content": "The lamp is on.", "role": "assistant"}}]
        }
    )

    result = await llama_client.chat_completion(
        [{"role": "user", "content": "Is the lamp on?"}],
        model="test-model",
    )

    assert result == ChatCompletionResult(content="The lamp is on.", tool_calls=[])


async def test_tool_calls_with_json_arguments(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(
        json_body={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "turn_on",
                                    "arguments": '{"entity_id": "light.kitchen"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "activate_scene",
                                    "arguments": {"entity_id": "scene.movie"},
                                },
                            },
                        ],
                    }
                }
            ]
        }
    )

    result = await llama_client.chat_completion(
        [{"role": "user", "content": "Turn on the kitchen light and movie mode."}],
        model="test-model",
    )

    assert result.content is None
    assert result.tool_calls == [
        ToolCall(
            id="call_1",
            name="turn_on",
            arguments={"entity_id": "light.kitchen"},
        ),
        ToolCall(
            id="call_2",
            name="activate_scene",
            arguments={"entity_id": "scene.movie"},
        ),
    ]


async def test_request_includes_bearer_and_payload(
    llama_client: LlamaCppClient,
    mock_session: aiohttp.ClientSession,
    configure_post: Any,
) -> None:
    configure_post(
        json_body={
            "choices": [{"message": {"content": "Done.", "role": "assistant"}}]
        }
    )
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "turn_on", "parameters": {}}}]

    await llama_client.chat_completion(
        messages,
        model="test-model",
        tools=tools,
        temperature=0.2,
        max_tokens=64,
    )

    mock_session.post.assert_called_once()
    call_kwargs = mock_session.post.call_args.kwargs
    assert call_kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert call_kwargs["json"] == {
        "model": "test-model",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 64,
        "tools": tools,
    }
    assert call_kwargs["timeout"] == ClientTimeout(total=30)


async def test_no_api_key_omits_authorization_header(
    mock_session: aiohttp.ClientSession,
    configure_post: Any,
) -> None:
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1")
    configure_post(
        json_body={
            "choices": [{"message": {"content": "Done.", "role": "assistant"}}]
        }
    )

    await client.chat_completion([{"role": "user", "content": "Hello"}], model="m")

    headers = mock_session.post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


async def test_auth_failure(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(status=401)

    with pytest.raises(SaySoAuthError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_connection_failure(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(side_effect=aiohttp.ClientConnectionError("connection refused"))

    with pytest.raises(SaySoConnectionError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_timeout(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(side_effect=aiohttp.ServerTimeoutError())

    with pytest.raises(SaySoTimeoutError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_http_error(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(status=500)

    with pytest.raises(SaySoHttpError) as exc_info:
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )

    assert exc_info.value.status == 500


async def test_invalid_json(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(status=200, text_body="not json")

    with pytest.raises(SaySoInvalidResponseError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_missing_choices(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(json_body={"choices": []})

    with pytest.raises(SaySoInvalidResponseError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_empty_content_and_tool_calls(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(
        json_body={"choices": [{"message": {"role": "assistant", "content": None}}]}
    )

    with pytest.raises(SaySoInvalidResponseError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_llama_error_payload(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(json_body={"error": {"message": "model not found"}})

    with pytest.raises(SaySoInvalidResponseError, match="model not found"):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )


async def test_no_retry_after_terminal_transport_failure(
    llama_client: LlamaCppClient,
    configure_post: Any,
) -> None:
    configure_post(side_effect=aiohttp.ClientConnectionError("connection refused"))

    with pytest.raises(SaySoConnectionError):
        await llama_client.chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="test-model",
        )

    assert llama_client._session.post.call_count == 1


async def test_list_models_uses_http_get(
    mock_session: aiohttp.ClientSession,
) -> None:
    """Test list_models uses the models endpoint over GET."""
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"data": [{"id": "test-model"}]})
    context_manager = AsyncMock()
    context_manager.__aenter__ = AsyncMock(return_value=response)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=context_manager)

    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", api_key="test-key")
    models = await client.list_models()

    assert models == ["test-model"]
    mock_session.get.assert_called_once()
    assert (
        mock_session.get.call_args.args[0]
        == "http://127.0.0.1:8080/v1/models"
    )


async def test_from_hass_uses_shared_session(hass: HomeAssistant) -> None:
    shared_session = MagicMock(spec=aiohttp.ClientSession)
    with patch(
        "custom_components.sayso.client.async_get_clientsession",
        return_value=shared_session,
    ) as get_session:
        client = LlamaCppClient.from_hass(
            hass,
            "http://127.0.0.1:8080",
            api_key="secret",
            timeout=15,
        )

    get_session.assert_called_once_with(hass)
    assert client._session is shared_session
    assert client.base_url == "http://127.0.0.1:8080/v1"
    assert client.chat_completions_url == "http://127.0.0.1:8080/v1/chat/completions"

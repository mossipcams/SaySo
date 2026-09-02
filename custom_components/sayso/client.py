"""Async llama.cpp OpenAI-compatible HTTP client."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import ClientError, ClientResponse, ClientTimeout
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from .exceptions import (
    SaySoAuthError,
    SaySoConnectionError,
    SaySoHttpError,
    SaySoInvalidResponseError,
    SaySoTimeoutError,
)

_LOGGER = logging.getLogger(__name__)


def normalize_base_url(base_url: str) -> str:
    """Normalize a llama.cpp base URL and ensure a single /v1 suffix."""
    url = base_url.strip().rstrip("/")
    while url.endswith("/v1/v1"):
        url = url[:-3]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


@dataclass(frozen=True, slots=True)
class ToolCall:
    """An OpenAI-style tool call from llama.cpp."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    """Parsed assistant output from a chat completion."""

    content: str | None
    tool_calls: list[ToolCall]


class LlamaCppClient:
    """Transport-only client for llama.cpp chat completions."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._session = session
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout = timeout

    @classmethod
    def from_hass(
        cls,
        hass: Any,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> LlamaCppClient:
        """Create a client using Home Assistant's shared aiohttp session."""
        return cls(
            async_get_clientsession(hass),
            base_url,
            api_key=api_key,
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        """Normalized llama.cpp base URL."""
        return self._base_url

    @property
    def chat_completions_url(self) -> str:
        """Full URL for chat completions."""
        return f"{self._base_url}{CHAT_COMPLETIONS_PATH}"

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ChatCompletionResult:
        """Request a single chat completion from llama.cpp."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with self._session.post(
                self.chat_completions_url,
                json=payload,
                headers=headers,
                timeout=ClientTimeout(total=self._timeout),
            ) as response:
                return await self._parse_response(response)
        except TimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except aiohttp.ServerTimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except ClientError as err:
            raise SaySoConnectionError("llama.cpp is unreachable") from err

    async def _parse_response(self, response: ClientResponse) -> ChatCompletionResult:
        if response.status in {401, 403}:
            raise SaySoAuthError("llama.cpp rejected the API key")

        if response.status >= 400:
            raise SaySoHttpError(response.status)

        try:
            body = await response.json(content_type=None)
        except (json.JSONDecodeError, aiohttp.ContentTypeError, ValueError) as err:
            raise SaySoInvalidResponseError("llama.cpp returned invalid JSON") from err

        if not isinstance(body, dict):
            raise SaySoInvalidResponseError("llama.cpp returned invalid JSON")

        if "error" in body:
            raise SaySoInvalidResponseError(
                _format_llama_error(body.get("error"))
            )

        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            raise SaySoInvalidResponseError("llama.cpp returned no choices")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise SaySoInvalidResponseError("llama.cpp returned no choices")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise SaySoInvalidResponseError("llama.cpp returned invalid content")

        raw_tool_calls = message.get("tool_calls")
        tool_calls: list[ToolCall] = []
        if raw_tool_calls is not None:
            if not isinstance(raw_tool_calls, list):
                raise SaySoInvalidResponseError("llama.cpp returned invalid tool calls")
            tool_calls = [_parse_tool_call(item) for item in raw_tool_calls]

        if content is None and not tool_calls:
            raise SaySoInvalidResponseError(
                "llama.cpp returned neither content nor tool calls"
            )

        return ChatCompletionResult(content=content, tool_calls=tool_calls)


def _parse_tool_call(raw: Any) -> ToolCall:
    if not isinstance(raw, dict):
        raise SaySoInvalidResponseError("llama.cpp returned invalid tool calls")

    tool_id = raw.get("id")
    function = raw.get("function")
    if not isinstance(tool_id, str) or not tool_id:
        raise SaySoInvalidResponseError("llama.cpp returned invalid tool calls")
    if not isinstance(function, dict):
        raise SaySoInvalidResponseError("llama.cpp returned invalid tool calls")

    name = function.get("name")
    arguments_raw = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise SaySoInvalidResponseError("llama.cpp returned invalid tool calls")

    if isinstance(arguments_raw, dict):
        arguments = arguments_raw
    elif isinstance(arguments_raw, str):
        try:
            parsed = json.loads(arguments_raw)
        except json.JSONDecodeError as err:
            raise SaySoInvalidResponseError(
                "llama.cpp returned invalid tool call arguments"
            ) from err
        if not isinstance(parsed, dict):
            raise SaySoInvalidResponseError(
                "llama.cpp returned invalid tool call arguments"
            )
        arguments = parsed
    else:
        raise SaySoInvalidResponseError("llama.cpp returned invalid tool calls")

    return ToolCall(id=tool_id, name=name, arguments=arguments)


def _format_llama_error(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(error, str) and error:
        return error
    return "llama.cpp returned an error response"

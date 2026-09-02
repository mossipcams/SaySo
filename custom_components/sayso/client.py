"""Async llama.cpp OpenAI-compatible HTTP client."""

from __future__ import annotations

import json
import logging
import time
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
    MODELS_PATH,
)
from .exceptions import (
    SaySoAuthError,
    SaySoConnectionError,
    SaySoHttpError,
    SaySoInvalidResponseError,
    SaySoModelNotFoundError,
    SaySoTimeoutError,
)

def normalize_base_url(base_url: str) -> str:
    """Normalize a llama.cpp base URL and ensure a single /v1 suffix."""
    url = base_url.strip().rstrip("/")
    while url.endswith("/v1/v1"):
        url = url[:-3]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def serialize_chat_completions_payload(payload: dict[str, Any]) -> bytes:
    """Serialize the chat-completions payload using aiohttp-compatible JSON."""
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def build_chat_completions_payload(
    messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Build the production chat-completions payload sent to llama.cpp."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        payload["tools"] = tools
    return payload


def _extract_prompt_tokens(body: dict[str, Any]) -> int | None:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    if isinstance(prompt_tokens, int):
        return prompt_tokens
    return None


_LOGGER = logging.getLogger(__name__)


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
    request_payload: dict[str, Any] | None = None
    request_bytes: int | None = None
    prompt_tokens: int | None = None


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

    @property
    def models_url(self) -> str:
        """Full URL for the models listing endpoint."""
        return f"{self._base_url}{MODELS_PATH}"

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers when an API key is configured."""
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def list_models(self) -> list[str]:
        """Return model identifiers advertised by llama.cpp."""
        try:
            async with self._session.get(
                self.models_url,
                headers=self._auth_headers(),
                timeout=ClientTimeout(total=self._timeout),
            ) as response:
                return await self._parse_models_response(response)
        except TimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except aiohttp.ServerTimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except ClientError as err:
            raise SaySoConnectionError("llama.cpp is unreachable") from err

    async def validate_model(self, model: str) -> None:
        """Ensure the configured model is available on llama.cpp."""
        models = await self.list_models()
        if model not in models:
            raise SaySoModelNotFoundError(f"Model {model!r} is not available")

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
        payload = build_chat_completions_payload(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request_bytes = len(serialize_chat_completions_payload(payload))

        try:
            async with self._session.post(
                self.chat_completions_url,
                json=payload,
                headers=self._auth_headers(),
                timeout=ClientTimeout(total=self._timeout),
            ) as response:
                return await self._parse_response(
                    response,
                    request_payload=payload,
                    request_bytes=request_bytes,
                )
        except TimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except aiohttp.ServerTimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except ClientError as err:
            raise SaySoConnectionError("llama.cpp is unreachable") from err

    async def probe_ttft_ms(self, payload: dict[str, Any]) -> float:
        """Measure time-to-first-token using an eval-only streaming probe."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        started_at = time.perf_counter()
        try:
            async with self._session.post(
                self.chat_completions_url,
                json=stream_payload,
                headers=self._auth_headers(),
                timeout=ClientTimeout(total=self._timeout),
            ) as response:
                if response.status in {401, 403}:
                    raise SaySoAuthError("llama.cpp rejected the API key")
                if response.status >= 400:
                    raise SaySoHttpError(response.status)

                while True:
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if _sse_event_has_generated_token(event):
                        return (time.perf_counter() - started_at) * 1000.0

                raise SaySoInvalidResponseError(
                    "llama.cpp stream ended without a generated token"
                )
        except TimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except aiohttp.ServerTimeoutError as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except ClientError as err:
            raise SaySoConnectionError("llama.cpp is unreachable") from err

    async def _parse_response(
        self,
        response: ClientResponse,
        *,
        request_payload: dict[str, Any] | None = None,
        request_bytes: int | None = None,
    ) -> ChatCompletionResult:
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

        return ChatCompletionResult(
            content=content,
            tool_calls=tool_calls,
            request_payload=request_payload,
            request_bytes=request_bytes,
            prompt_tokens=_extract_prompt_tokens(body),
        )

    async def _parse_models_response(self, response: ClientResponse) -> list[str]:
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

        data = body.get("data")
        if not isinstance(data, list):
            raise SaySoInvalidResponseError("llama.cpp returned invalid models list")

        models: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                raise SaySoInvalidResponseError("llama.cpp returned invalid models list")
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise SaySoInvalidResponseError("llama.cpp returned invalid models list")
            models.append(model_id)

        if not models:
            raise SaySoInvalidResponseError("llama.cpp returned no models")

        return models


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


def _sse_event_has_generated_token(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0]
    if not isinstance(choice, dict):
        return False
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False

    content = delta.get("content")
    if isinstance(content, str) and content:
        return True

    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        name = function.get("name")
        if isinstance(arguments, str) and arguments:
            return True
        if isinstance(name, str) and name:
            return True
    return False

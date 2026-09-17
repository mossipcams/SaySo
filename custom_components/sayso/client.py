"""Async llama.cpp OpenAI-compatible HTTP client."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
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
# Re-exported: the parsing half of this module lives in .completion.
from .completion import (  # noqa: F401
    ChatCompletionResult,
    ToolCall,
    parse_choice_message,
    parse_tool_calls,
    prompt_tokens_of,
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


def _format_llama_error(error: Any) -> str:
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"] or "llama.cpp returned an error response"
    if isinstance(error, str) and error:
        return error
    return "llama.cpp returned an error response"


def _raise_for_status(response: ClientResponse) -> None:
    """Translate llama.cpp's HTTP status into a SaySo error."""
    if response.status in {401, 403}:
        raise SaySoAuthError("llama.cpp rejected the API key")
    if response.status >= 400:
        raise SaySoHttpError(response.status)


async def _read_json_body(response: ClientResponse) -> dict[str, Any]:
    """Decode a llama.cpp JSON envelope, surfacing its own ``error`` field."""
    try:
        body = await response.json(content_type=None)
    except (json.JSONDecodeError, aiohttp.ContentTypeError, ValueError) as err:
        raise SaySoInvalidResponseError("llama.cpp returned invalid JSON") from err
    if not isinstance(body, dict):
        raise SaySoInvalidResponseError("llama.cpp returned invalid JSON")
    if "error" in body:
        raise SaySoInvalidResponseError(_format_llama_error(body.get("error")))
    return body


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
            async_get_clientsession(hass), base_url, api_key=api_key, timeout=timeout
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

    def _request_kwargs(self) -> dict[str, Any]:
        """Auth and timeout applied identically to every request."""
        return {
            "headers": (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            ),
            "timeout": ClientTimeout(total=self._timeout),
        }

    async def _send[T](
        self,
        open_request: Callable[[], Any],
        handle: Callable[[ClientResponse], Awaitable[T]],
    ) -> T:
        """Run one request, translating transport failures into SaySo errors.

        The request is opened inside the ``try`` so a session that fails at
        connect time is reported the same way as one that fails mid-response.
        """
        try:
            async with open_request() as response:
                _raise_for_status(response)
                return await handle(response)
        except (TimeoutError, aiohttp.ServerTimeoutError) as err:
            raise SaySoTimeoutError("llama.cpp request timed out") from err
        except ClientError as err:
            raise SaySoConnectionError("llama.cpp is unreachable") from err

    async def list_models(self) -> list[str]:
        """Return model identifiers advertised by llama.cpp."""
        return await self._send(
            lambda: self._session.get(self.models_url, **self._request_kwargs()),
            _parse_models,
        )

    async def validate_model(self, model: str) -> None:
        """Ensure the configured model is available on llama.cpp."""
        if model not in await self.list_models():
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

        async def parse(response: ClientResponse) -> ChatCompletionResult:
            body = await _read_json_body(response)
            content, message = parse_choice_message(body, "llama.cpp")
            tool_calls = parse_tool_calls(
                message.get("tool_calls"), subject="llama.cpp"
            )
            if content is None and not tool_calls:
                raise SaySoInvalidResponseError(
                    "llama.cpp returned neither content nor tool calls"
                )
            return ChatCompletionResult(
                content=content,
                tool_calls=tool_calls,
                request_payload=payload,
                request_bytes=request_bytes,
                prompt_tokens=prompt_tokens_of(body),
            )

        return await self._send(
            lambda: self._session.post(
                self.chat_completions_url, json=payload, **self._request_kwargs()
            ),
            parse,
        )

    async def probe_ttft_ms(self, payload: dict[str, Any]) -> float:
        """Measure time-to-first-token using an eval-only streaming probe."""
        started_at = time.perf_counter()

        async def first_token(response: ClientResponse) -> float:
            while raw_line := await response.content.readline():
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

        return await self._send(
            lambda: self._session.post(
                self.chat_completions_url,
                json={**payload, "stream": True},
                **self._request_kwargs(),
            ),
            first_token,
        )


async def _parse_models(response: ClientResponse) -> list[str]:
    """Read the ``/v1/models`` listing, rejecting anything unusable."""
    body = await _read_json_body(response)
    data = body.get("data")
    invalid = SaySoInvalidResponseError("llama.cpp returned invalid models list")
    if not isinstance(data, list):
        raise invalid

    models: list[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str) or not model_id:
            raise invalid
        models.append(model_id)

    if not models:
        raise SaySoInvalidResponseError("llama.cpp returned no models")
    return models


def _sse_event_has_generated_token(event: Any) -> bool:
    """Return whether one stream event carries the first real output token."""
    choices = event.get("choices") if isinstance(event, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    delta = choice.get("delta") if isinstance(choice, dict) else None
    if not isinstance(delta, dict):
        return False

    content = delta.get("content")
    if isinstance(content, str) and content:
        return True

    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    return any(
        isinstance(function := tool_call.get("function"), dict)
        and any(
            isinstance(value := function.get(key), str) and value
            for key in ("arguments", "name")
        )
        for tool_call in tool_calls
        if isinstance(tool_call, dict)
    )

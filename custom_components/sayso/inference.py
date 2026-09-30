
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from homeassistant.const import CONF_URL

from .client import ChatCompletionResult, LlamaCppClient
from .completion import (
    extract_tool_calls,
    parse_completion_result,
    parse_completion_result as _parse_embedded_result,
)
from .const import (
    BACKEND_EMBEDDED,
    BACKEND_EXTERNAL,
    CONF_BACKEND,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_N_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MAX_DEFAULT_THREADS,
)
from .exceptions import (
    SaySoInvalidResponseError,
    SaySoModelLoadError,
    SaySoTimeoutError,
)

_LOGGER = logging.getLogger(__name__)


def _messages_for_embedded_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            normalized.append(message)
            continue

        new_message = dict(message)
        new_tool_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                new_tool_calls.append(call)
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                new_tool_calls.append(call)
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    new_call = dict(call)
                    new_call["function"] = {**function, "arguments": parsed}
                    new_tool_calls.append(new_call)
                    continue
            new_tool_calls.append(call)
        new_message["tool_calls"] = new_tool_calls
        normalized.append(new_message)
    return normalized


def default_thread_count() -> int:
    return max(1, min(MAX_DEFAULT_THREADS, os.cpu_count() or 1))


def entry_backend(entry: Any) -> str:
    backend = entry.data.get(CONF_BACKEND)
    if backend in (BACKEND_EMBEDDED, BACKEND_EXTERNAL):
        return backend
    return BACKEND_EXTERNAL if entry.data.get(CONF_URL) else BACKEND_EMBEDDED


class SaySoInferenceEngine(Protocol):

    @property
    def model_name(self) -> str:
        pass

    async def async_start(self) -> None:
        pass

    async def async_shutdown(self) -> None:
        pass

    async def async_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ChatCompletionResult:
        pass


class ExternalEngine:

    def __init__(self, client: LlamaCppClient, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def async_start(self) -> None:
        pass

    async def async_shutdown(self) -> None:
        pass

    async def async_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ChatCompletionResult:
        return await self._client.chat_completion(
            messages,
            model=self._model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class EmbeddedEngine:

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads or default_thread_count()
        self._timeout = timeout
        self._llm: Any | None = None
        self._executor: ThreadPoolExecutor | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def model_name(self) -> str:
        return self._model_path.stem

    def _load(self) -> Any:
        from llama_cpp import Llama

        return Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=False,
        )

    async def async_start(self) -> None:
        if self._llm is not None:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sayso_inference"
        )
        loop = asyncio.get_running_loop()
        try:
            self._llm = await loop.run_in_executor(self._executor, self._load)
        except Exception as err:
            await self.async_shutdown()
            raise SaySoModelLoadError(
                f"Could not load {self._model_path.name}: {err}"
            ) from err
        _LOGGER.info(
            "SaySo loaded %s (n_ctx=%s, threads=%s)",
            self._model_path.name,
            self._n_ctx,
            self._n_threads,
        )

    async def async_shutdown(self) -> None:
        self._llm = None
        if self._executor is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._executor.shutdown(wait=True)
            )
            self._executor = None

    def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        assert self._llm is not None
        kwargs: dict[str, Any] = {
            "messages": _messages_for_embedded_template(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self._llm.create_chat_completion(**kwargs)

    async def async_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ChatCompletionResult:
        if self._llm is None or self._executor is None:
            raise SaySoModelLoadError("The embedded model is not loaded")

        loop = asyncio.get_running_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._complete,
                    messages,
                    tools,
                    temperature,
                    max_tokens,
                ),
                timeout=self._timeout,
            )
        except TimeoutError as err:
            raise SaySoTimeoutError("Local inference timed out") from err
        except SaySoModelLoadError:
            raise
        except Exception as err:
            raise SaySoInvalidResponseError(f"Local inference failed: {err}") from err

        return parse_completion_result(raw)




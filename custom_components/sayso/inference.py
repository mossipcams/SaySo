"""Inference backends for SaySo.

Both backends return the same ``ChatCompletionResult`` and raise the same
``SaySoError`` subclasses, so everything downstream — tool validation,
correction retries, boundary diagnostics, tracing — is identical whichever one
is active.

Two implementations exist: ``EmbeddedEngine`` runs the GGUF in-process on the
CPU, ``ExternalEngine`` keeps the original OpenAI-compatible HTTP path as an
advanced fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from homeassistant.const import CONF_URL

from .client import (
    ChatCompletionResult,
    LlamaCppClient,
    ToolCall,
    parse_choice_message,
    parse_tool_calls,
    prompt_tokens_of,
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
from .lfm_parse import LfmPythonParseError, parse_lfm_python_tool_calls

_LOGGER = logging.getLogger(__name__)

# LFM2 wraps tool calls in these markers. llama-cpp-python ships libllama only,
# not llama.cpp's common/chat.cpp, so the server-side parser that llama-server
# applies with --jinja is not available and SaySo strips them itself.
_TOOL_CALL_START = "<|tool_call_start|>"
_TOOL_CALL_END = "<|tool_call_end|>"

# Prefix for the errors this backend raises, so a trace says which one failed.
_SUBJECT = "Local inference"


def _call_id() -> str:
    """Mint a tool-call id for a backend that does not supply one."""
    return f"call_{uuid.uuid4().hex[:8]}"


def default_thread_count() -> int:
    """Return a thread count that leaves headroom for Home Assistant."""
    return max(1, min(MAX_DEFAULT_THREADS, os.cpu_count() or 1))


def entry_backend(entry: Any) -> str:
    """Return which backend a config entry uses.

    Entries created before embedded inference existed have no stored backend
    but always have a URL, so they keep using the external one without a
    migration step.
    """
    backend = entry.data.get(CONF_BACKEND)
    if backend in (BACKEND_EMBEDDED, BACKEND_EXTERNAL):
        return backend
    return BACKEND_EXTERNAL if entry.data.get(CONF_URL) else BACKEND_EMBEDDED


def extract_tool_calls(content: str) -> tuple[str | None, list[ToolCall]]:
    """Split assistant text into leftover prose and structured tool calls.

    Returns ``(text, [])`` when the model answered in prose. Malformed tool-call
    syntax raises, because a half-understood action must fail closed rather than
    execute something approximate.
    """
    if not content:
        return None, []

    body = content
    if _TOOL_CALL_START in body:
        prefix, _, rest = body.partition(_TOOL_CALL_START)
        body, _, suffix = rest.partition(_TOOL_CALL_END)
        leftover = f"{prefix}{suffix}".strip()
    else:
        stripped = body.strip()
        # A bare bracketed call list is the same payload without the markers.
        if not (stripped.startswith("[") and stripped.endswith("]")):
            return content, []
        body, leftover = stripped, ""

    try:
        parsed = parse_lfm_python_tool_calls(body)
    except LfmPythonParseError as err:
        raise SaySoInvalidResponseError(
            f"Could not parse model tool call {body!r}: {err}"
        ) from err

    calls = [
        ToolCall(
            id=_call_id(),
            name=call["name"],
            arguments=call["arguments"],
        )
        for call in parsed
    ]
    return (leftover or None), calls


class SaySoInferenceEngine(Protocol):
    """One chat-completion backend."""

    @property
    def model_name(self) -> str:
        """Identifier recorded in traces and diagnostics."""

    async def async_start(self) -> None:
        """Acquire whatever the backend needs to serve requests."""

    async def async_shutdown(self) -> None:
        """Release resources. Must be safe to call more than once."""

    async def async_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ChatCompletionResult:
        """Run one completion."""


class ExternalEngine:
    """Advanced fallback: a user-managed OpenAI-compatible llama.cpp server."""

    def __init__(self, client: LlamaCppClient, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        """Model identifier advertised by the external server."""
        return self._model

    async def async_start(self) -> None:
        """Nothing to acquire; the server owns the model."""

    async def async_shutdown(self) -> None:
        """Nothing to release; the shared aiohttp session is Home Assistant's."""

    async def async_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ChatCompletionResult:
        """Delegate to the HTTP client unchanged."""
        return await self._client.chat_completion(
            messages,
            model=self._model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class EmbeddedEngine:
    """Runs the GGUF in this process, off the event loop.

    The model stays resident between turns — reloading a 230M model per request
    would cost more than the inference does.
    """

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
        # ponytail: one worker, not a pool. llama.cpp's Llama object is not
        # thread-safe and a 230M model already saturates the cores it is given,
        # so concurrent turns would contend rather than parallelise. A dedicated
        # executor also keeps a slow inference off Home Assistant's shared pool.
        self._executor: ThreadPoolExecutor | None = None

    @property
    def model_path(self) -> Path:
        """Path of the loaded GGUF."""
        return self._model_path

    @property
    def model_name(self) -> str:
        """GGUF filename without its extension."""
        return self._model_path.stem

    def _load(self) -> Any:
        """Construct the Llama object. Runs in the inference worker."""
        from llama_cpp import Llama

        return Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=False,
        )

    async def async_start(self) -> None:
        """Load the model into memory and keep it there."""
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
        """Free the model and stop the worker."""
        self._llm = None
        if self._executor is not None:
            # wait=True: a running inference holds a pointer into model memory,
            # so it must finish before the Llama object is collected.
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
        """Run one completion. Runs in the inference worker."""
        assert self._llm is not None
        kwargs: dict[str, Any] = {
            "messages": messages,
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
        """Run one completion against the resident model."""
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
            # llama.cpp cannot be interrupted mid-token, so the worker finishes
            # the turn we abandoned. The single worker then backpressures the
            # next request rather than piling up concurrent inferences.
            raise SaySoTimeoutError("Local inference timed out") from err
        except SaySoModelLoadError:
            raise
        except Exception as err:
            raise SaySoInvalidResponseError(f"Local inference failed: {err}") from err

        return _parse_embedded_result(raw)


def _parse_embedded_result(raw: Any) -> ChatCompletionResult:
    """Convert llama-cpp-python output into SaySo's transport-neutral result."""
    if not isinstance(raw, dict):
        raise SaySoInvalidResponseError("Local inference returned no result")

    content, message = parse_choice_message(raw, _SUBJECT)

    # Honour structured tool_calls when a handler produced them, and fall back
    # to parsing LFM2's native text format otherwise. llama-cpp-python omits
    # call ids, so one is minted here rather than failing the turn.
    tool_calls = parse_tool_calls(
        message.get("tool_calls"), subject=_SUBJECT, mint_id=_call_id
    )
    if tool_calls:
        text: str | None = content
    else:
        text, tool_calls = extract_tool_calls(content or "")

    if text is None and not tool_calls:
        raise SaySoInvalidResponseError(
            "Local inference returned neither content nor tool calls"
        )

    return ChatCompletionResult(
        content=text,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens_of(raw),
    )

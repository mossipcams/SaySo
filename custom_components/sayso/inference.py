
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from homeassistant.const import CONF_URL

from .area_context import tools_before_area_messages
from .client import ChatCompletionResult, LlamaCppClient
from .completion import (
    extract_tool_calls,
    parse_completion_result,
    parse_completion_result as _parse_embedded_result,
    prefix_hash,
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


MAX_PREFIX_CHECKPOINTS = 4


def _last_user_turn_start(tokens: list[int], marker: list[int]) -> int | None:
    width = len(marker)
    for start in range(len(tokens) - width, -1, -1):
        if tokens[start : start + width] == marker:
            return start
    return None


def _shared_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _checkpoint_boundaries(
    prompt: list[int], area_marker: list[int], user_marker: list[int]
) -> list[int]:
    boundaries: set[int] = set()
    for marker in (area_marker, user_marker):
        start = _last_user_turn_start(prompt, marker)
        if start is not None and start > 0:
            boundaries.add(start)
    return sorted(boundaries)


def _longest_checkpoint(
    checkpoints: list[list[int]], prompt: list[int], limit: int
) -> int | None:
    best: int | None = None
    for index, tokens in enumerate(checkpoints):
        if len(tokens) > limit or prompt[: len(tokens)] != tokens:
            continue
        if best is None or len(tokens) > len(checkpoints[best]):
            best = index
    return best


def _checkpointing_llama_class() -> type:
    import ctypes

    import llama_cpp
    from llama_cpp import Llama

    flags = llama_cpp.LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY

    class CheckpointingLlama(Llama):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._area_context_marker = self.tokenize(
                b"Area context:\n", add_bos=False, special=True
            )
            self._user_turn_marker = self.tokenize(
                b"<|im_start|>user", add_bos=False, special=True
            )
            self._checkpoints: list[tuple[list[int], bytes]] = []
            self.prefill_evaluated = 0
            self._prefilling = False

        def eval(self, tokens: Any) -> None:
            if self._prefilling:
                self.prefill_evaluated += len(tokens)
            super().eval(tokens)

        def sample(self, *args: Any, **kwargs: Any) -> Any:
            self._prefilling = False
            return super().sample(*args, **kwargs)

        def _save_checkpoint(self, tokens: list[int]) -> None:
            size = llama_cpp.llama_state_seq_get_size_ext(self._ctx.ctx, 0, flags)
            buffer = (ctypes.c_uint8 * size)()
            written = llama_cpp.llama_state_seq_get_data_ext(
                self._ctx.ctx, buffer, size, 0, flags
            )
            if len(self._checkpoints) >= MAX_PREFIX_CHECKPOINTS:
                self._checkpoints.pop(0)
            self._checkpoints.append((tokens, bytes(buffer[:written])))

        def _restore_checkpoint(self, index: int) -> bool:
            tokens, data = self._checkpoints[index]
            buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            read = llama_cpp.llama_state_seq_set_data_ext(
                self._ctx.ctx, buffer, len(data), 0, flags
            )
            if read != len(data) or not self._ctx.kv_cache_seq_rm(0, len(tokens), -1):
                return False
            self.n_tokens = len(tokens)
            self._requires_eval = True
            self._checkpoints.append(self._checkpoints.pop(index))
            return True

        def generate(self, tokens: Any, *args: Any, **kwargs: Any) -> Any:
            self.prefill_evaluated = 0
            self._prefilling = True
            prompt = list(tokens)
            boundaries = _checkpoint_boundaries(
                prompt, self._area_context_marker, self._user_turn_marker
            )
            if (
                not boundaries
                or not kwargs.get("reset", True)
                or not (self._is_recurrent or self._is_hybrid)
            ):
                return super().generate(prompt, *args, **kwargs)

            live = self.input_ids[: self.n_tokens].tolist()
            shared = _shared_prefix_len(live, prompt)
            if self.n_tokens > 0 and shared == self.n_tokens:
                return super().generate(prompt, *args, **kwargs)

            index = _longest_checkpoint(
                [tokens for tokens, _ in self._checkpoints],
                prompt,
                min(shared, boundaries[-1]),
            )
            if index is None or not self._restore_checkpoint(index):
                self.reset()
            for boundary in boundaries:
                if boundary < self.n_tokens:
                    continue
                if self.n_tokens < boundary:
                    self.eval(prompt[self.n_tokens : boundary])
                prefix = prompt[:boundary]
                if prefix not in [tokens for tokens, _ in self._checkpoints]:
                    self._save_checkpoint(prefix)
            return super().generate(prompt, *args, **kwargs)

    return CheckpointingLlama


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
        return _checkpointing_llama_class()(
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
        raw = self._llm.create_chat_completion(
            messages=_messages_for_embedded_template(
                tools_before_area_messages(messages, tools)
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prompt_tokens = prompt_tokens_of(raw)
        reused = None
        if prompt_tokens is not None:
            reused = prompt_tokens - self._llm.prefill_evaluated
        raw["sayso_cache"] = {
            "prefix_hash": prefix_hash(messages, tools),
            "reused_tokens": reused,
        }
        return raw

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




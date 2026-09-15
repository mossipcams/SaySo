"""Tests for the SaySo inference backends."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.sayso.client import ChatCompletionResult
from custom_components.sayso.const import BACKEND_EMBEDDED, BACKEND_EXTERNAL
from custom_components.sayso.exceptions import (
    SaySoInvalidResponseError,
    SaySoModelLoadError,
    SaySoTimeoutError,
)
from custom_components.sayso.inference import (
    EmbeddedEngine,
    ExternalEngine,
    _parse_embedded_result,
    default_thread_count,
    entry_backend,
    extract_tool_calls,
)


def _completion(message: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"choices": [{"message": message}], **extra}


class TestExtractToolCalls:
    """LFM2 emits tool calls as text; SaySo must recover them itself."""

    def test_marker_wrapped_call(self) -> None:
        text, calls = extract_tool_calls(
            "<|tool_call_start|>[HassTurnOn(name='kitchen lights')]<|tool_call_end|>"
        )
        assert text is None
        assert len(calls) == 1
        assert calls[0].name == "HassTurnOn"
        assert calls[0].arguments == {"name": "kitchen lights"}

    def test_bare_bracketed_call(self) -> None:
        _text, calls = extract_tool_calls("[HassTurnOff(name='lamp')]")
        assert [call.name for call in calls] == ["HassTurnOff"]

    def test_plain_prose_is_not_a_tool_call(self) -> None:
        text, calls = extract_tool_calls("The kitchen light is on.")
        assert text == "The kitchen light is on."
        assert calls == []

    def test_apostrophe_names_survive(self) -> None:
        """The bug that llama-server's structured tool_calls hits."""
        _text, calls = extract_tool_calls("[HassTurnOn(name='O'Malley's lamp')]")
        assert calls[0].arguments["name"] == "O'Malley's lamp"

    def test_double_quoted_arguments(self) -> None:
        """What LFM2.5 actually emits at inference time."""
        _text, calls = extract_tool_calls('[HassTurnOn(name="kitchen lights")]')
        assert calls[0].arguments == {"name": "kitchen lights"}

    def test_double_quoted_name_with_apostrophe(self) -> None:
        _text, calls = extract_tool_calls("""[HassTurnOn(name="O'Malley's lamp")]""")
        assert calls[0].arguments["name"] == "O'Malley's lamp"

    def test_mixed_quoting_and_literals(self) -> None:
        _text, calls = extract_tool_calls(
            '[HassSetPosition(name="blind", area=\'hall\', open=True, extra=None)]'
        )
        assert calls[0].arguments == {
            "name": "blind",
            "area": "hall",
            "open": True,
            "extra": None,
        }

    def test_double_quoted_list(self) -> None:
        _text, calls = extract_tool_calls(
            '[HassTurnOn(name="tv", domain=["media_player", "light"])]'
        )
        assert calls[0].arguments["domain"] == ["media_player", "light"]

    def test_multiple_calls(self) -> None:
        _text, calls = extract_tool_calls(
            "[HassTurnOn(name='a'), HassTurnOff(name='b')]"
        )
        assert [call.name for call in calls] == ["HassTurnOn", "HassTurnOff"]

    def test_list_argument(self) -> None:
        _text, calls = extract_tool_calls(
            "[HassTurnOn(name='tv', domain=['media_player'])]"
        )
        assert calls[0].arguments["domain"] == ["media_player"]

    def test_call_ids_are_unique(self) -> None:
        _text, calls = extract_tool_calls(
            "[HassTurnOn(name='a'), HassTurnOff(name='b')]"
        )
        assert calls[0].id != calls[1].id

    def test_malformed_call_fails_closed(self) -> None:
        """A half-understood action must never reach Home Assistant."""
        with pytest.raises(SaySoInvalidResponseError):
            extract_tool_calls("<|tool_call_start|>[HassTurnOn(name=<|tool_call_end|>")

    def test_empty_content(self) -> None:
        assert extract_tool_calls("") == (None, [])


class TestParseEmbeddedResult:
    """llama-cpp-python output maps onto the shared ChatCompletionResult."""

    def test_text_response(self) -> None:
        result = _parse_embedded_result(
            _completion({"role": "assistant", "content": "Done."})
        )
        assert isinstance(result, ChatCompletionResult)
        assert result.content == "Done."
        assert result.tool_calls == []

    def test_text_tool_call(self) -> None:
        result = _parse_embedded_result(
            _completion({"content": "[HassTurnOn(name='lamp')]"})
        )
        assert result.tool_calls[0].name == "HassTurnOn"

    def test_structured_tool_calls_are_preferred(self) -> None:
        result = _parse_embedded_result(
            _completion(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "HassTurnOn",
                                "arguments": '{"name": "lamp"}',
                            },
                        }
                    ],
                }
            )
        )
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].arguments == {"name": "lamp"}

    def test_prompt_tokens_recorded(self) -> None:
        result = _parse_embedded_result(
            _completion({"content": "hi"}, usage={"prompt_tokens": 42})
        )
        assert result.prompt_tokens == 42

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            {},
            {"choices": []},
            {"choices": [{}]},
            _completion({"content": ""}),
            _completion({"content": 5}),
        ],
    )
    def test_unusable_output_raises(self, raw: Any) -> None:
        with pytest.raises(SaySoInvalidResponseError):
            _parse_embedded_result(raw)

    def test_invalid_structured_arguments_raise(self) -> None:
        with pytest.raises(SaySoInvalidResponseError):
            _parse_embedded_result(
                _completion(
                    {
                        "content": None,
                        "tool_calls": [
                            {"function": {"name": "X", "arguments": "not json"}}
                        ],
                    }
                )
            )


class TestEmbeddedEngine:
    """Lifecycle and error mapping, without loading a real model."""

    async def test_completion_before_start_raises(self) -> None:
        engine = EmbeddedEngine(Path("/nonexistent.gguf"))
        with pytest.raises(SaySoModelLoadError):
            await engine.async_chat_completion([{"role": "user", "content": "hi"}])

    async def test_load_failure_is_wrapped_and_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/nonexistent.gguf"))
        monkeypatch.setattr(
            engine, "_load", MagicMock(side_effect=OSError("no such file"))
        )
        with pytest.raises(SaySoModelLoadError):
            await engine.async_start()
        # The worker must not survive a failed load.
        assert engine._executor is None

    async def test_shutdown_is_idempotent(self) -> None:
        engine = EmbeddedEngine(Path("/nonexistent.gguf"))
        await engine.async_shutdown()
        await engine.async_shutdown()

    async def test_inference_error_maps_to_invalid_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = MagicMock()
        llm.create_chat_completion.side_effect = RuntimeError("ggml assert")
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        try:
            with pytest.raises(SaySoInvalidResponseError):
                await engine.async_chat_completion([{"role": "user", "content": "hi"}])
        finally:
            await engine.async_shutdown()

    async def test_timeout_maps_to_timeout_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"), timeout=0.05)
        llm = MagicMock()

        def _slow(**_kwargs: Any) -> dict[str, Any]:
            import time

            time.sleep(1.0)
            return _completion({"content": "too late"})

        llm.create_chat_completion.side_effect = _slow
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        try:
            with pytest.raises(SaySoTimeoutError):
                await engine.async_chat_completion([{"role": "user", "content": "hi"}])
        finally:
            await engine.async_shutdown()

    async def test_successful_completion_round_trip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = MagicMock()
        llm.create_chat_completion.return_value = _completion(
            {"content": "<|tool_call_start|>[HassTurnOn(name='lamp')]<|tool_call_end|>"}
        )
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        try:
            result = await engine.async_chat_completion(
                [{"role": "user", "content": "turn on the lamp"}],
                tools=[{"type": "function", "function": {"name": "HassTurnOn"}}],
            )
        finally:
            await engine.async_shutdown()
        assert result.tool_calls[0].arguments == {"name": "lamp"}
        # Tools must reach llama.cpp, or the model cannot call anything.
        assert llm.create_chat_completion.call_args.kwargs["tools"]

    async def test_inference_runs_off_the_event_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        seen: dict[str, Any] = {}
        llm = MagicMock()

        def _record(**_kwargs: Any) -> dict[str, Any]:
            import threading

            seen["thread"] = threading.current_thread().name
            return _completion({"content": "ok"})

        llm.create_chat_completion.side_effect = _record
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        try:
            await engine.async_chat_completion([{"role": "user", "content": "hi"}])
        finally:
            await engine.async_shutdown()
        assert seen["thread"].startswith("sayso_inference")

    def test_model_name_is_the_file_stem(self) -> None:
        assert EmbeddedEngine(Path("/m/LFM2.5-230M-Q8_0.gguf")).model_name == (
            "LFM2.5-230M-Q8_0"
        )

    def test_default_threads_leave_headroom(self) -> None:
        assert 1 <= default_thread_count() <= 4


class TestExternalEngine:
    """The fallback backend still delegates to the HTTP client."""

    async def test_delegates_to_client(self) -> None:
        client = MagicMock()
        expected = ChatCompletionResult(content="hi", tool_calls=[])
        client.chat_completion = AsyncMock(return_value=expected)
        engine = ExternalEngine(client, "test-model")

        await engine.async_start()
        result = await engine.async_chat_completion(
            [{"role": "user", "content": "hi"}], tools=None, temperature=0, max_tokens=10
        )
        await engine.async_shutdown()

        assert result is expected
        assert client.chat_completion.call_args.kwargs["model"] == "test-model"

    def test_model_name(self) -> None:
        assert ExternalEngine(MagicMock(), "abc").model_name == "abc"


class TestModelStore:
    """Weights are never loaded unverified."""

    async def test_existing_file_with_matching_checksum_is_reused(
        self, hass: Any
    ) -> None:
        from custom_components.sayso.model_store import async_ensure_model, models_dir

        directory = models_dir(hass)
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "m.gguf"
        model.write_bytes(b"weights")
        digest = hashlib.sha256(b"weights").hexdigest()

        path = await async_ensure_model(
            hass, url="http://example.invalid/m.gguf", filename="m.gguf", sha256=digest
        )
        assert path == model

    async def test_corrupt_existing_file_is_removed_and_raises(
        self, hass: Any
    ) -> None:
        from custom_components.sayso.model_store import async_ensure_model, models_dir

        directory = models_dir(hass)
        directory.mkdir(parents=True, exist_ok=True)
        model = directory / "m.gguf"
        model.write_bytes(b"corrupt")

        with pytest.raises(SaySoModelLoadError):
            await async_ensure_model(
                hass,
                url="http://example.invalid/m.gguf",
                filename="m.gguf",
                sha256=hashlib.sha256(b"weights").hexdigest(),
            )
        assert not model.exists()

    async def test_existing_file_without_checksum_is_reused(self, hass: Any) -> None:
        from custom_components.sayso.model_store import async_ensure_model, models_dir

        directory = models_dir(hass)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "m.gguf").write_bytes(b"anything")

        path = await async_ensure_model(
            hass, url="http://example.invalid/m.gguf", filename="m.gguf"
        )
        assert path.name == "m.gguf"


class TestEntryBackend:
    """Entries created before embedded inference existed keep working."""

    def test_explicit_embedded(self) -> None:
        entry = SimpleNamespace(data={"backend": BACKEND_EMBEDDED})
        assert entry_backend(entry) == BACKEND_EMBEDDED

    def test_explicit_external(self) -> None:
        entry = SimpleNamespace(data={"backend": BACKEND_EXTERNAL})
        assert entry_backend(entry) == BACKEND_EXTERNAL

    def test_legacy_entry_with_url_stays_external(self) -> None:
        entry = SimpleNamespace(data={"url": "http://127.0.0.1:8080/v1"})
        assert entry_backend(entry) == BACKEND_EXTERNAL

    def test_entry_without_url_is_embedded(self) -> None:
        assert entry_backend(SimpleNamespace(data={})) == BACKEND_EMBEDDED


from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.sayso.area_context import tools_before_area_messages
from custom_components.sayso.client import ChatCompletionResult
from custom_components.sayso.completion import prefix_hash
from custom_components.sayso.const import BACKEND_EMBEDDED, BACKEND_EXTERNAL
from custom_components.sayso.exceptions import (
    SaySoInvalidResponseError,
    SaySoModelLoadError,
    SaySoTimeoutError,
)
from custom_components.sayso.inference import (
    EmbeddedEngine,
    ExternalEngine,
    _checkpoint_boundaries,
    _checkpointing_llama_class,
    _last_user_turn_start,
    _longest_checkpoint,
    _messages_for_embedded_template,
    _parse_embedded_result,
    _shared_prefix_len,
    default_thread_count,
    entry_backend,
    extract_tool_calls,
)


def _completion(message: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"choices": [{"message": message}], **extra}


class TestExtractToolCalls:

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
        _text, calls = extract_tool_calls("[HassTurnOn(name='O'Malley's lamp')]")
        assert calls[0].arguments["name"] == "O'Malley's lamp"

    def test_double_quoted_arguments(self) -> None:
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
        with pytest.raises(SaySoInvalidResponseError):
            extract_tool_calls("<|tool_call_start|>[HassTurnOn(name=<|tool_call_end|>")

    def test_empty_content(self) -> None:
        assert extract_tool_calls("") == (None, [])


class TestParseEmbeddedResult:

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

    def test_cache_trace_recorded(self) -> None:
        result = _parse_embedded_result(
            _completion(
                {"content": "hi"},
                sayso_cache={"prefix_hash": "abc123def456", "reused_tokens": 40},
            )
        )
        assert result.prefix_hash == "abc123def456"
        assert result.reused_tokens == 40

    @pytest.mark.parametrize(
        ("cache", "prefix"),
        [
            ("not a dict", None),
            ({"prefix_hash": 5, "reused_tokens": True}, None),
            ({"prefix_hash": "abc", "reused_tokens": 1.5}, "abc"),
        ],
    )
    def test_malformed_cache_trace_ignored(
        self, cache: Any, prefix: str | None
    ) -> None:
        result = _parse_embedded_result(
            _completion({"content": "hi"}, sayso_cache=cache)
        )
        assert result.prefix_hash == prefix
        assert result.reused_tokens is None

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
                [
                    {"role": "system", "content": "SaySo.\nArea context:\nKitchen"},
                    {"role": "user", "content": "turn on the lamp"},
                ],
                tools=[{"type": "function", "function": {"name": "HassTurnOn"}}],
            )
        finally:
            await engine.async_shutdown()
        assert result.tool_calls[0].arguments == {"name": "lamp"}
        kwargs = llm.create_chat_completion.call_args.kwargs
        assert "tools" not in kwargs
        system_content = kwargs["messages"][0]["content"]
        assert "List of tools: [" in system_content
        assert system_content.index("List of tools: [") < system_content.index(
            "Area context:"
        )

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

    async def test_tool_call_json_strings_converted_for_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = MagicMock()
        llm.create_chat_completion.return_value = _completion({"content": "ok"})
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        messages = [
            {"role": "user", "content": "turn on the lamp"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": '{"name": "lamp"}',
                        },
                    }
                ],
            },
        ]
        try:
            await engine.async_chat_completion(messages)
        finally:
            await engine.async_shutdown()

        sent = llm.create_chat_completion.call_args.kwargs["messages"]
        assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"name": "lamp"}

    async def test_tool_call_dict_arguments_pass_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = MagicMock()
        llm.create_chat_completion.return_value = _completion({"content": "ok"})
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": {"name": "lamp"},
                        },
                    }
                ],
            }
        ]
        try:
            await engine.async_chat_completion(messages)
        finally:
            await engine.async_shutdown()

        sent = llm.create_chat_completion.call_args.kwargs["messages"]
        assert sent[0]["tool_calls"][0]["function"]["arguments"] == {"name": "lamp"}

    async def test_invalid_tool_call_json_strings_are_not_guessed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = MagicMock()
        llm.create_chat_completion.return_value = _completion({"content": "ok"})
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        bad_args = "not json"
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "HassTurnOn", "arguments": bad_args},
                    }
                ],
            }
        ]
        try:
            await engine.async_chat_completion(messages)
        finally:
            await engine.async_shutdown()

        sent = llm.create_chat_completion.call_args.kwargs["messages"]
        assert sent[0]["tool_calls"][0]["function"]["arguments"] == bad_args


class TestMessagesForEmbeddedTemplate:

    def test_json_string_arguments_become_dicts(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": '{"name": "lamp"}',
                        }
                    }
                ],
            }
        ]
        normalized = _messages_for_embedded_template(messages)
        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "name": "lamp"
        }
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
            '{"name": "lamp"}'
        )

    def test_dict_arguments_unchanged(self) -> None:
        args = {"name": "lamp"}
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "HassTurnOn", "arguments": args}}],
            }
        ]
        normalized = _messages_for_embedded_template(messages)
        assert normalized[0]["tool_calls"][0]["function"]["arguments"] is args

    def test_invalid_json_left_as_string(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "HassTurnOn", "arguments": "not json"}}
                ],
            }
        ]
        normalized = _messages_for_embedded_template(messages)
        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == "not json"


class TestExternalEngine:

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

    async def test_messages_keep_json_string_tool_arguments(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value=ChatCompletionResult(content="ok", tool_calls=[])
        )
        engine = ExternalEngine(client, "test-model")
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": '{"name": "lamp"}',
                        },
                    }
                ],
            }
        ]

        await engine.async_chat_completion(messages)

        sent = client.chat_completion.call_args.args[0]
        assert sent[0]["tool_calls"][0]["function"]["arguments"] == '{"name": "lamp"}'


class TestModelStore:

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


class TestEmbeddedPrefixCheckpointDecision:
    def test_boundary_shared_prefix_and_longest_checkpoint(self) -> None:
        marker = [9, 4]
        prompt = [1, 2, 9, 4, 7, 9, 4, 8, 9, 4, 5]
        boundary = _last_user_turn_start(prompt, marker)
        assert boundary == 8
        assert _last_user_turn_start([1, 2, 3], marker) is None
        assert _last_user_turn_start([9, 4], marker) == 0

        assert _shared_prefix_len(prompt[:boundary] + [0], prompt) == boundary
        assert _shared_prefix_len([1, 3], prompt) == 1

        checkpoints = [prompt[:2], prompt[:8], [1, 3], prompt[:5]]
        assert _longest_checkpoint(checkpoints, prompt, 8) == 1
        assert _longest_checkpoint(checkpoints, prompt, 7) == 3
        assert _longest_checkpoint(checkpoints, prompt, 1) is None
        assert _longest_checkpoint([], prompt, 8) is None

        area_marker = [3]
        assert _checkpoint_boundaries(prompt, area_marker, marker) == [8]
        area_prompt = [1, 2, 3, 7, 9, 4, 8]
        assert _checkpoint_boundaries(area_prompt, area_marker, marker) == [2, 4]
        assert _checkpoint_boundaries([9, 4], marker, marker) == []
        assert _checkpoint_boundaries([1, 2], area_marker, marker) == []
        assert _checkpoint_boundaries([1, 3, 7], area_marker, marker) == [1]
        repeated = [1, 3, 7, 9, 4, 8, 3, 6, 9, 4, 5]
        assert _checkpoint_boundaries(repeated, area_marker, marker) == [6, 8]

        checkpoints = [area_prompt[:2], area_prompt[:4]]
        changed_area = [1, 2, 3, 6, 9, 4, 8]
        boundaries = _checkpoint_boundaries(changed_area, area_marker, marker)
        shared = _shared_prefix_len(area_prompt, changed_area)
        assert _longest_checkpoint(
            checkpoints, changed_area, min(shared, boundaries[-1])
        ) == 0
        assert _longest_checkpoint(checkpoints, area_prompt, boundaries[-1]) == 1


_TURN_ON_TOOL = {
    "type": "function",
    "function": {
        "name": "HassTurnOn",
        "parameters": {"type": "object", "properties": {}},
    },
}
_TURN_OFF_TOOL = {
    "type": "function",
    "function": {
        "name": "HassTurnOff",
        "parameters": {"type": "object", "properties": {}},
    },
}
_AREA_BLOCK = "\n".join(
    [
        "Area context:",
        "satellite_area: kitchen",
        "target_area: kitchen",
        "target_area_source: satellite",
    ]
)


class TestToolsBeforeArea:

    def test_tool_list_moves_before_area_block(self) -> None:
        messages = [
            {"role": "system", "content": f"You are SaySo.\n{_AREA_BLOCK}"},
            {"role": "user", "content": "turn on the lamp"},
        ]
        tools = [_TURN_ON_TOOL, _TURN_OFF_TOOL]

        laid_out = tools_before_area_messages(messages, tools)

        expected_list = (
            'List of tools: [{"type": "function", "function": {"name": "HassTurnOn", '
            '"parameters": {"type": "object", "properties": {}}}}, {"type": '
            '"function", "function": {"name": "HassTurnOff", "parameters": '
            '{"type": "object", "properties": {}}}}]'
        )
        assert laid_out == [
            {
                "role": "system",
                "content": f"You are SaySo.\n{expected_list}\n{_AREA_BLOCK}",
            },
            {"role": "user", "content": "turn on the lamp"},
        ]
        assert messages[0]["content"] == f"You are SaySo.\n{_AREA_BLOCK}"

    def test_tool_text_escapes_non_ascii_like_template(self) -> None:
        tool = {
            "type": "function",
            "function": {"name": "HassTurnOn", "description": "café lamp"},
        }
        messages = [{"role": "system", "content": f"Head\n{_AREA_BLOCK}"}]

        content = tools_before_area_messages(messages, [tool])[0]["content"]

        assert content.startswith(
            'Head\nList of tools: [{"type": "function", "function": '
            '{"name": "HassTurnOn", "description": "café lamp"}}]\n'
        )

    def test_unchanged_without_tools_or_area_block(self) -> None:
        with_area = [
            {"role": "system", "content": f"Head\n{_AREA_BLOCK}"},
            {"role": "user", "content": "hi"},
        ]
        assert tools_before_area_messages(with_area, None) is with_area
        assert tools_before_area_messages(with_area, []) is with_area

        without_area = [
            {"role": "system", "content": "Head"},
            {"role": "user", "content": "hi"},
        ]
        assert tools_before_area_messages(without_area, [_TURN_ON_TOOL]) is (
            without_area
        )


class TestEmbeddedToolListLayout:

    async def test_tools_rendered_in_system_prompt_not_passed_to_llama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = MagicMock()
        llm.create_chat_completion.return_value = _completion({"content": "ok"})
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        messages = [
            {"role": "system", "content": f"You are SaySo.\n{_AREA_BLOCK}"},
            {"role": "user", "content": "turn on the lamp"},
        ]
        try:
            await engine.async_chat_completion(messages, tools=[_TURN_ON_TOOL])
        finally:
            await engine.async_shutdown()

        kwargs = llm.create_chat_completion.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        system = kwargs["messages"][0]["content"]
        assert system.index("List of tools: [") < system.index("Area context:")
        assert system.endswith(_AREA_BLOCK)


class _FakeLlama:

    def __init__(self, prompt_tokens: int, prefill_evaluated: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.prefill_evaluated = prefill_evaluated

    def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        return _completion(
            {"content": "ok"}, usage={"prompt_tokens": self.prompt_tokens}
        )


class TestPrefixHash:

    def test_stable_when_area_block_or_user_message_changes(self) -> None:
        tools = [_TURN_ON_TOOL]
        before = [
            {"role": "system", "content": f"You are SaySo.\n{_AREA_BLOCK}"},
            {"role": "user", "content": "turn on the lamp"},
        ]
        other_area = "\n".join(
            [
                "Area context:",
                "satellite_area: hall",
                "target_area: hall",
                "target_area_source: satellite",
            ]
        )
        after = [
            {"role": "system", "content": f"You are SaySo.\n{other_area}"},
            {"role": "user", "content": "turn off the fan"},
        ]
        assert prefix_hash(before, tools) == prefix_hash(after, tools)

    def test_changes_when_tools_change(self) -> None:
        messages = [{"role": "system", "content": f"You are SaySo.\n{_AREA_BLOCK}"}]
        assert prefix_hash(messages, [_TURN_ON_TOOL]) != prefix_hash(
            messages, [_TURN_OFF_TOOL]
        )


class TestEmbeddedCacheTrace:

    async def test_reused_tokens_are_prompt_minus_prefill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = EmbeddedEngine(Path("/model.gguf"))
        llm = _FakeLlama(prompt_tokens=1000, prefill_evaluated=40)
        monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
        await engine.async_start()
        messages = [
            {"role": "system", "content": f"You are SaySo.\n{_AREA_BLOCK}"},
            {"role": "user", "content": "turn on the lamp"},
        ]
        try:
            result = await engine.async_chat_completion(
                messages, tools=[_TURN_ON_TOOL]
            )
        finally:
            await engine.async_shutdown()

        assert result.prompt_tokens == 1000
        assert result.reused_tokens == 960
        assert result.prefix_hash == prefix_hash(messages, [_TURN_ON_TOOL])


class TestCheckpointingLlamaPrefill:

    def test_counts_prompt_evals_but_not_decode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llama_cpp = pytest.importorskip("llama_cpp")

        def fake_eval(self: Any, tokens: list[int]) -> None:
            self.n_tokens += len(tokens)

        def fake_sample(self: Any, **_kwargs: Any) -> int:
            return 7

        def fake_generate(self: Any, tokens: list[int], **_kwargs: Any) -> Any:
            self.eval(tokens[self.n_tokens :])
            yield self.sample()
            self.eval([7])

        monkeypatch.setattr(llama_cpp.Llama, "eval", fake_eval)
        monkeypatch.setattr(llama_cpp.Llama, "sample", fake_sample)
        monkeypatch.setattr(llama_cpp.Llama, "generate", fake_generate)
        monkeypatch.setattr(llama_cpp.Llama, "close", lambda self: None)

        cls = _checkpointing_llama_class()
        llm = cls.__new__(cls)
        llm._area_context_marker = [3]
        llm._user_turn_marker = [9, 4]
        llm._checkpoints = []
        llm._is_recurrent = False
        llm._is_hybrid = False
        llm.n_tokens = 4
        llm.prefill_evaluated = 0
        llm._prefilling = False

        prompt = [1, 2, 3, 5, 6, 9, 4, 7, 8]
        assert list(llm.generate(prompt)) == [7]

        assert llm.prefill_evaluated == 5
        assert len(prompt) - llm.prefill_evaluated == 4

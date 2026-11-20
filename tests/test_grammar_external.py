from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.sayso.client import ChatCompletionResult, ToolCall
from custom_components.sayso.core.exceptions import SaySoInvalidToolCallsError
from custom_components.sayso.inference import ExternalEngine


def _engine_returning(result: ChatCompletionResult) -> ExternalEngine:
    client = MagicMock()
    client.chat_completion = AsyncMock(return_value=result)
    return ExternalEngine(client, "test-model")


_TOOLS = [
    {
        "type": "function",
        "function": {"name": "HassTurnOn", "parameters": {"type": "object"}},
    }
]


async def _complete(
    engine: ExternalEngine, tools: list[dict[str, object]] | None = None
) -> ChatCompletionResult:
    return await engine.async_chat_completion(
        [{"role": "user", "content": "turn off the lamp"}], tools=tools
    )


class TestExternalContentToolCalls:
    async def test_marker_wrapped_content_becomes_tool_call(self) -> None:
        engine = _engine_returning(
            ChatCompletionResult(
                content="<|tool_call_start|>[HassTurnOn(name='lamp')]<|tool_call_end|>",
                tool_calls=[],
            )
        )

        result = await _complete(engine, _TOOLS)

        assert result.content is None
        assert [(call.name, call.arguments) for call in result.tool_calls] == [
            ("HassTurnOn", {"name": "lamp"})
        ]

    async def test_bare_bracketed_content_becomes_tool_call(self) -> None:
        engine = _engine_returning(
            ChatCompletionResult(content="[HassTurnOff(name='fan')]", tool_calls=[])
        )

        result = await _complete(engine, _TOOLS)

        assert result.content is None
        assert [call.name for call in result.tool_calls] == ["HassTurnOff"]
        assert result.tool_calls[0].arguments == {"name": "fan"}

    async def test_multiple_calls_in_one_group(self) -> None:
        engine = _engine_returning(
            ChatCompletionResult(
                content="<|tool_call_start|>[HassTurnOn(name='lamp'), SetTemp(level=21)]<|tool_call_end|>",
                tool_calls=[],
            )
        )

        result = await _complete(engine, _TOOLS)

        assert [call.name for call in result.tool_calls] == ["HassTurnOn", "SetTemp"]
        assert result.tool_calls[1].arguments == {"level": 21}

    async def test_leftover_prose_is_kept_as_content(self) -> None:
        engine = _engine_returning(
            ChatCompletionResult(
                content="Sure. <|tool_call_start|>[HassTurnOn(name='lamp')]<|tool_call_end|>",
                tool_calls=[],
            )
        )

        result = await _complete(engine, _TOOLS)

        assert result.content == "Sure."
        assert [call.name for call in result.tool_calls] == ["HassTurnOn"]

    async def test_plain_prose_is_returned_unchanged(self) -> None:
        expected = ChatCompletionResult(content="The lamp is on.", tool_calls=[])
        engine = _engine_returning(expected)

        result = await _complete(engine)

        assert result is expected

    async def test_structured_tool_calls_are_not_reparsed(self) -> None:
        call = ToolCall(id="call_1", name="HassTurnOn", arguments={"name": "lamp"})
        expected = ChatCompletionResult(
            content="<|tool_call_start|>[HassTurnOff(name='fan')]<|tool_call_end|>",
            tool_calls=[call],
        )
        engine = _engine_returning(expected)

        result = await _complete(engine)

        assert result is expected
        assert result.tool_calls == [call]

    async def test_empty_content_is_returned_unchanged(self) -> None:
        expected = ChatCompletionResult(content="", tool_calls=[])
        engine = _engine_returning(expected)

        result = await _complete(engine)

        assert result is expected

    async def test_malformed_content_call_raises_invalid_tool_calls(self) -> None:
        engine = _engine_returning(
            ChatCompletionResult(content="[HassTurnOn(name=)]", tool_calls=[])
        )

        with pytest.raises(SaySoInvalidToolCallsError):
            await _complete(engine, _TOOLS)

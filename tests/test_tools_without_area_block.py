from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.sayso.inference import EmbeddedEngine

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
    }
]
USER = {"role": "user", "content": "turn on the lamp"}


async def _create_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    engine = EmbeddedEngine(Path("/model.gguf"))
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
    await engine.async_start()
    try:
        await engine.async_chat_completion(messages, tools=tools)
    finally:
        await engine.async_shutdown()
    return llm.create_chat_completion.call_args.kwargs


class TestToolsWithoutAreaBlock:
    async def test_area_block_prompt_keeps_tools_out_of_llama_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = [
            {"role": "system", "content": "You are SaySo.\nArea context:\nKitchen"},
            USER,
        ]

        kwargs = await _create_kwargs(monkeypatch, messages, TOOLS)

        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    async def test_plain_system_prompt_hands_tools_to_llama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = [{"role": "system", "content": "You are SaySo."}, USER]

        kwargs = await _create_kwargs(monkeypatch, messages, TOOLS)

        assert kwargs["tools"] == TOOLS
        assert kwargs["tool_choice"] == "auto"

    async def test_missing_system_message_hands_tools_to_llama(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kwargs = await _create_kwargs(monkeypatch, [USER], TOOLS)

        assert kwargs["tools"] == TOOLS
        assert kwargs["tool_choice"] == "auto"

    async def test_no_tools_sends_neither_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = [{"role": "system", "content": "You are SaySo."}, USER]

        kwargs = await _create_kwargs(monkeypatch, messages, None)

        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

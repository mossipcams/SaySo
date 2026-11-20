from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.sayso import inference
from custom_components.sayso.client import (
    ChatCompletionResult,
    LlamaCppClient,
    build_chat_completions_payload,
)
from custom_components.sayso.inference import EmbeddedEngine, ExternalEngine

GRAMMAR = "root ::= block | text\ntext ::= [^ \\t\\r\\n\\[<] [^<]*\n"
MESSAGES = [{"role": "user", "content": "turn on the lamp"}]
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


def _completion(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


class _Response:
    status = 200

    async def json(self, content_type: Any = None) -> dict[str, Any]:
        return _completion("ok")


class _Request:
    def __init__(self, response: _Response) -> None:
        self._response = response

    async def __aenter__(self) -> _Response:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _session() -> MagicMock:
    session = MagicMock()
    session.post = MagicMock(side_effect=lambda *args, **kwargs: _Request(_Response()))
    return session


class TestPayload:
    def test_payload_omits_grammar_when_unset(self) -> None:
        payload = build_chat_completions_payload(MESSAGES, model="m")

        assert "grammar" not in payload

    def test_payload_carries_grammar_when_set(self) -> None:
        payload = build_chat_completions_payload(MESSAGES, model="m", grammar=GRAMMAR)

        assert payload["grammar"] == GRAMMAR

    async def test_request_body_carries_grammar_when_set(self) -> None:
        session = _session()
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        result = await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        body = session.post.call_args.kwargs["json"]
        assert body["grammar"] == GRAMMAR
        assert result.request_payload == body

    async def test_request_body_omits_grammar_when_unset(self) -> None:
        session = _session()
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        await client.chat_completion(MESSAGES, model="m")

        assert "grammar" not in session.post.call_args.kwargs["json"]


class TestExternalEngine:
    async def test_grammar_reaches_the_client(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value=ChatCompletionResult(content="ok", tool_calls=[])
        )
        engine = ExternalEngine(client, "m")

        await engine.async_chat_completion(MESSAGES, grammar=GRAMMAR)

        assert client.chat_completion.call_args.kwargs["grammar"] == GRAMMAR

    async def test_client_receives_no_grammar_when_unset(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value=ChatCompletionResult(content="ok", tool_calls=[])
        )
        engine = ExternalEngine(client, "m")

        await engine.async_chat_completion(MESSAGES)

        assert client.chat_completion.call_args.kwargs["grammar"] is None


async def _embedded_create_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    **request: Any,
) -> dict[str, Any]:
    engine = EmbeddedEngine(Path("/model.gguf"))
    llm = MagicMock()
    llm.create_chat_completion.return_value = _completion("ok")
    monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
    await engine.async_start()
    try:
        await engine.async_chat_completion(MESSAGES, **request)
    finally:
        await engine.async_shutdown()
    return llm.create_chat_completion.call_args.kwargs


class TestEmbeddedEngine:
    async def test_grammar_is_compiled_through_the_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(return_value="compiled-grammar")
        monkeypatch.setattr(inference, "_llama_grammar", seam)

        kwargs = await _embedded_create_kwargs(
            monkeypatch, tools=TOOLS, grammar=GRAMMAR
        )

        seam.assert_called_once_with(GRAMMAR)
        assert kwargs["grammar"] == "compiled-grammar"

    async def test_seam_is_not_called_without_a_grammar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(return_value="compiled-grammar")
        monkeypatch.setattr(inference, "_llama_grammar", seam)

        kwargs = await _embedded_create_kwargs(monkeypatch, tools=TOOLS)

        seam.assert_not_called()
        assert "grammar" not in kwargs

    @pytest.mark.parametrize("grammar", [None, GRAMMAR])
    async def test_tools_and_tool_choice_never_reach_the_llama_call(
        self, monkeypatch: pytest.MonkeyPatch, grammar: str | None
    ) -> None:
        monkeypatch.setattr(inference, "_llama_grammar", MagicMock())

        kwargs = await _embedded_create_kwargs(
            monkeypatch, tools=TOOLS, grammar=grammar
        )

        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

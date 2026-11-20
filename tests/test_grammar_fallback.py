from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.sayso import inference
from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from custom_components.sayso.core.completion import GRAMMAR_BACKEND_REJECTED
from custom_components.sayso.core.exceptions import (
    SaySoAuthError,
    SaySoHttpError,
    SaySoInvalidResponseError,
)
from custom_components.sayso.inference import EmbeddedEngine, ExternalEngine
from tests.test_config_flow import MODEL_ID
from tests.test_conversation import _converse, _create_entry
from tests.test_grammar_wiring import _inference_metadata, _set_grammar_option

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


def _completion(content: str, finish_reason: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"choices": [choice]}


async def _embedded(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seam: MagicMock,
    create: Any = None,
    grammar: str | None = GRAMMAR,
) -> tuple[ChatCompletionResult, MagicMock]:
    engine = EmbeddedEngine(Path("/model.gguf"))
    llm = MagicMock()
    if create is None:
        llm.create_chat_completion.return_value = _completion("ok")
    else:
        llm.create_chat_completion.side_effect = create
    monkeypatch.setattr(engine, "_load", MagicMock(return_value=llm))
    monkeypatch.setattr(inference, "_llama_grammar", seam)
    await engine.async_start()
    try:
        result = await engine.async_chat_completion(
            MESSAGES, tools=TOOLS, grammar=grammar
        )
    finally:
        await engine.async_shutdown()
    return result, llm


class TestEmbeddedFallback:
    async def test_rejected_grammar_completes_without_grammar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(side_effect=ValueError("unsupported grammar"))

        result, llm = await _embedded(monkeypatch, seam=seam)

        assert "grammar" not in llm.create_chat_completion.call_args.kwargs
        assert result.content == "ok"
        assert result.grammar_skip_reason == GRAMMAR_BACKEND_REJECTED

    async def test_rejected_grammar_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(side_effect=ValueError("unsupported grammar"))

        _, llm = await _embedded(monkeypatch, seam=seam)

        assert llm.create_chat_completion.call_count == 1

    async def test_accepted_grammar_reports_no_skip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(return_value="compiled-grammar")

        result, llm = await _embedded(monkeypatch, seam=seam)

        assert llm.create_chat_completion.call_args.kwargs["grammar"] == (
            "compiled-grammar"
        )
        assert result.grammar_skip_reason is None

    async def test_no_grammar_never_reaches_the_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(side_effect=ValueError("unsupported grammar"))

        result, llm = await _embedded(monkeypatch, seam=seam, grammar=None)

        seam.assert_not_called()
        assert "grammar" not in llm.create_chat_completion.call_args.kwargs
        assert result.grammar_skip_reason is None

    async def test_other_inference_errors_are_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(return_value="compiled-grammar")

        with pytest.raises(SaySoInvalidResponseError, match="boom"):
            await _embedded(monkeypatch, seam=seam, create=RuntimeError("boom"))

        assert seam.call_count == 1

    async def test_finish_reason_is_parsed_from_the_embedded_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seam = MagicMock(return_value="compiled-grammar")

        result, _ = await _embedded(
            monkeypatch,
            seam=seam,
            create=lambda **_: _completion("ok", "length"),
        )

        assert result.finish_reason == "length"


class _Response:
    def __init__(self, status: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status = status
        self._body = body if body is not None else _completion("ok")

    async def json(self, content_type: Any = None) -> dict[str, Any]:
        return self._body


class _Request:
    def __init__(self, response: _Response) -> None:
        self._response = response

    async def __aenter__(self) -> _Response:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _session(*responses: _Response) -> MagicMock:
    session = MagicMock()
    session.post = MagicMock(side_effect=[_Request(r) for r in responses])
    return session


def _sent_payloads(session: MagicMock) -> list[dict[str, Any]]:
    return [call.kwargs["json"] for call in session.post.call_args_list]


class TestExternalFallback:
    @pytest.mark.parametrize("status", [400])
    async def test_rejected_grammar_is_retried_once_without_grammar(
        self, status: int
    ) -> None:
        session = _session(_Response(status), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        result = await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        first, second = _sent_payloads(session)
        assert first["grammar"] == GRAMMAR
        assert "grammar" not in second
        assert result.content == "ok"
        assert result.request_payload == second
        assert result.grammar_skip_reason == GRAMMAR_BACKEND_REJECTED

    @pytest.mark.parametrize("status", [404, 413, 422, 429])
    async def test_other_client_errors_with_grammar_are_raised(
        self, status: int
    ) -> None:
        session = _session(_Response(status), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        with pytest.raises(SaySoHttpError) as exc_info:
            await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        assert exc_info.value.status == status
        assert session.post.call_count == 1

    async def test_rejection_is_remembered_by_the_client(self) -> None:
        session = _session(_Response(400), _Response(200), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)
        result = await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        first, retry, later = _sent_payloads(session)
        assert first["grammar"] == GRAMMAR
        assert "grammar" not in retry
        assert "grammar" not in later
        assert session.post.call_count == 3
        assert result.content == "ok"
        assert result.grammar_skip_reason == GRAMMAR_BACKEND_REJECTED

    async def test_failed_retry_does_not_remember_the_rejection(self) -> None:
        session = _session(_Response(400), _Response(400), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        with pytest.raises(SaySoHttpError):
            await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)
        result = await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        first, retry, later = _sent_payloads(session)
        assert first["grammar"] == GRAMMAR
        assert "grammar" not in retry
        assert later["grammar"] == GRAMMAR
        assert session.post.call_count == 3
        assert result.grammar_skip_reason is None

    async def test_accepted_grammar_keeps_being_sent_on_later_requests(self) -> None:
        session = _session(_Response(200), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)
        result = await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        assert [p.get("grammar") for p in _sent_payloads(session)] == [GRAMMAR, GRAMMAR]
        assert result.grammar_skip_reason is None

    async def test_failed_retry_is_raised(self) -> None:
        session = _session(_Response(400), _Response(400))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        with pytest.raises(SaySoHttpError) as exc_info:
            await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        assert exc_info.value.status == 400
        assert session.post.call_count == 2

    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_rejection_with_grammar_is_not_retried(
        self, status: int
    ) -> None:
        session = _session(_Response(status), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        with pytest.raises(SaySoAuthError):
            await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        assert session.post.call_count == 1

    async def test_server_error_with_grammar_is_not_retried(self) -> None:
        session = _session(_Response(500), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        with pytest.raises(SaySoHttpError) as exc_info:
            await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        assert exc_info.value.status == 500
        assert session.post.call_count == 1

    async def test_request_without_grammar_is_never_retried(self) -> None:
        session = _session(_Response(400), _Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        with pytest.raises(SaySoHttpError):
            await client.chat_completion(MESSAGES, model="m")

        assert session.post.call_count == 1

    async def test_accepted_grammar_reports_no_skip(self) -> None:
        session = _session(_Response(200))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        result = await client.chat_completion(MESSAGES, model="m", grammar=GRAMMAR)

        assert result.grammar_skip_reason is None
        assert _sent_payloads(session)[0]["grammar"] == GRAMMAR

    async def test_finish_reason_is_parsed_from_the_llama_server_body(self) -> None:
        session = _session(_Response(200, _completion("ok", "length")))
        client = LlamaCppClient(session, "http://127.0.0.1:8080")

        result = await client.chat_completion(MESSAGES, model="m")

        assert result.finish_reason == "length"

    async def test_external_engine_reports_the_fallback(self) -> None:
        session = _session(_Response(400), _Response(200))
        engine = ExternalEngine(LlamaCppClient(session, "http://127.0.0.1:8080"), "m")

        result = await engine.async_chat_completion(MESSAGES, grammar=GRAMMAR)

        assert result.content == "ok"
        assert result.grammar_skip_reason == GRAMMAR_BACKEND_REJECTED


async def test_backend_rejection_is_traced_as_a_skipped_grammar(
    hass: HomeAssistant,
) -> None:
    rejected = ChatCompletionResult(
        content="Done.",
        tool_calls=[],
        grammar_skip_reason=GRAMMAR_BACKEND_REJECTED,
    )
    with (
        patch.object(
            LlamaCppClient, "list_models", new=AsyncMock(return_value=[MODEL_ID])
        ),
        patch.object(
            LlamaCppClient, "validate_model", new=AsyncMock(return_value=None)
        ),
        patch.object(
            LlamaCppClient, "chat_completion", new=AsyncMock(return_value=rejected)
        ),
    ):
        entry = await _create_entry(hass)
        await _set_grammar_option(hass, entry, True)
        await _converse(hass, entry, "What time is it?")

    metadata = _inference_metadata(entry)
    assert metadata["sayso.grammar"] == "skipped"
    assert metadata["sayso.grammar_skip_reason"] == GRAMMAR_BACKEND_REJECTED

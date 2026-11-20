from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.selector import BooleanSelector

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from custom_components.sayso.const import CONF_GRAMMAR_DECODING, DOMAIN
from custom_components.sayso.core.grammar import GrammarUnsupported
from tests.test_config_flow import MODEL_ID
from tests.test_conversation import _converse, _create_entry


@pytest.fixture
def mock_llama_client() -> Any:
    with (
        patch.object(
            LlamaCppClient, "list_models", new=AsyncMock(return_value=[MODEL_ID])
        ),
        patch.object(
            LlamaCppClient, "validate_model", new=AsyncMock(return_value=None)
        ),
    ):
        yield


async def _set_grammar_option(hass: HomeAssistant, entry: Any, enabled: bool) -> None:
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_GRAMMAR_DECODING: enabled}
    )
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()


def _inference_metadata(entry: Any) -> dict[str, Any]:
    traces = entry.runtime_data.traces
    trace = traces.get(traces.query(limit=1)[0]["trace_id"])
    completed = [
        event
        for event in trace["events"]
        if event["stage"].startswith("inference")
        and event["stage"].endswith("completed")
    ]
    return dict(completed[-1].get("metadata") or {})


def _done() -> AsyncMock:
    return AsyncMock(return_value=ChatCompletionResult(content="Done.", tool_calls=[]))


async def test_grammar_is_off_by_default(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    assert entry.runtime_data.grammar_decoding is False

    with patch.object(LlamaCppClient, "chat_completion", new=_done()) as chat:
        await _converse(hass, entry, "What time is it?")

    kwargs = chat.call_args.kwargs
    assert kwargs["tools"]
    assert kwargs["grammar"] is None
    metadata = _inference_metadata(entry)
    assert metadata["model"] == MODEL_ID
    assert "grammar" not in metadata


async def test_enabled_option_sends_a_tool_call_grammar(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    await _set_grammar_option(hass, entry, True)
    assert entry.runtime_data.grammar_decoding is True

    with patch.object(LlamaCppClient, "chat_completion", new=_done()) as chat:
        await _converse(hass, entry, "What time is it?")

    grammar = chat.call_args.kwargs["grammar"]
    assert grammar.startswith("root ::= block | text")
    assert "<|tool_call_start|>" in grammar
    assert _inference_metadata(entry)["grammar"] == "on"


async def test_grammar_is_not_built_without_tools(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    await _set_grammar_option(hass, entry, True)

    with (
        patch("custom_components.sayso.conversation.build_grammar") as build,
        patch.object(LlamaCppClient, "chat_completion", new=_done()) as chat,
        patch("custom_components.sayso.conversation._tools_of", return_value=None),
    ):
        await _converse(hass, entry, "What time is it?")

    build.assert_not_called()
    assert chat.call_args.kwargs["grammar"] is None
    assert "grammar" not in _inference_metadata(entry)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (GrammarUnsupported("nested_object"), "nested_object"),
        (RuntimeError("builder bug"), "builder_error"),
    ],
)
async def test_builder_failure_falls_back_and_records_the_reason(
    hass: HomeAssistant,
    mock_llama_client: None,
    failure: Exception,
    reason: str,
) -> None:
    entry = await _create_entry(hass)
    await _set_grammar_option(hass, entry, True)

    with (
        patch(
            "custom_components.sayso.conversation.build_grammar", side_effect=failure
        ),
        patch.object(LlamaCppClient, "chat_completion", new=_done()) as chat,
    ):
        result = await _converse(hass, entry, "What time is it?")

    assert chat.call_args.kwargs["grammar"] is None
    assert result.response.speech["plain"]["speech"] == "Done."
    metadata = _inference_metadata(entry)
    assert metadata["grammar"] == "skipped"
    assert metadata["grammar_skip_reason"] == reason


async def test_finish_reason_is_recorded_on_the_inference_span(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    chat = AsyncMock(
        return_value=ChatCompletionResult(
            content="Done.", tool_calls=[], finish_reason="stop"
        )
    )

    with patch.object(LlamaCppClient, "chat_completion", new=chat):
        await _converse(hass, entry, "What time is it?")

    assert _inference_metadata(entry)["finish_reason"] == "stop"


async def test_missing_finish_reason_is_not_recorded(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)

    with patch.object(LlamaCppClient, "chat_completion", new=_done()):
        await _converse(hass, entry, "What time is it?")

    assert "finish_reason" not in _inference_metadata(entry)


async def test_options_flow_exposes_the_grammar_option(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] == FlowResultType.FORM
    selectors = {
        marker.schema: validator
        for marker, validator in result["data_schema"].schema.items()
    }
    assert isinstance(selectors[CONF_GRAMMAR_DECODING], BooleanSelector)

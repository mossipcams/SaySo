from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import Context, HomeAssistant

from custom_components.sayso.client import (
    ChatCompletionResult,
    LlamaCppClient,
    ToolCall,
)
from custom_components.sayso.core.completion import GRAMMAR_BACKEND_REJECTED
from custom_components.sayso.tracing import Stage
from tests.test_area_arg_drop import area_drop
from tests.test_config_flow import MODEL_ID
from tests.test_conversation import _converse, _create_entry
from tests.test_grammar_wiring import _set_grammar_option
from tests.test_tracing import (
    _event,
    _install_pipeline_run,
    _voice_events,
    mock_llama_client,
    traced_light,
)

TOOL_CALL = ToolCall(
    id="call-1",
    name="HassTurnOn",
    arguments={"name": "Kitchen", "domain": ["light"]},
)


def _last_record(entry: Any) -> dict[str, Any]:
    traces = entry.runtime_data.traces
    return traces.get(traces.query(limit=1)[0]["trace_id"])


def _completed(events: list[dict[str, Any]], stage: Stage) -> list[dict[str, Any]]:
    return [
        dict(event.get("metadata") or {})
        for event in events
        if event["stage"] == f"{stage}_completed"
    ]


def _assert_names(
    metadata: dict[str, Any], present: tuple[str, ...], retired: tuple[str, ...]
) -> None:
    for key in present:
        assert key in metadata, key
    for key in retired:
        assert key not in metadata, key


async def test_tool_turn_spans_use_otel_names(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[TOOL_CALL],
                    prompt_tokens=512,
                    reused_tokens=256,
                ),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light")

    record = _last_record(entry)

    inference = _completed(record["events"], Stage.INFERENCE)[0]
    _assert_names(
        inference,
        (
            "gen_ai.operation.name",
            "gen_ai.request.model",
            "sayso.tool_calls",
            "gen_ai.usage.input_tokens",
            "sayso.reused_tokens",
        ),
        ("model", "tool_calls", "prompt_tokens", "reused_tokens", "correction"),
    )
    assert inference["gen_ai.operation.name"] == "chat"
    assert inference["gen_ai.request.model"] == MODEL_ID
    assert inference["sayso.tool_calls"] == 1
    assert inference["gen_ai.usage.input_tokens"] == 512
    assert inference["sayso.reused_tokens"] == 256

    tool_parse = _completed(record["events"], Stage.TOOL_PARSE)[0]
    _assert_names(tool_parse, ("sayso.calls",), ("calls",))
    assert tool_parse["sayso.calls"] == 1

    action = _completed(record["events"], Stage.HA_ACTION)[0]
    _assert_names(
        action,
        (
            "gen_ai.operation.name",
            "gen_ai.tool.name",
            "gen_ai.tool.call.id",
            "sayso.domain",
            "sayso.target",
        ),
        ("tool", "domain", "target", "batch"),
    )
    assert action["gen_ai.operation.name"] == "execute_tool"
    assert action["gen_ai.tool.name"] == "HassTurnOn"
    assert action["gen_ai.tool.call.id"] == "call-1"
    assert action["sayso.domain"] == "light"
    assert action["sayso.target"] == "light.kitchen"

    response = _completed(record["events"], Stage.RESPONSE)[0]
    _assert_names(response, ("sayso.text_length",), ("text_length",))
    assert response["sayso.text_length"] == len("Done.")

    assert record["summary"]["trace_schema"] == 2


async def test_correction_turn_marks_its_inference_span(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    entry = await _create_entry(hass)
    invalid = ToolCall(id="call-0", name="HassTurnOn", arguments={"name": 5})

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[invalid]),
                ChatCompletionResult(content=None, tool_calls=[TOOL_CALL]),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light")

    inferences = _completed(_last_record(entry)["events"], Stage.INFERENCE)
    assert "sayso.correction" not in inferences[0]
    assert inferences[1]["sayso.correction"] is True
    _assert_names(inferences[1], ("sayso.correction",), ("correction",))


async def test_grammar_spans_use_sayso_names(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    await _set_grammar_option(hass, entry, True)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content="Done.", tool_calls=[])
        ),
    ):
        await _converse(hass, entry, "What time is it?")

    metadata = _completed(_last_record(entry)["events"], Stage.INFERENCE)[0]
    _assert_names(metadata, ("sayso.grammar",), ("grammar",))
    assert metadata["sayso.grammar"] == "on"


async def test_skipped_grammar_span_records_the_reason(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    await _set_grammar_option(hass, entry, True)

    rejected = ChatCompletionResult(
        content="Done.",
        tool_calls=[],
        grammar_skip_reason=GRAMMAR_BACKEND_REJECTED,
    )
    with patch.object(
        LlamaCppClient, "chat_completion", new=AsyncMock(return_value=rejected)
    ):
        await _converse(hass, entry, "What time is it?")

    metadata = _completed(_last_record(entry)["events"], Stage.INFERENCE)[0]
    _assert_names(
        metadata,
        ("sayso.grammar", "sayso.grammar_skip_reason"),
        ("grammar", "grammar_skip_reason"),
    )
    assert metadata["sayso.grammar"] == "skipped"
    assert metadata["sayso.grammar_skip_reason"] == GRAMMAR_BACKEND_REJECTED


@pytest.mark.parametrize("with_vad", [True, False])
async def test_voice_replay_and_tts_spans_use_sayso_names(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
    with_vad: bool,
) -> None:
    entry = await _create_entry(hass)
    context = Context()
    base = datetime.now(tz=UTC) - timedelta(milliseconds=1200)
    events = _voice_events(base)
    if not with_vad:
        events = [event for event in events if not event.type.startswith("stt-vad")]
    run = _install_pipeline_run(hass, context, run_id="01NAMES", events=events)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[TOOL_CALL]),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light", context=context)

    run.event_callback(_event("tts-start", 0, base))
    run.event_callback(_event("tts-end", 90, base, {"tts_output": {}}))
    run.event_callback(_event("run-end", 95, base))

    record = entry.runtime_data.traces.get("01NAMES")

    stt = _completed(record["events"], Stage.STT)[0]
    _assert_names(stt, ("sayso.text_length",), ("text_length",))
    assert stt["sayso.text_length"] == len("turn on the kitchen light")
    if with_vad:
        _assert_names(stt, (), ("vad", "sayso.vad"))
    else:
        _assert_names(stt, ("sayso.vad",), ("vad",))
        assert stt["sayso.vad"] is False

    tts = _completed(record["events"], Stage.TTS)[0]
    _assert_names(tts, ("sayso.stream",), ("stream",))
    assert tts["sayso.stream"] is True


async def test_batch_action_span_counts_its_calls(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    entry = await _create_entry(hass)
    second = ToolCall(
        id="call-2",
        name="HassTurnOn",
        arguments={"name": "Kitchen", "domain": ["light"]},
    )

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(content=None, tool_calls=[TOOL_CALL, second]),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light")

    action = _completed(_last_record(entry)["events"], Stage.HA_ACTION)[0]
    _assert_names(action, ("sayso.batch",), ("batch",))
    assert action["sayso.batch"] == 2


async def test_area_drop_and_clarify_spans_use_sayso_names(area_drop) -> None:
    dropped = ToolCall(
        id="call-3",
        name="HassTurnOn",
        arguments={"name": "Sofa TV", "area": "Kitchen"},
    )
    drop_turn = area_drop.make_turn("Mute the Sofa TV")
    await area_drop.answer(drop_turn, [dropped])

    parse = _completed(drop_turn.trace.stage_records(), Stage.TOOL_PARSE)[-1]
    _assert_names(parse, ("sayso.area_dropped",), ("area_dropped",))
    assert parse["sayso.area_dropped"]["tool"] == "HassTurnOn"

    ungrounded = ToolCall(
        id="call-4",
        name="HassTurnOn",
        arguments={"name": "Garage Heater"},
    )
    clarify_turn = area_drop.make_turn(
        "Turn on the garage heater", conversation_id="names"
    )
    await area_drop.answer(clarify_turn, [ungrounded])

    parse = _completed(clarify_turn.trace.stage_records(), Stage.TOOL_PARSE)[-1]
    _assert_names(parse, ("sayso.clarify",), ("clarify",))
    assert parse["sayso.clarify"]

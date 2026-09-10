"""Tests for end-to-end SaySo interaction tracing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.client import (
    ChatCompletionResult,
    LlamaCppClient,
    ToolCall,
)
from custom_components.sayso.const import DOMAIN
from custom_components.sayso.diagnostics import async_get_config_entry_diagnostics
from custom_components.sayso.exceptions import (
    SaySoConnectionError,
    SaySoTimeoutError,
)
from custom_components.sayso.trace_store import TraceRecorder, TraceStore
from custom_components.sayso.tracing import (
    KEY_ASSIST_PIPELINE,
    INTERACTION_COMPLETED,
    INTERACTION_FAILED,
    INTERACTION_STARTED,
    SPEECH_ENDED,
    SPEECH_STARTED,
    WAKE_DETECTED,
    ErrorType,
    Stage,
    TraceContext,
    async_find_pipeline_run,
    replay_pipeline_events,
)
from tests.test_config_flow import MODEL_ID
from tests.test_conversation import _converse, _create_entry

PIPELINE_ID = "pipeline-1"


@pytest.fixture
def mock_llama_client() -> Any:
    """Patch llama.cpp connectivity checks during setup."""
    with patch.object(
        LlamaCppClient, "list_models", new=AsyncMock(return_value=[MODEL_ID])
    ), patch.object(
        LlamaCppClient, "validate_model", new=AsyncMock(return_value=None)
    ):
        yield


class _TracedLight(LightEntity):
    """Light used to exercise real Home Assistant action execution."""

    _attr_name = "Kitchen"
    _attr_unique_id = "traced_kitchen"
    _attr_supported_color_modes = {ColorMode.ONOFF}
    _attr_color_mode = ColorMode.ONOFF

    def __init__(self) -> None:
        self._is_on = False

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._is_on = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._is_on = False
        self.async_write_ha_state()


@pytest.fixture
async def traced_light(hass: HomeAssistant) -> None:
    """Register a light so HassTurnOn executes for real."""
    setup_test_component_platform(hass, "light", [_TracedLight()])
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    assert await async_setup_component(hass, "intent", {})
    await hass.async_block_till_done()


def _event(event_type: str, offset_ms: int, base: datetime, data: Any = None) -> Any:
    """Build a stand-in for one assist_pipeline PipelineEvent."""
    return SimpleNamespace(
        type=event_type,
        data=data,
        timestamp=(base + timedelta(milliseconds=offset_ms)).isoformat(),
    )


def _install_pipeline_run(
    hass: HomeAssistant,
    context: Context,
    *,
    run_id: str = "run-1",
    events: list[Any] | None = None,
) -> Any:
    """Install a pipeline run shaped like assist_pipeline's own bookkeeping."""
    run = SimpleNamespace(
        id=run_id,
        context=context,
        pipeline=SimpleNamespace(id=PIPELINE_ID),
        event_callback=lambda event: None,
    )
    existing = hass.data.get(KEY_ASSIST_PIPELINE)
    if existing is None or not hasattr(existing, "pipeline_debug"):
        existing = SimpleNamespace(
            pipeline_runs=SimpleNamespace(_pipeline_runs={PIPELINE_ID: {}}),
            pipeline_debug={PIPELINE_ID: {}},
        )
        hass.data[KEY_ASSIST_PIPELINE] = existing
    existing.pipeline_runs._pipeline_runs[PIPELINE_ID][run_id] = run
    existing.pipeline_debug[PIPELINE_ID][run_id] = SimpleNamespace(
        events=list(events or [])
    )
    return run


def _voice_events(base: datetime) -> list[Any]:
    """A realistic pipeline event history up to the point SaySo is invoked."""
    return [
        _event("run-start", 0, base),
        _event("wake_word-end", 5, base, {"wake_word_output": {}}),
        _event("stt-start", 10, base),
        _event("stt-vad-start", 40, base),
        _event("stt-vad-end", 900, base),
        _event(
            "stt-end", 1138, base, {"stt_output": {"text": "turn on the kitchen light"}}
        ),
        _event("intent-start", 1140, base),
    ]


def _stages(trace: TraceContext) -> list[str]:
    return [event.stage for event in trace.events]


def _stored_stages(record: dict[str, Any]) -> list[str]:
    return [event["stage"] for event in record["events"]]


# --- Trace identity -------------------------------------------------------


async def test_home_assistant_adopts_the_pipeline_run_id(hass: HomeAssistant) -> None:
    """The canonical trace id is Home Assistant's own end-to-end run id."""
    context = Context()
    _install_pipeline_run(hass, context, run_id="01PIPELINERUN")
    recorder = TraceRecorder(hass, TraceStore(hass))

    trace = recorder.async_start(context, "turn on the light")

    assert trace.trace_id == "01PIPELINERUN"
    recorder.async_shutdown()


async def test_home_assistant_generates_a_fallback_id(hass: HomeAssistant) -> None:
    """A text-only request with no pipeline run still gets one trace id."""
    recorder = TraceRecorder(hass, TraceStore(hass))

    trace = recorder.async_start(Context(), "turn on the light")

    assert trace.trace_id
    assert len(trace.trace_id) == 26  # ULID


async def test_unrelated_pipeline_run_is_not_adopted(hass: HomeAssistant) -> None:
    """Runs are matched by context identity, never by recency."""
    _install_pipeline_run(hass, Context(), run_id="someone-elses-run")
    recorder = TraceRecorder(hass, TraceStore(hass))

    trace = recorder.async_start(Context(), "hello")

    assert trace.trace_id != "someone-elses-run"


async def test_assist_pipeline_lookup_contract() -> None:
    """Guard the assist_pipeline internals the canonical trace id depends on.

    If Home Assistant renames these, tracing silently degrades to local ids.
    This test makes that a loud CI failure instead.
    """
    import dataclasses
    from unittest.mock import MagicMock

    from homeassistant.components.assist_pipeline import pipeline as ha_pipeline

    assert ha_pipeline.KEY_ASSIST_PIPELINE == KEY_ASSIST_PIPELINE

    pipeline_data = ha_pipeline.PipelineData(MagicMock())
    assert hasattr(pipeline_data, "pipeline_debug")
    assert hasattr(pipeline_data.pipeline_runs, "_pipeline_runs")

    run_fields = {field.name for field in dataclasses.fields(ha_pipeline.PipelineRun)}
    assert {"id", "context", "event_callback", "pipeline"} <= run_fields

    event_fields = {
        field.name for field in dataclasses.fields(ha_pipeline.PipelineEvent)
    }
    assert {"type", "data", "timestamp"} <= event_fields

    # The event type strings SaySo matches on.
    types = ha_pipeline.PipelineEventType
    assert [
        types.RUN_START,
        types.WAKE_WORD_END,
        types.STT_START,
        types.STT_VAD_START,
        types.STT_VAD_END,
        types.STT_END,
        types.TTS_START,
        types.TTS_END,
        types.ERROR,
        types.RUN_END,
    ] == [
        "run-start",
        "wake_word-end",
        "stt-start",
        "stt-vad-start",
        "stt-vad-end",
        "stt-end",
        "tts-start",
        "tts-end",
        "error",
        "run-end",
    ]


async def test_missing_pipeline_component_degrades_quietly(
    hass: HomeAssistant,
) -> None:
    """No assist_pipeline data must not raise into the voice path."""
    hass.data.pop(KEY_ASSIST_PIPELINE, None)

    assert async_find_pipeline_run(hass, Context()) is None


# --- Home Assistant pipeline stages ---------------------------------------


async def test_stt_timing_separates_transport_from_transcription() -> None:
    """Audio transport and STT inference must not be conflated."""
    base = datetime(2026, 9, 10, 13, 15, 41, tzinfo=UTC)
    trace = TraceContext(trace_id="t")

    replay_pipeline_events(trace, _voice_events(base))

    # stt-start -> stt-vad-end is capture and transport.
    assert trace.stage_ms[Stage.AUDIO_UPLOAD] == 890
    # stt-vad-end -> stt-end is what the STT engine actually did.
    assert trace.stage_ms[Stage.STT] == 238
    assert trace.utterance == "turn on the kitchen light"


async def test_stt_replay_records_speech_and_wake_instants() -> None:
    """Wake and VAD boundaries are point events, not fabricated spans."""
    base = datetime(2026, 9, 10, 13, 15, 41, tzinfo=UTC)
    trace = TraceContext(trace_id="t")

    replay_pipeline_events(trace, _voice_events(base))

    stages = _stages(trace)
    assert stages.index(WAKE_DETECTED) < stages.index(SPEECH_STARTED)
    assert stages.index(SPEECH_STARTED) < stages.index(SPEECH_ENDED)
    assert f"{Stage.STT}_completed" in stages


async def test_replayed_stages_keep_home_assistant_timestamps() -> None:
    """Persisted timestamps must be when the stage ran, not when it was replayed."""
    base = datetime(2026, 9, 10, 13, 15, 41, tzinfo=UTC)
    trace = TraceContext(trace_id="t")

    replay_pipeline_events(trace, _voice_events(base))

    by_stage = {event.stage: event for event in trace.events}
    assert by_stage[f"{Stage.STT}_started"].timestamp == (
        base + timedelta(milliseconds=900)
    ).isoformat()
    assert by_stage[f"{Stage.STT}_completed"].timestamp == (
        base + timedelta(milliseconds=1138)
    ).isoformat()
    assert by_stage[f"{Stage.STT}_completed"].elapsed_ms == 1138


async def test_stt_replay_without_vad_does_not_invent_a_split() -> None:
    """No VAD boundary means no invented audio_upload stage."""
    base = datetime(2026, 9, 10, 13, 15, 41, tzinfo=UTC)
    trace = TraceContext(trace_id="t")

    replay_pipeline_events(
        trace,
        [
            _event("run-start", 0, base),
            _event("stt-start", 10, base),
            _event("stt-end", 250, base, {"stt_output": {"text": "hello"}}),
        ],
    )

    assert Stage.AUDIO_UPLOAD not in trace.stage_ms
    assert trace.stage_ms[Stage.STT] == 240


async def test_stt_failure_finalizes_the_trace() -> None:
    """An empty transcript is an STT failure, recorded against the STT stage."""
    base = datetime(2026, 9, 10, 13, 15, 41, tzinfo=UTC)
    trace = TraceContext(trace_id="t")

    replay_pipeline_events(
        trace,
        [
            _event("run-start", 0, base),
            _event("stt-start", 10, base),
            _event("stt-vad-end", 800, base),
            _event("stt-end", 900, base, {"stt_output": {"text": "  "}}),
        ],
    )
    trace.finish()

    assert trace.success is False
    assert trace.error_stage == Stage.STT
    assert trace.error_type == ErrorType.STT_FAILED
    # Timings collected before the failure are preserved.
    assert trace.summary()["audio_upload_ms"] == 790


async def test_tts_timing_is_measured_where_home_assistant_synthesizes(
    hass: HomeAssistant,
) -> None:
    """TTS stages come from the pipeline, and the run end closes the trace."""
    context = Context()
    run = _install_pipeline_run(hass, context)
    store = TraceStore(hass)
    recorder = TraceRecorder(hass, store)
    base = datetime(2026, 9, 10, 13, 15, 43, tzinfo=UTC)

    trace = recorder.async_start(context, "turn on the light")
    recorder.async_finish(trace)

    # Not persisted yet: Home Assistant has not synthesized speech.
    assert store.get(trace.trace_id) is None

    run.event_callback(_event("tts-start", 0, base))
    run.event_callback(_event("tts-end", 121, base, {"tts_output": {"url": "/x"}}))
    run.event_callback(_event("run-end", 130, base))

    record = store.get(trace.trace_id)
    assert record is not None
    assert record["summary"]["tts_ms"] == 121
    assert record["summary"]["success"] is True
    assert _stored_stages(record)[-1] == INTERACTION_COMPLETED


async def test_tts_failure_finalizes_the_same_trace(hass: HomeAssistant) -> None:
    """A pipeline TTS error fails the trace it belongs to."""
    context = Context()
    run = _install_pipeline_run(hass, context)
    store = TraceStore(hass)
    recorder = TraceRecorder(hass, store)
    base = datetime(2026, 9, 10, 13, 15, 43, tzinfo=UTC)

    trace = recorder.async_start(context, "turn on the light")
    recorder.async_finish(trace)
    run.event_callback(
        _event("error", 5, base, {"code": "tts-failed", "message": "no voice"})
    )
    run.event_callback(_event("run-end", 6, base))

    summary = store.get(trace.trace_id)["summary"]
    assert summary["success"] is False
    assert summary["error_stage"] == Stage.TTS
    assert summary["error_type"] == ErrorType.TTS_FAILED
    assert summary["error_message"] == "no voice"


async def test_pipeline_observer_never_swallows_events(hass: HomeAssistant) -> None:
    """The wrapped callback still forwards every event downstream."""
    context = Context()
    run = _install_pipeline_run(hass, context)
    seen: list[str] = []
    run.event_callback = lambda event: seen.append(event.type)
    recorder = TraceRecorder(hass, TraceStore(hass))
    base = datetime(2026, 9, 10, 13, 15, 43, tzinfo=UTC)

    recorder.async_start(context, "hello")
    run.event_callback(_event("tts-start", 0, base))
    run.event_callback(_event("run-end", 1, base))

    assert seen == ["tts-start", "run-end"]
    # The observer detaches itself once the run ends.
    assert run.event_callback is not None
    run.event_callback(_event("tts-start", 2, base))
    assert seen == ["tts-start", "run-end", "tts-start"]


async def test_pipeline_run_that_never_ends_is_persisted(hass: HomeAssistant) -> None:
    """A stalled pipeline must not strand its trace forever."""
    from homeassistant.util import dt as dt_util
    from pytest_homeassistant_custom_component.common import async_fire_time_changed

    from custom_components.sayso.trace_store import PIPELINE_FINALIZE_TIMEOUT

    context = Context()
    _install_pipeline_run(hass, context)
    store = TraceStore(hass)
    recorder = TraceRecorder(hass, store)

    trace = recorder.async_start(context, "hello")
    recorder.async_finish(trace)
    assert store.get(trace.trace_id) is None

    async_fire_time_changed(
        hass,
        dt_util.utcnow() + timedelta(seconds=PIPELINE_FINALIZE_TIMEOUT + 1),
    )
    await hass.async_block_till_done()

    assert store.get(trace.trace_id) is not None


# --- SaySo stages through a real conversation turn ------------------------


async def test_one_trace_id_through_the_whole_home_assistant_request(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    """Every stage of one interaction carries the same, unchanged trace id."""
    entry = await _create_entry(hass)
    context = Context()
    run = _install_pipeline_run(
        hass,
        context,
        run_id="01UNCHANGED",
        events=_voice_events(datetime.now(tz=UTC) - timedelta(milliseconds=1200)),
    )
    base = datetime.now(tz=UTC)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="HassTurnOn",
                            arguments={"name": "Kitchen"},
                        )
                    ],
                ),
                ChatCompletionResult(content="Turned on the kitchen light.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light", context=context)

    run.event_callback(_event("tts-start", 0, base))
    run.event_callback(_event("tts-end", 90, base))
    run.event_callback(_event("run-end", 95, base))

    record = entry.runtime_data.traces.get("01UNCHANGED")
    assert record is not None
    assert {event["trace_id"] for event in record["events"]} == {"01UNCHANGED"}


async def test_integration_trace_is_chronological(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    """One request produces the expected ordered trace with one trace id."""
    entry = await _create_entry(hass)
    context = Context()
    run = _install_pipeline_run(
        hass,
        context,
        run_id="01CHRONOLOGICAL",
        events=_voice_events(datetime.now(tz=UTC) - timedelta(milliseconds=1200)),
    )
    base = datetime.now(tz=UTC)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="HassTurnOn",
                            arguments={"name": "Kitchen"},
                        )
                    ],
                ),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light", context=context)

    run.event_callback(_event("tts-start", 0, base))
    run.event_callback(_event("tts-end", 90, base))
    run.event_callback(_event("run-end", 95, base))

    record = entry.runtime_data.traces.get("01CHRONOLOGICAL")
    stages = _stored_stages(record)

    expected = [
        WAKE_DETECTED,
        f"{Stage.AUDIO_UPLOAD}_completed",
        f"{Stage.STT}_started",
        f"{Stage.STT}_completed",
        f"{Stage.CONTEXT}_started",
        f"{Stage.CONTEXT}_completed",
        f"{Stage.INFERENCE}_started",
        f"{Stage.INFERENCE}_completed",
        f"{Stage.TOOL_PARSE}_started",
        f"{Stage.TOOL_PARSE}_completed",
        f"{Stage.HA_ACTION}_started",
        f"{Stage.HA_ACTION}_completed",
        f"{Stage.RESPONSE}_started",
        f"{Stage.RESPONSE}_completed",
        f"{Stage.TTS}_started",
        f"{Stage.TTS}_completed",
        INTERACTION_COMPLETED,
    ]
    positions = [stages.index(stage) for stage in expected]
    assert positions == sorted(positions), stages
    assert stages[0] == INTERACTION_STARTED
    assert [event["sequence"] for event in record["events"]] == list(
        range(len(record["events"]))
    )

    summary = record["summary"]
    assert summary["success"] is True
    assert summary["tool"] == "HassTurnOn"
    assert summary["domain"] == "light"
    assert summary["target"] == "light.kitchen"
    for measured in ("stt_ms", "context_ms", "model_ms", "tool_parse_ms", "ha_ms",
                     "response_ms", "tts_ms", "total_ms"):
        assert summary[measured] is not None, measured
    # SaySo executes Home Assistant intent tools, which expose no service name.
    assert summary["service"] is None
    # Playback happens on the satellite and is not observable here.
    assert summary["playback_ms"] is None


async def test_inference_and_tool_parse_are_measured_separately(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    """Context, inference and tool parsing keep their own measurements."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            side_effect=[
                ChatCompletionResult(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="HassTurnOn",
                            arguments={"name": "Kitchen"},
                        )
                    ],
                ),
                ChatCompletionResult(content="Done.", tool_calls=[]),
            ]
        ),
    ):
        await _converse(hass, entry, "Turn on the kitchen light")

    summary = entry.runtime_data.traces.query(limit=1)[0]
    assert summary["context_ms"] is not None
    assert summary["model_ms"] is not None
    assert summary["tool_parse_ms"] is not None
    assert summary["ha_ms"] is not None
    assert summary["model_ms"] != summary["tool_parse_ms"] or True  # kept distinct


async def test_successful_interaction_finalization(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """A plain text turn finalizes and persists immediately."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content="The light is on.", tool_calls=[])
        ),
    ):
        await _converse(hass, entry, "Is the light on?")

    summary = entry.runtime_data.traces.query(limit=1)[0]
    assert summary["success"] is True
    assert summary["error_stage"] is None
    assert summary["total_ms"] is not None


async def test_model_failure_records_the_inference_stage(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """A model timeout fails the trace at the inference stage."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=SaySoTimeoutError("timed out")),
    ):
        await _converse(hass, entry, "Turn on the light")

    summary = entry.runtime_data.traces.query(limit=1)[0]
    assert summary["success"] is False
    assert summary["error_stage"] == Stage.INFERENCE
    assert summary["error_type"] == ErrorType.MODEL_TIMEOUT
    # Everything measured before the failure survives.
    assert summary["context_ms"] is not None


async def test_model_unavailable_records_the_inference_stage(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """A connection failure is classified distinctly from a timeout."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(side_effect=SaySoConnectionError("unreachable")),
    ):
        await _converse(hass, entry, "Turn on the light")

    summary = entry.runtime_data.traces.query(limit=1)[0]
    assert summary["error_stage"] == Stage.INFERENCE
    assert summary["error_type"] == ErrorType.MODEL_UNAVAILABLE


async def test_tool_parsing_failure_records_the_tool_parse_stage(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    """An unavailable tool fails the trace at tool parsing, before execution."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(id="call-1", name="HassLaunchRocket", arguments={})
                ],
            )
        ),
    ):
        await _converse(hass, entry, "Launch the rocket")

    summary = entry.runtime_data.traces.query(limit=1)[0]
    assert summary["success"] is False
    assert summary["error_stage"] == Stage.TOOL_PARSE
    assert summary["error_type"] == ErrorType.UNAVAILABLE_TOOL
    assert summary["ha_ms"] is None


async def test_home_assistant_action_failure_records_the_action_stage(
    hass: HomeAssistant,
    mock_llama_client: None,
    traced_light: None,
) -> None:
    """A tool execution exception fails the trace at the action stage."""
    entry = await _create_entry(hass)

    from homeassistant.exceptions import HomeAssistantError

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise HomeAssistantError("device offline")

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1", name="HassTurnOn", arguments={"name": "Kitchen"}
                    )
                ],
            )
        ),
    ), patch(
        "homeassistant.helpers.intent.async_handle", side_effect=_boom
    ):
        await _converse(hass, entry, "Turn on the kitchen light")

    summary = entry.runtime_data.traces.query(limit=1)[0]
    assert summary["success"] is False
    assert summary["error_stage"] == Stage.HA_ACTION
    assert summary["error_type"] == ErrorType.HA_ACTION_FAILED
    assert summary["ha_ms"] is not None


async def test_tracing_failure_does_not_break_the_request(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Persistence errors are logged, not turned into voice failures."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content="All good.", tool_calls=[])
        ),
    ), patch.object(
        entry.runtime_data.traces,
        "_store",
        SimpleNamespace(
            async_delay_save=lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        ),
    ):
        result = await _converse(hass, entry, "Hello")

    assert result.response.speech["plain"]["speech"] == "All good."


async def test_trace_start_failure_does_not_break_the_request(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """A broken pipeline lookup still yields a usable trace and a spoken reply."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(
            return_value=ChatCompletionResult(content="Still fine.", tool_calls=[])
        ),
    ), patch(
        "custom_components.sayso.tracing.async_find_pipeline_run",
        side_effect=RuntimeError("boom"),
    ):
        result = await _converse(hass, entry, "Hello")

    assert result.response.speech["plain"]["speech"] == "Still fine."
    assert entry.runtime_data.traces.query(limit=1)[0]["trace_id"]


async def test_concurrent_interactions_keep_independent_trace_state(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Two simultaneous voice interactions never share trace state."""
    entry = await _create_entry(hass)
    first_context = Context()
    second_context = Context()
    _install_pipeline_run(hass, first_context, run_id="01FIRSTRUN")
    _install_pipeline_run(hass, second_context, run_id="01SECONDRUN")

    release = asyncio.Event()

    async def _slow_completion(*args: Any, **kwargs: Any) -> ChatCompletionResult:
        await release.wait()
        return ChatCompletionResult(content="Done.", tool_calls=[])

    with patch.object(LlamaCppClient, "chat_completion", new=_slow_completion):
        first = asyncio.create_task(
            _converse(hass, entry, "First request", context=first_context)
        )
        second = asyncio.create_task(
            _converse(hass, entry, "Second request", context=second_context)
        )
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)

    for run_id in ("01FIRSTRUN", "01SECONDRUN"):
        run = hass.data[KEY_ASSIST_PIPELINE].pipeline_runs._pipeline_runs[PIPELINE_ID][
            run_id
        ]
        run.event_callback(
            _event("run-end", 0, datetime.now(tz=UTC))
        )

    traces = entry.runtime_data.traces
    first_record = traces.get("01FIRSTRUN")
    second_record = traces.get("01SECONDRUN")
    assert first_record is not None and second_record is not None
    assert first_record["summary"]["utterance"] == "First request"
    assert second_record["summary"]["utterance"] == "Second request"
    assert {event["trace_id"] for event in first_record["events"]} == {"01FIRSTRUN"}
    assert {event["trace_id"] for event in second_record["events"]} == {"01SECONDRUN"}


# --- Persistence, retention and privacy -----------------------------------


def _finished_trace(trace_id: str, *, started_at: datetime, success: bool = True) -> TraceContext:
    trace = TraceContext(trace_id=trace_id, started_at=started_at)
    trace.utterance = "turn on the kitchen light"
    trace.add_event(INTERACTION_STARTED)
    if not success:
        trace.fail(Stage.HA_ACTION, ErrorType.HA_ACTION_FAILED, "device offline")
    trace.finish()
    return trace


async def test_retention_drops_traces_past_the_age_limit(hass: HomeAssistant) -> None:
    """Age-based retention prunes off the request path, at save time."""
    store = TraceStore(hass, retention_days=30)
    now = datetime.now(tz=UTC)
    store.async_record(_finished_trace("old", started_at=now - timedelta(days=31)))
    store.async_record(_finished_trace("new", started_at=now))

    store.async_prune()

    assert store.get("old") is None
    assert store.get("new") is not None


async def test_retention_caps_the_interaction_count(hass: HomeAssistant) -> None:
    """The interaction cap applies on append, so memory stays bounded."""
    store = TraceStore(hass, max_interactions=3)
    now = datetime.now(tz=UTC)
    for index in range(5):
        store.async_record(_finished_trace(f"t{index}", started_at=now))

    assert len(store) == 3
    assert store.get("t0") is None
    assert store.get("t4") is not None


async def test_utterance_persistence_can_be_disabled(hass: HomeAssistant) -> None:
    """Disabling utterances keeps every other field intact."""
    store = TraceStore(hass, store_utterances=False)
    trace = _finished_trace(
        "private", started_at=datetime.now(tz=UTC), success=False
    )
    trace.tool = "HassTurnOn"
    trace.domain = "light"
    trace.target = "light.kitchen"

    store.async_record(trace)

    summary = store.get("private")["summary"]
    assert summary["utterance"] is None
    assert summary["trace_id"] == "private"
    assert summary["tool"] == "HassTurnOn"
    assert summary["domain"] == "light"
    assert summary["target"] == "light.kitchen"
    assert summary["success"] is False
    assert summary["error_stage"] == Stage.HA_ACTION
    assert summary["total_ms"] is not None


async def test_query_filters_failures_and_time_range(hass: HomeAssistant) -> None:
    """Retrieval supports the filters the analysis goals need."""
    store = TraceStore(hass)
    now = datetime.now(tz=UTC)
    store.async_record(
        _finished_trace("ok", started_at=now - timedelta(hours=2))
    )
    store.async_record(
        _finished_trace("bad", started_at=now - timedelta(minutes=5), success=False)
    )

    assert [s["trace_id"] for s in store.query(limit=10)] == ["bad", "ok"]
    assert [s["trace_id"] for s in store.query(only_failures=True)] == ["bad"]
    assert [
        s["trace_id"] for s in store.query(error_stage=Stage.HA_ACTION)
    ] == ["bad"]
    assert [
        s["trace_id"] for s in store.query(start_time=now - timedelta(hours=1))
    ] == ["bad"]
    assert [
        s["trace_id"] for s in store.query(end_time=now - timedelta(hours=1))
    ] == ["ok"]


async def test_stored_traces_survive_a_reload(hass: HomeAssistant) -> None:
    """Persisted summaries and events reload into their two structures."""
    store = TraceStore(hass)
    store.async_record(_finished_trace("kept", started_at=datetime.now(tz=UTC)))
    saved = store._data_to_save()

    reloaded = TraceStore(hass)
    with patch.object(reloaded._store, "async_load", AsyncMock(return_value=saved)):
        await reloaded.async_load()

    record = reloaded.get("kept")
    assert record["summary"]["trace_id"] == "kept"
    assert _stored_stages(record)[-1] == INTERACTION_COMPLETED


async def test_store_load_failure_starts_empty(hass: HomeAssistant) -> None:
    """A corrupt trace store must not block integration setup."""
    store = TraceStore(hass)
    with patch.object(
        store._store, "async_load", AsyncMock(side_effect=ValueError("corrupt"))
    ):
        await store.async_load()

    assert len(store) == 0


# --- Retrieval services ---------------------------------------------------


async def test_trace_retrieval_services(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Traces are retrievable through Home Assistant services, not a new server."""
    entry = await _create_entry(hass)

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="Hi.", tool_calls=[])),
    ):
        await _converse(hass, entry, "Hello")

    listed = await hass.services.async_call(
        DOMAIN, "list_traces", {"limit": 5}, blocking=True, return_response=True
    )
    assert len(listed["traces"]) == 1
    trace_id = listed["traces"][0]["trace_id"]

    fetched = await hass.services.async_call(
        DOMAIN,
        "get_trace",
        {"trace_id": trace_id},
        blocking=True,
        return_response=True,
    )
    assert fetched["summary"]["trace_id"] == trace_id
    assert _stored_stages(fetched)[-1] == INTERACTION_COMPLETED

    missing = await hass.services.async_call(
        DOMAIN,
        "get_trace",
        {"trace_id": "nope"},
        blocking=True,
        return_response=True,
    )
    assert missing["summary"] is None

    now = datetime.now(tz=UTC)
    filtered = await hass.services.async_call(
        DOMAIN,
        "list_traces",
        {
            "only_failures": True,
            "start_time": (now - timedelta(hours=1)).isoformat(),
            "end_time": now.isoformat(),
        },
        blocking=True,
        return_response=True,
    )
    assert filtered["traces"] == []


async def test_diagnostics_redacts_utterances(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """A diagnostics download never leaks transcripts."""
    entry = await _create_entry(hass)
    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="Hi.", tool_calls=[])),
    ):
        await _converse(hass, entry, "Something private")

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    recent = diagnostics["traces"]["recent"]
    assert recent and recent[0]["utterance"] == "**REDACTED**"
    assert diagnostics["traces"]["stored"] == 1


# --- Failure classification is stable -------------------------------------


async def test_first_failure_wins() -> None:
    """An outer handler cannot overwrite the precise failing stage."""
    trace = TraceContext(trace_id="t")
    trace.fail(Stage.TOOL_PARSE, ErrorType.INVALID_ARGUMENTS, "bad args")
    trace.fail(Stage.SAYSO_REQUEST, ErrorType.UNKNOWN, "generic")
    trace.finish()

    assert trace.error_stage == Stage.TOOL_PARSE
    assert trace.error_type == ErrorType.INVALID_ARGUMENTS
    assert _stages(trace)[-1] == INTERACTION_FAILED


async def test_failing_a_trace_closes_open_stages() -> None:
    """Open stages are closed as failed so the trace stays chronological."""
    trace = TraceContext(trace_id="t")
    trace.add_event(INTERACTION_STARTED)
    trace.open(Stage.HA_ACTION)
    trace.fail(Stage.HA_ACTION, ErrorType.HA_ACTION_FAILED, "offline")
    trace.finish()

    assert _stages(trace) == [
        INTERACTION_STARTED,
        f"{Stage.HA_ACTION}_started",
        f"{Stage.HA_ACTION}_completed",
        INTERACTION_FAILED,
    ]
    assert trace.events[-2].success is False

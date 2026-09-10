"""Tests for satellite-side interaction tracing."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from .tracing import (
    AUDIO_UPLOAD,
    INTERACTION_COMPLETED,
    INTERACTION_FAILED,
    PLAYBACK,
    WAKE_DETECTED,
    SatelliteTracer,
)


class _Recorder:
    """Collect emitted trace events instead of logging them."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    @property
    def stages(self) -> list[str]:
        return [event["stage"] for event in self.events]


def test_wake_creates_a_trace_id() -> None:
    """The satellite mints its own id for the interaction it observes."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    trace = tracer.wake("SaySo")

    assert trace.trace_id
    assert recorder.stages == [WAKE_DETECTED]
    assert recorder.events[0]["trace_id"] == trace.trace_id
    assert recorder.events[0]["metadata"] == {"phrase": "SaySo"}


def test_each_interaction_gets_a_distinct_id() -> None:
    """Consecutive wakes never reuse a trace id."""
    tracer = SatelliteTracer(emit=lambda event: None)

    first = tracer.wake()
    tracer.finish()
    second = tracer.wake()

    assert first.trace_id != second.trace_id


def test_satellite_reports_only_the_stages_it_owns() -> None:
    """Wake, audio upload and playback - never STT or TTS."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.wake("SaySo")
    tracer.upload_started()
    tracer.upload_completed()
    tracer.playback_started()
    tracer.playback_completed()
    tracer.finish()

    assert recorder.stages == [
        WAKE_DETECTED,
        f"{AUDIO_UPLOAD}_started",
        f"{AUDIO_UPLOAD}_completed",
        f"{PLAYBACK}_started",
        f"{PLAYBACK}_completed",
        INTERACTION_COMPLETED,
    ]
    assert not any("stt" in stage or "tts" in stage for stage in recorder.stages)


def test_one_trace_id_across_every_satellite_stage() -> None:
    """The id is never regenerated for a downstream stage."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    trace = tracer.wake()
    tracer.upload_started()
    tracer.upload_completed()
    tracer.playback_started()
    tracer.playback_completed()
    tracer.finish()

    assert {event["trace_id"] for event in recorder.events} == {trace.trace_id}
    assert [event["sequence"] for event in recorder.events] == list(
        range(len(recorder.events))
    )


def test_stages_record_durations_and_elapsed_time() -> None:
    """Every completed stage carries a duration and an elapsed offset."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.wake()
    tracer.upload_started()
    tracer.upload_completed()

    completed = recorder.events[-1]
    assert completed["duration_ms"] is not None
    assert completed["duration_ms"] >= 0
    assert completed["elapsed_ms"] >= completed["duration_ms"]
    assert completed["success"] is True


def test_home_assistant_conversation_id_is_attached() -> None:
    """The HA conversation id joins satellite timings to the canonical trace."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.wake()
    tracer.set_conversation_id("01HACONVERSATION")
    tracer.playback_started()
    tracer.playback_completed()
    tracer.finish()

    assert recorder.events[0].get("conversation_id") is None
    assert all(
        event["conversation_id"] == "01HACONVERSATION"
        for event in recorder.events[1:]
    )


def test_playback_started_is_idempotent() -> None:
    """Early TTS streaming and tts-end must not open playback twice."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.wake()
    tracer.playback_started()
    tracer.playback_started()
    tracer.playback_completed()

    assert recorder.stages.count(f"{PLAYBACK}_started") == 1


def test_failure_closes_open_stages_and_finalizes() -> None:
    """A failure preserves timings collected before it."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.wake()
    tracer.upload_started()
    tracer.fail("stt", "stt_failed")
    tracer.finish()

    assert recorder.stages == [
        WAKE_DETECTED,
        f"{AUDIO_UPLOAD}_started",
        f"{AUDIO_UPLOAD}_completed",
        INTERACTION_FAILED,
    ]
    assert recorder.events[-2]["success"] is False
    assert recorder.events[-1]["metadata"] == {
        "error_stage": "stt",
        "error_type": "stt_failed",
    }


def test_first_failure_wins() -> None:
    """A later generic failure cannot overwrite the precise one."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.wake()
    tracer.fail("stt", "stt_failed")
    tracer.fail("playback", "playback_interrupted")
    tracer.finish()

    assert recorder.events[-1]["metadata"]["error_stage"] == "stt"


def test_stage_events_outside_an_interaction_are_ignored() -> None:
    """Playback of a chime or timer must not invent an interaction."""
    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)

    tracer.upload_started()
    tracer.playback_started()
    tracer.playback_completed()
    tracer.finish()

    assert recorder.events == []


def test_emit_failure_does_not_break_the_voice_path() -> None:
    """An observability error is never raised into the interaction."""

    def _broken(event: dict[str, Any]) -> None:
        raise OSError("log volume full")

    tracer = SatelliteTracer(emit=_broken)

    tracer.wake()
    tracer.upload_started()
    tracer.upload_completed()
    tracer.finish()

    assert tracer.current is None


def test_finish_clears_the_in_flight_interaction() -> None:
    """The tracer holds at most one interaction, matching the satellite."""
    tracer = SatelliteTracer(emit=lambda event: None)

    tracer.wake()
    assert tracer.current is not None
    tracer.finish()
    assert tracer.current is None


# --- Wiring into the patched LVA voice handlers ---------------------------


class _EventType:
    """Minimal stand-in for aioesphomeapi's voice event enum."""

    VOICE_ASSISTANT_STT_END = 1
    VOICE_ASSISTANT_ERROR = 2
    VOICE_ASSISTANT_RUN_START = 3
    VOICE_ASSISTANT_STT_VAD_END = 4
    VOICE_ASSISTANT_INTENT_END = 5
    VOICE_ASSISTANT_INTENT_PROGRESS = 6
    VOICE_ASSISTANT_TTS_END = 7


class _Player:
    def play(self, path: str, done_callback: Any = None) -> None:
        if done_callback is not None:
            done_callback()

    def stop(self) -> None:
        pass


class _State:
    def __init__(self) -> None:
        self.muted = False
        self.tts_player = _Player()


class _Protocol:
    """Stand-in for LVA's VoiceSatelliteProtocol."""

    def __init__(self) -> None:
        self.state = _State()
        self._pipeline_active = False
        self._timer_finished = False
        self._timer_ring_start = None
        self._tts_played = False
        self.streaming_started = False

    def handle_voice_event(self, event_type: Any, data: dict) -> None:
        pass

    def _tts_finished(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _emit(self, event: Any, data: Any = None) -> None:
        pass

    def duck(self) -> None:
        pass

    def _start_audio_streaming(self, phrase: str) -> None:
        self.streaming_started = True


class _Wake:
    wake_word = "SaySo"


@pytest.fixture
def patched_protocol(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install the SaySo voice handlers onto a stand-in protocol class."""
    model = types.ModuleType("aioesphomeapi.model")
    model.VoiceAssistantEventType = _EventType  # type: ignore[attr-defined]
    aioesphomeapi = types.ModuleType("aioesphomeapi")
    aioesphomeapi.model = model  # type: ignore[attr-defined]
    events_module = types.ModuleType("linux_voice_assistant.events")
    events_module.LVAEvent = types.SimpleNamespace(  # type: ignore[attr-defined]
        WAKE_WORD_DETECTED="wake",
        LISTENING="listening",
        PIPELINE_ERROR="error",
        IDLE="idle",
    )
    lva = types.ModuleType("linux_voice_assistant")
    monkeypatch.setitem(sys.modules, "aioesphomeapi", aioesphomeapi)
    monkeypatch.setitem(sys.modules, "aioesphomeapi.model", model)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", lva)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.events", events_module)
    return _Protocol


def _sounds(tmp_path: Any) -> Any:
    from .config import SoundsCfg

    return SoundsCfg(
        wake=tmp_path / "wake.wav",
        failure=tmp_path / "failure.wav",
        unavailable=tmp_path / "unavailable.wav",
    )


def test_voice_handlers_trace_a_full_satellite_cycle(
    patched_protocol: Any, tmp_path: Any
) -> None:
    """Wake through playback produces one chronological satellite trace."""
    from .events import install_voice_handlers

    recorder = _Recorder()
    tracer = SatelliteTracer(emit=recorder)
    install_voice_handlers(patched_protocol, _sounds(tmp_path), None, tracer)
    satellite = patched_protocol()

    satellite.wakeup(_Wake())
    patched_protocol.handle_voice_event(
        satellite, _EventType.VOICE_ASSISTANT_STT_END, {"text": "turn on the light"}
    )
    patched_protocol.handle_voice_event(
        satellite,
        _EventType.VOICE_ASSISTANT_INTENT_END,
        {"conversation_id": "01HACONVERSATION"},
    )
    satellite._tts_played = True
    patched_protocol.handle_voice_event(
        satellite, _EventType.VOICE_ASSISTANT_TTS_END, {"url": "http://x/y.wav"}
    )
    patched_protocol._tts_finished(satellite)

    assert recorder.stages == [
        WAKE_DETECTED,
        f"{AUDIO_UPLOAD}_started",
        f"{AUDIO_UPLOAD}_completed",
        f"{PLAYBACK}_started",
        f"{PLAYBACK}_completed",
        INTERACTION_COMPLETED,
    ]
    assert recorder.events[-1]["conversation_id"] == "01HACONVERSATION"
    assert len({event["trace_id"] for event in recorder.events}) == 1


def test_voice_handlers_trace_a_pipeline_error(
    patched_protocol: Any, tmp_path: Any
) -> None:
    """A pipeline error fails the satellite trace at the pipeline stage."""
    from .events import install_voice_handlers

    recorder = _Recorder()
    install_voice_handlers(
        patched_protocol, _sounds(tmp_path), None, SatelliteTracer(emit=recorder)
    )
    satellite = patched_protocol()

    satellite.wakeup(_Wake())
    patched_protocol.handle_voice_event(
        satellite, _EventType.VOICE_ASSISTANT_ERROR, {}
    )
    patched_protocol._tts_finished(satellite)

    assert recorder.stages[-1] == INTERACTION_FAILED
    assert recorder.events[-1]["metadata"]["error_type"] == "pipeline_error"


def test_voice_handlers_trace_an_empty_transcript(
    patched_protocol: Any, tmp_path: Any
) -> None:
    """An empty STT result is recorded as an STT failure, not a success."""
    from .events import install_voice_handlers

    recorder = _Recorder()
    install_voice_handlers(
        patched_protocol, _sounds(tmp_path), None, SatelliteTracer(emit=recorder)
    )
    satellite = patched_protocol()

    satellite.wakeup(_Wake())
    patched_protocol.handle_voice_event(
        satellite, _EventType.VOICE_ASSISTANT_STT_END, {"text": "   "}
    )
    patched_protocol._tts_finished(satellite)

    assert recorder.stages[-1] == INTERACTION_FAILED
    assert recorder.events[-1]["metadata"] == {
        "error_stage": "stt",
        "error_type": "stt_failed",
    }

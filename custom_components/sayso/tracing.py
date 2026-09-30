
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from homeassistant.core import Context, HomeAssistant
from homeassistant.util import ulid as ulid_util

_LOGGER = logging.getLogger(__name__)

KEY_ASSIST_PIPELINE = "assist_pipeline"


class Stage(StrEnum):

    AUDIO_UPLOAD = "audio_upload"
    STT = "stt"
    SAYSO_REQUEST = "sayso_request"
    CONTEXT = "context"
    INFERENCE = "inference"
    TOOL_PARSE = "tool_parse"
    HA_ACTION = "ha_action"
    RESPONSE = "response"
    TTS = "tts"
    PLAYBACK = "playback"


INTERACTION_STARTED = "interaction_started"
INTERACTION_COMPLETED = "interaction_completed"
INTERACTION_FAILED = "interaction_failed"
WAKE_DETECTED = "wake_detected"
SPEECH_STARTED = "speech_started"
SPEECH_ENDED = "speech_ended"


class ErrorType(StrEnum):

    STT_FAILED = "stt_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_TIMEOUT = "model_timeout"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    EMPTY_RESPONSE = "empty_response"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE_TOOL = "unavailable_tool"
    ITERATION_LIMIT = "iteration_limit"
    HA_ACTION_FAILED = "ha_action_failed"
    TTS_FAILED = "tts_failed"
    PLAYBACK_FAILED = "playback_failed"
    PIPELINE_ERROR = "pipeline_error"
    UNKNOWN = "unknown"


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


@dataclass(slots=True)
class StageEvent:

    stage: str
    timestamp: str
    elapsed_ms: int
    success: bool = True
    duration_ms: int | None = None
    metadata: dict[str, Any] | None = None

    def as_dict(self, trace_id: str, sequence: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trace_id": trace_id,
            "sequence": sequence,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
            "duration_ms": self.duration_ms,
            "success": self.success,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


@dataclass
class StageSpan:

    stage: str
    started: float
    metadata: dict[str, Any] = field(default_factory=dict)
    closed: bool = False


@dataclass
class TraceContext:

    trace_id: str
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=_now)
    events: list[StageEvent] = field(default_factory=list)
    stage_ms: dict[str, int] = field(default_factory=dict)
    completed_at: datetime | None = None
    success: bool | None = None
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    utterance: str | None = None
    tool: str | None = None
    domain: str | None = None
    service: str | None = None
    target: str | None = None
    _open: list[StageSpan] = field(default_factory=list, repr=False)

    def elapsed_ms(self) -> int:
        return _ms(time.monotonic() - self.started_monotonic)

    def rebase(self, started_at: datetime, age_seconds: float) -> None:
        self.started_at = started_at
        self.started_monotonic = time.monotonic() - max(age_seconds, 0.0)

    def add_event(
        self,
        stage: str,
        *,
        success: bool = True,
        duration_ms: int | None = None,
        elapsed_ms: int | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StageEvent:
        event = StageEvent(
            stage=stage,
            timestamp=(timestamp or _now()).isoformat(),
            elapsed_ms=self.elapsed_ms() if elapsed_ms is None else elapsed_ms,
            success=success,
            duration_ms=duration_ms,
            metadata=metadata or None,
        )
        self.events.append(event)
        _LOGGER.debug(
            "trace_id=%s %s elapsed_ms=%d duration_ms=%s success=%s",
            self.trace_id,
            stage,
            event.elapsed_ms,
            event.duration_ms,
            event.success,
        )
        return event

    def record_span(
        self,
        stage: Stage | str,
        *,
        duration_ms: int,
        started_at: datetime | None = None,
        started_elapsed_ms: int | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        name = str(stage)
        completed_at = (
            started_at + timedelta(milliseconds=duration_ms)
            if started_at is not None
            else None
        )
        self.add_event(
            f"{name}_started",
            elapsed_ms=started_elapsed_ms,
            timestamp=started_at,
            metadata=metadata,
        )
        self.add_event(
            f"{name}_completed",
            elapsed_ms=(
                None
                if started_elapsed_ms is None
                else started_elapsed_ms + duration_ms
            ),
            timestamp=completed_at,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata,
        )
        self.stage_ms[name] = self.stage_ms.get(name, 0) + duration_ms

    def open(self, stage: Stage | str, **metadata: Any) -> StageSpan:
        span = StageSpan(
            stage=str(stage), started=time.monotonic(), metadata=dict(metadata)
        )
        self._open.append(span)
        self.add_event(f"{span.stage}_started", metadata=dict(span.metadata) or None)
        return span

    def close(self, span: StageSpan, *, success: bool = True) -> None:
        if span.closed:
            return
        span.closed = True
        if span in self._open:
            self._open.remove(span)
        duration = _ms(time.monotonic() - span.started)
        self.stage_ms[span.stage] = self.stage_ms.get(span.stage, 0) + duration
        self.add_event(
            f"{span.stage}_completed",
            success=success,
            duration_ms=duration,
            metadata=span.metadata or None,
        )

    @contextmanager
    def stage(
        self,
        stage: Stage | str,
        *,
        error_type: ErrorType = ErrorType.UNKNOWN,
        **metadata: Any,
    ) -> Iterator[StageSpan]:
        span = self.open(stage, **metadata)
        try:
            yield span
        except BaseException as err:
            self.close(span, success=False)
            self.fail(span.stage, error_type, type(err).__name__)
            raise
        self.close(span)

    def fail(
        self,
        stage: Stage | str,
        error_type: ErrorType | str,
        message: str | None = None,
    ) -> None:
        if self.success is False:
            return
        for span in reversed(list(self._open)):
            self.close(span, success=False)
        self.success = False
        self.error_stage = str(stage)
        self.error_type = str(error_type)
        self.error_message = (message or "")[:200] or None
        _LOGGER.debug(
            "trace_id=%s failed stage=%s error_type=%s",
            self.trace_id,
            self.error_stage,
            self.error_type,
        )

    def finish(self) -> None:
        if self.completed_at is not None:
            return
        self.completed_at = _now()
        if self.success is False:
            self.add_event(
                INTERACTION_FAILED,
                success=False,
                metadata={
                    "error_stage": self.error_stage,
                    "error_type": self.error_type,
                },
            )
            return
        self.success = True
        self.add_event(INTERACTION_COMPLETED)

    def total_ms(self) -> int:
        return self.events[-1].elapsed_ms if self.events else self.elapsed_ms()

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "success": self.success,
            "utterance": self.utterance,
            "tool": self.tool,
            "domain": self.domain,
            "service": self.service,
            "target": self.target,
            "wake_ms": self.stage_ms.get("wake"),
            "audio_upload_ms": self.stage_ms.get(Stage.AUDIO_UPLOAD),
            "stt_ms": self.stage_ms.get(Stage.STT),
            "context_ms": self.stage_ms.get(Stage.CONTEXT),
            "model_ms": self.stage_ms.get(Stage.INFERENCE),
            "tool_parse_ms": self.stage_ms.get(Stage.TOOL_PARSE),
            "ha_ms": self.stage_ms.get(Stage.HA_ACTION),
            "response_ms": self.stage_ms.get(Stage.RESPONSE),
            "tts_ms": self.stage_ms.get(Stage.TTS),
            "playback_ms": self.stage_ms.get(Stage.PLAYBACK),
            "total_ms": self.total_ms(),
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    def stage_records(self) -> list[dict[str, Any]]:
        return [
            event.as_dict(self.trace_id, sequence)
            for sequence, event in enumerate(self.events)
        ]


def async_create_trace(
    hass: HomeAssistant, context: Context
) -> tuple[TraceContext, Any | None]:
    trace = TraceContext(trace_id=ulid_util.ulid_now())
    trace.add_event(INTERACTION_STARTED)
    run: Any | None = None
    try:
        run = async_find_pipeline_run(hass, context)
        if run is not None:
            adopt_pipeline_run(hass, trace, run)
    except Exception:
        _LOGGER.debug("SaySo could not adopt the pipeline run", exc_info=True)
        run = None
    return trace, run


def async_find_pipeline_run(hass: HomeAssistant, context: Context) -> Any | None:
    pipeline_data = hass.data.get(KEY_ASSIST_PIPELINE)
    if pipeline_data is None:
        return None
    active = getattr(
        getattr(pipeline_data, "pipeline_runs", None), "_pipeline_runs", None
    )
    if not active:
        return None
    for runs in list(active.values()):
        for run in list(runs.values()):
            if getattr(run, "context", None) is context:
                return run
    return None


def _pipeline_events(hass: HomeAssistant, run: Any) -> list[Any]:
    try:
        debug = hass.data[KEY_ASSIST_PIPELINE].pipeline_debug
        return list(debug[run.pipeline.id][run.id].events)
    except (KeyError, AttributeError):
        return []


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def adopt_pipeline_run(hass: HomeAssistant, trace: TraceContext, run: Any) -> None:
    run_id = getattr(run, "id", None)
    if isinstance(run_id, str) and run_id:
        trace.trace_id = run_id
    replay_pipeline_events(trace, _pipeline_events(hass, run))


def replay_pipeline_events(trace: TraceContext, events: list[Any]) -> None:
    seen: dict[str, tuple[datetime, dict[str, Any] | None]] = {}
    for event in events:
        timestamp = _parse_ts(getattr(event, "timestamp", None))
        if timestamp is None:
            continue
        seen[str(getattr(event, "type", ""))] = (timestamp, getattr(event, "data", None))

    run_start = seen.get("run-start")
    if run_start is not None:
        trace.rebase(run_start[0], (_now() - run_start[0]).total_seconds())

    def elapsed(at: datetime) -> int:
        return max(_ms((at - trace.started_at).total_seconds()), 0)

    if (wake_end := seen.get("wake_word-end")) is not None:
        trace.add_event(
            WAKE_DETECTED, timestamp=wake_end[0], elapsed_ms=elapsed(wake_end[0])
        )

    stt_start = seen.get("stt-start")
    vad_start = seen.get("stt-vad-start")
    vad_end = seen.get("stt-vad-end")
    stt_end = seen.get("stt-end")

    if vad_start is not None:
        trace.add_event(
            SPEECH_STARTED, timestamp=vad_start[0], elapsed_ms=elapsed(vad_start[0])
        )
    if vad_end is not None:
        trace.add_event(
            SPEECH_ENDED, timestamp=vad_end[0], elapsed_ms=elapsed(vad_end[0])
        )

    if stt_start is not None and vad_end is not None:
        trace.record_span(
            Stage.AUDIO_UPLOAD,
            duration_ms=max(_ms((vad_end[0] - stt_start[0]).total_seconds()), 0),
            started_at=stt_start[0],
            started_elapsed_ms=elapsed(stt_start[0]),
        )

    if stt_end is None:
        return

    transcription_start = vad_end or stt_start
    if transcription_start is None:
        return

    metadata: dict[str, Any] = {}
    if vad_end is None:
        metadata["vad"] = False
    text = ""
    data = stt_end[1]
    if isinstance(data, dict):
        output = data.get("stt_output")
        if isinstance(output, dict) and isinstance(output.get("text"), str):
            text = output["text"]
    metadata["text_length"] = len(text)
    trace.utterance = text or trace.utterance

    trace.record_span(
        Stage.STT,
        duration_ms=max(
            _ms((stt_end[0] - transcription_start[0]).total_seconds()), 0
        ),
        started_at=transcription_start[0],
        started_elapsed_ms=elapsed(transcription_start[0]),
        success=bool(text.strip()),
        metadata=metadata,
    )
    if not text.strip():
        trace.fail(Stage.STT, ErrorType.STT_FAILED, "empty transcript")


def observe_pipeline_completion(
    trace: TraceContext,
    run: Any,
    on_complete: Callable[[TraceContext], None],
) -> Callable[[], None]:
    original = run.event_callback
    tts_started: dict[str, datetime] = {}
    detached = False

    def detach() -> None:
        nonlocal detached
        if detached:
            return
        detached = True
        if run.event_callback is observer:
            run.event_callback = original

    def record(event: Any) -> None:
        event_type = str(getattr(event, "type", ""))
        timestamp = _parse_ts(getattr(event, "timestamp", None)) or _now()
        data = getattr(event, "data", None)

        if event_type == "tts-start":
            tts_started["at"] = timestamp
            return
        if event_type == "tts-end":
            started = tts_started.pop("at", timestamp)
            metadata: dict[str, Any] = {}
            if isinstance(data, dict) and isinstance(data.get("tts_output"), dict):
                metadata["stream"] = True
            trace.record_span(
                Stage.TTS,
                duration_ms=max(_ms((timestamp - started).total_seconds()), 0),
                started_at=started,
                started_elapsed_ms=max(
                    _ms((started - trace.started_at).total_seconds()), 0
                ),
                metadata=metadata or None,
            )
            return
        if event_type == "error":
            code = message = ""
            if isinstance(data, dict):
                code = str(data.get("code") or "")
                message = str(data.get("message") or "")
            if "stt" in code:
                stage, error_type = Stage.STT, ErrorType.STT_FAILED
            elif "tts" in code:
                stage, error_type = Stage.TTS, ErrorType.TTS_FAILED
            else:
                stage, error_type = Stage.SAYSO_REQUEST, ErrorType.PIPELINE_ERROR
            trace.fail(stage, error_type, message or code)
            return
        if event_type == "run-end":
            detach()
            trace.finish()
            on_complete(trace)

    def observer(event: Any) -> None:
        try:
            record(event)
        except Exception:
            _LOGGER.debug("SaySo trace observer failed", exc_info=True)
        original(event)

    run.event_callback = observer
    return detach


from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)

_TRACE_LOGGER = logging.getLogger("sayso.trace")

WAKE_DETECTED = "wake_detected"
AUDIO_UPLOAD = "audio_upload"
PLAYBACK = "playback"
INTERACTION_COMPLETED = "interaction_completed"
INTERACTION_FAILED = "interaction_failed"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


@dataclass
class SatelliteTrace:

    trace_id: str
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=_now)
    events: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    success: bool | None = None
    error_stage: str | None = None
    error_type: str | None = None
    _open: dict[str, float] = field(default_factory=dict, repr=False)

    def elapsed_ms(self) -> int:
        return _ms(time.monotonic() - self.started_monotonic)

    def event(
        self,
        stage: str,
        *,
        duration_ms: int | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trace_id": self.trace_id,
            "sequence": len(self.events),
            "stage": stage,
            "timestamp": _now().isoformat(),
            "elapsed_ms": self.elapsed_ms(),
            "duration_ms": duration_ms,
            "success": success,
        }
        if self.conversation_id:
            record["conversation_id"] = self.conversation_id
        if metadata:
            record["metadata"] = metadata
        self.events.append(record)
        return record

    def begin(self, stage: str, **metadata: Any) -> dict[str, Any]:
        self._open[stage] = time.monotonic()
        return self.event(f"{stage}_started", metadata=metadata or None)

    def end(
        self, stage: str, *, success: bool = True, **metadata: Any
    ) -> dict[str, Any] | None:
        started = self._open.pop(stage, None)
        if started is None:
            return None
        return self.event(
            f"{stage}_completed",
            duration_ms=_ms(time.monotonic() - started),
            success=success,
            metadata=metadata or None,
        )

    def fail(self, stage: str, error_type: str) -> None:
        if self.success is False:
            return
        for open_stage in list(self._open):
            self.end(open_stage, success=False)
        self.success = False
        self.error_stage = stage
        self.error_type = error_type

    def finish(self) -> dict[str, Any]:
        for open_stage in list(self._open):
            self.end(open_stage, success=self.success is not False)
        if self.success is False:
            return self.event(
                INTERACTION_FAILED,
                success=False,
                metadata={"error_stage": self.error_stage, "error_type": self.error_type},
            )
        self.success = True
        return self.event(INTERACTION_COMPLETED)


def _log_event(event: dict[str, Any]) -> None:
    _TRACE_LOGGER.info("%s", json.dumps(event, sort_keys=True))


class SatelliteTracer:

    def __init__(self, emit: Callable[[dict[str, Any]], None] = _log_event) -> None:
        self._emit = emit
        self.current: SatelliteTrace | None = None
        self._published = 0

    def _drain(self) -> None:
        trace = self.current
        if trace is None:
            return
        while self._published < len(trace.events):
            event = trace.events[self._published]
            self._published += 1
            try:
                self._emit(event)
            except Exception:
                _LOGGER.debug("SaySo could not emit a trace event", exc_info=True)

    def wake(self, phrase: str | None = None) -> SatelliteTrace:
        trace = SatelliteTrace(trace_id=uuid4().hex)
        self.current = trace
        self._published = 0
        trace.event(WAKE_DETECTED, metadata={"phrase": phrase} if phrase else None)
        self._drain()
        return trace

    def upload_started(self) -> None:
        if self.current is not None:
            self.current.begin(AUDIO_UPLOAD)
            self._drain()

    def upload_completed(self) -> None:
        if self.current is not None:
            self.current.end(AUDIO_UPLOAD)
            self._drain()

    def playback_started(self) -> None:
        trace = self.current
        if trace is None or PLAYBACK in trace._open:
            return
        trace.begin(PLAYBACK)
        self._drain()

    def playback_completed(self, *, success: bool = True) -> None:
        if self.current is not None:
            self.current.end(PLAYBACK, success=success)
            self._drain()

    def set_conversation_id(self, conversation_id: str | None) -> None:
        if self.current is not None and conversation_id:
            self.current.conversation_id = conversation_id

    def fail(self, stage: str, error_type: str) -> None:
        if self.current is not None:
            self.current.fail(stage, error_type)
            self._drain()

    def finish(self) -> None:
        trace = self.current
        if trace is None:
            return
        trace.finish()
        self._drain()
        self.current = None

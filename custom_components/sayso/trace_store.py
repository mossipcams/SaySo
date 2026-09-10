"""Persistence and lifecycle for SaySo interaction traces.

Traces are kept out of ``home-assistant.log`` and stored as structured records
in Home Assistant's own ``helpers.storage.Store``. That is the simplest native
persistence already appropriate to a custom integration: no extra service, no
external database, and writes are coalesced off the request path.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store
from homeassistant.util import ulid as ulid_util

from .const import (
    DEFAULT_TRACE_MAX_INTERACTIONS,
    DEFAULT_TRACE_RETENTION_DAYS,
    DOMAIN,
)
from .tracing import (
    ErrorType,
    Stage,
    TraceContext,
    async_create_trace,
    observe_pipeline_completion,
)

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.traces"

# Writes are coalesced: a burst of interactions produces one file write.
SAVE_DELAY = 60

# A pipeline run that never emits ``run-end`` must not strand its trace.
PIPELINE_FINALIZE_TIMEOUT = 120


class TraceStore:
    """In-memory trace ring buffer backed by a delayed Home Assistant Store."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        retention_days: int = DEFAULT_TRACE_RETENTION_DAYS,
        max_interactions: int = DEFAULT_TRACE_MAX_INTERACTIONS,
        store_utterances: bool = True,
    ) -> None:
        """Initialize the trace store."""
        self.hass = hass
        self.retention_days = retention_days
        self.store_utterances = store_utterances
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, STORAGE_KEY, private=True
        )
        # maxlen enforces the interaction cap for free on every append.
        self._records: deque[dict[str, Any]] = deque(maxlen=max_interactions)

    async def async_load(self) -> None:
        """Load persisted traces. A corrupt or missing store starts empty."""
        try:
            data = await self._store.async_load()
        except Exception:  # noqa: BLE001 - never block setup on observability
            _LOGGER.warning("SaySo could not load stored traces", exc_info=True)
            return
        if not isinstance(data, dict):
            return
        summaries = data.get("interactions")
        events = data.get("events")
        if not isinstance(summaries, list):
            return
        by_trace: dict[str, list[dict[str, Any]]] = {}
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict) and isinstance(event.get("trace_id"), str):
                    by_trace.setdefault(event["trace_id"], []).append(event)
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            trace_id = summary.get("trace_id")
            self._records.append(
                {
                    "summary": summary,
                    "events": by_trace.get(trace_id, []) if trace_id else [],
                }
            )

    @callback
    def async_record(self, trace: TraceContext) -> None:
        """Append a finished trace and schedule a delayed write.

        Retention pruning happens when the write runs, never on this path.
        """
        try:
            summary = trace.summary()
            if not self.store_utterances:
                summary["utterance"] = None
            self._records.append(
                {"summary": summary, "events": trace.stage_records()}
            )
            self._store.async_delay_save(self._data_to_save, SAVE_DELAY)
        except Exception:  # noqa: BLE001 - an observability problem is not a
            # voice failure; log it and let the interaction stand.
            _LOGGER.warning(
                "SaySo could not record trace_id=%s", trace.trace_id, exc_info=True
            )

    @callback
    def _data_to_save(self) -> dict[str, Any]:
        """Prune expired traces and split them into the two stored structures."""
        cutoff = datetime.now(tz=UTC) - timedelta(days=self.retention_days)
        kept = [
            record
            for record in self._records
            if _started_at(record) is None or _started_at(record) >= cutoff
        ]
        if len(kept) != len(self._records):
            self._records = deque(kept, maxlen=self._records.maxlen)
        return {
            "interactions": [record["summary"] for record in self._records],
            "events": [
                event for record in self._records for event in record["events"]
            ],
        }

    @callback
    def async_prune(self) -> None:
        """Apply retention now. Used by the periodic cleanup and by tests."""
        self._data_to_save()
        self._store.async_delay_save(self._data_to_save, SAVE_DELAY)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return one full trace: summary plus chronological stage events."""
        for record in reversed(self._records):
            if record["summary"].get("trace_id") == trace_id:
                return {
                    "summary": record["summary"],
                    "events": list(record["events"]),
                }
        return None

    def query(
        self,
        *,
        limit: int = 50,
        only_failures: bool = False,
        error_stage: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching interaction summaries, most recent first."""
        # ponytail: linear scan over a capped deque. At the 5,000-interaction
        # ceiling this is well under a millisecond; move to SQLite only if the
        # cap is ever raised past that.
        results: list[dict[str, Any]] = []
        for record in reversed(self._records):
            summary = record["summary"]
            if only_failures and summary.get("success") is not False:
                continue
            if error_stage is not None and summary.get("error_stage") != error_stage:
                continue
            started = _started_at(record)
            if start_time is not None and (started is None or started < start_time):
                continue
            if end_time is not None and (started is None or started > end_time):
                continue
            results.append(summary)
            if len(results) >= limit:
                break
        return results

    def __len__(self) -> int:
        """Return the number of retained interactions."""
        return len(self._records)


def _started_at(record: dict[str, Any]) -> datetime | None:
    value = record["summary"].get("started_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class TraceRecorder:
    """Creates, tracks and persists one trace per interaction."""

    def __init__(self, hass: HomeAssistant, store: TraceStore) -> None:
        """Initialize the recorder."""
        self.hass = hass
        self.store = store
        # trace_id -> (trace, detach, cancel timeout) for traces still awaiting
        # Home Assistant's TTS stage. Keyed per interaction so concurrent
        # requests stay isolated.
        self._pending: dict[
            str, tuple[TraceContext, Callable[[], None], Callable[[], None]]
        ] = {}

    @callback
    def async_start(self, context: Any, utterance: str | None = None) -> TraceContext:
        """Begin a trace for one interaction. Never raises."""
        try:
            trace, run = async_create_trace(self.hass, context)
        except Exception:  # noqa: BLE001 - tracing must never break the request
            _LOGGER.debug("SaySo could not start a trace", exc_info=True)
            return TraceContext(trace_id=ulid_util.ulid_now())
        if utterance:
            trace.utterance = utterance
        if run is not None:
            self._async_defer_to_pipeline(trace, run)
        return trace

    def _async_defer_to_pipeline(self, trace: TraceContext, run: Any) -> None:
        """Hold the trace open until Home Assistant finishes TTS for this run."""
        try:
            detach = observe_pipeline_completion(trace, run, self._async_complete)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("SaySo could not observe the pipeline run", exc_info=True)
            return

        @callback
        def _timeout(_now: datetime) -> None:
            if trace.trace_id in self._pending:
                _LOGGER.debug(
                    "trace_id=%s pipeline never reported run-end; persisting early",
                    trace.trace_id,
                )
                self._async_complete(trace)

        cancel = async_call_later(self.hass, PIPELINE_FINALIZE_TIMEOUT, _timeout)
        self._pending[trace.trace_id] = (trace, detach, cancel)

    @callback
    def async_finish(self, trace: TraceContext) -> None:
        """Called when the conversation turn ends.

        When a voice pipeline owns this interaction the trace stays open until
        TTS completes; otherwise it is persisted now.
        """
        if trace.trace_id in self._pending:
            return
        trace.finish()
        self.store.async_record(trace)

    @callback
    def _async_complete(self, trace: TraceContext) -> None:
        """Detach the pipeline observer and persist the finished trace."""
        pending = self._pending.pop(trace.trace_id, None)
        if pending is not None:
            for teardown in pending[1:]:
                try:
                    teardown()
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("SaySo trace teardown failed", exc_info=True)
        trace.finish()
        self.store.async_record(trace)

    @callback
    def async_shutdown(self) -> None:
        """Persist and detach every trace still waiting on a pipeline run."""
        for trace_id in list(self._pending):
            pending = self._pending.get(trace_id)
            if pending is not None:
                self._async_complete(pending[0])


__all__ = [
    "ErrorType",
    "Stage",
    "TraceContext",
    "TraceRecorder",
    "TraceStore",
]

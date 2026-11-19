from __future__ import annotations

from datetime import UTC, datetime

from homeassistant.core import HomeAssistant

from custom_components.sayso.const import DEFAULT_TRACE_STORE_UTTERANCES
from custom_components.sayso.trace_store import TraceStore
from custom_components.sayso.tracing import INTERACTION_STARTED, TraceContext


def test_trace_store_utterances_are_off_by_default() -> None:
    assert DEFAULT_TRACE_STORE_UTTERANCES is False


async def test_trace_store_defaults_do_not_record_the_utterance(
    hass: HomeAssistant,
) -> None:
    trace = TraceContext(trace_id="default", started_at=datetime.now(tz=UTC))
    trace.utterance = "turn on the kitchen light"
    trace.add_event(INTERACTION_STARTED)
    trace.finish()

    store = TraceStore(hass)
    store.async_record(trace)

    summary = store.get("default")["summary"]
    assert summary["utterance"] is None

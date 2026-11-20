from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.sayso.client import LlamaCppClient
from custom_components.sayso.const import (
    CONF_TRACE_EXPORT,
    CONF_TRACE_EXPORT_ENDPOINT,
    DEFAULT_TRACE_EXPORT,
    DOMAIN,
    TRACE_EXPORT_JSONL,
    TRACE_EXPORT_OFF,
    TRACE_EXPORT_OTLP_HTTP,
)
from custom_components.sayso.trace_export import (
    JSONL_FILENAME,
    JSONL_ROTATED_FILENAME,
    TraceExporter,
    build_trace_exporter,
    otlp_payload,
)
from custom_components.sayso.trace_store import TraceStore
from custom_components.sayso.tracing import INTERACTION_STARTED, TraceContext
from tests.test_config_flow import (
    BASE_URL,
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)

TRACE_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
SECOND_TRACE_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
ENDPOINT = "http://collector:4318"
EXPORT_URL = f"{ENDPOINT}/v1/traces"
UTTERANCE = "turn on the secret lamp"
LOGGER_NAME = "custom_components.sayso.trace_export"


@pytest.fixture(autouse=True)
def config_dir(hass: HomeAssistant, tmp_path: Path) -> Path:
    hass.config.config_dir = str(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
async def setup_llm(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "llm", {})


@pytest.fixture
def mock_list_models() -> Any:
    with patch.object(
        LlamaCppClient, "list_models", new=AsyncMock(return_value=[MODEL_ID])
    ) as mocked:
        yield mocked


@pytest.fixture
def mock_validate_model() -> Any:
    with patch.object(
        LlamaCppClient, "validate_model", new=AsyncMock(return_value=None)
    ) as mocked:
        yield mocked


def _trace(trace_id: str, utterance: str | None = None) -> TraceContext:
    trace = TraceContext(trace_id=trace_id, started_at=datetime.now(tz=UTC))
    trace.utterance = utterance
    trace.add_event(INTERACTION_STARTED)
    trace.add_event(
        "inference_completed",
        duration_ms=12,
        metadata={"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 42},
    )
    trace.finish()
    return trace


def _jsonl_exporter(hass: HomeAssistant) -> TraceExporter:
    exporter = build_trace_exporter(hass, {CONF_TRACE_EXPORT: TRACE_EXPORT_JSONL})
    assert exporter is not None
    return exporter


def _otlp_exporter(hass: HomeAssistant) -> TraceExporter:
    exporter = build_trace_exporter(
        hass,
        {
            CONF_TRACE_EXPORT: TRACE_EXPORT_OTLP_HTTP,
            CONF_TRACE_EXPORT_ENDPOINT: ENDPOINT,
        },
    )
    assert exporter is not None
    return exporter


def _jsonl_file(hass: HomeAssistant, *, rotated: bool = False) -> Path:
    name = JSONL_ROTATED_FILENAME if rotated else JSONL_FILENAME
    return Path(hass.config.path(name))


def _jsonl_exists(hass: HomeAssistant) -> bool:
    return _jsonl_file(hass).exists()


def _jsonl_text(hass: HomeAssistant, *, rotated: bool = False) -> str:
    return _jsonl_file(hass, rotated=rotated).read_text()


def _read_jsonl(hass: HomeAssistant, *, rotated: bool = False) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in _jsonl_text(hass, rotated=rotated).splitlines()
    ]


def test_export_is_off_by_default(hass: HomeAssistant) -> None:
    assert DEFAULT_TRACE_EXPORT == TRACE_EXPORT_OFF
    assert build_trace_exporter(hass, {}) is None
    assert build_trace_exporter(hass, {CONF_TRACE_EXPORT: TRACE_EXPORT_OFF}) is None
    assert (
        build_trace_exporter(hass, {CONF_TRACE_EXPORT: TRACE_EXPORT_OTLP_HTTP}) is None
    )


async def test_off_writes_no_file_and_makes_no_http_call(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    store = TraceStore(hass)
    store.async_record(_trace(TRACE_ID, UTTERANCE))
    await hass.async_block_till_done(wait_background_tasks=True)

    assert not _jsonl_exists(hass)
    assert aioclient_mock.call_count == 0
    assert store.get(TRACE_ID) is not None


async def test_jsonl_writes_one_line_per_record(hass: HomeAssistant) -> None:
    store = TraceStore(hass, exporter=_jsonl_exporter(hass))
    store.async_record(_trace(TRACE_ID))
    store.async_record(_trace(SECOND_TRACE_ID))
    await hass.async_block_till_done(wait_background_tasks=True)

    records = _read_jsonl(hass)
    assert [record["summary"]["trace_id"] for record in records] == [
        TRACE_ID,
        SECOND_TRACE_ID,
    ]
    assert records[0]["events"] == store.get(TRACE_ID)["events"]


async def test_jsonl_rotates_once_at_the_size_cap(hass: HomeAssistant) -> None:
    store = TraceStore(hass, exporter=_jsonl_exporter(hass))
    third_trace_id = "01ARZ3NDEKTSV4RRFFQ69G5FAX"

    with patch("custom_components.sayso.trace_export.JSONL_MAX_BYTES", 1):
        for trace_id in (TRACE_ID, SECOND_TRACE_ID, third_trace_id):
            store.async_record(_trace(trace_id))
            await hass.async_block_till_done(wait_background_tasks=True)

    assert [record["summary"]["trace_id"] for record in _read_jsonl(hass)] == [
        third_trace_id
    ]
    assert [
        record["summary"]["trace_id"] for record in _read_jsonl(hass, rotated=True)
    ] == [SECOND_TRACE_ID]


async def test_utterance_is_absent_when_trace_store_utterances_is_off(
    hass: HomeAssistant,
) -> None:
    store = TraceStore(hass, store_utterances=False, exporter=_jsonl_exporter(hass))
    trace = _trace(TRACE_ID, UTTERANCE)
    store.async_record(trace)
    await hass.async_block_till_done(wait_background_tasks=True)

    assert "secret lamp" not in _jsonl_text(hass)
    assert _read_jsonl(hass)[0]["summary"]["utterance"] is None


async def test_otlp_payload_has_hex_ids_and_gen_ai_attributes(
    hass: HomeAssistant,
) -> None:
    store = TraceStore(hass)
    store.async_record(_trace(TRACE_ID))
    payload = otlp_payload(store.get(TRACE_ID))

    [resource] = payload["resourceSpans"]
    assert {"key": "service.name", "value": {"stringValue": "sayso"}} in resource[
        "resource"
    ]["attributes"]
    spans = resource["scopeSpans"][0]["spans"]
    root, *children = spans

    assert root["traceId"] == "01563e3ab5d3d6764c61efb99302bd5b"
    assert re.fullmatch(r"[0-9a-f]{32}", root["traceId"])
    for span in spans:
        assert re.fullmatch(r"[0-9a-f]{16}", span["spanId"])
        assert span["traceId"] == root["traceId"]
        assert int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"])

    [inference] = [span for span in children if span["name"] == "inference_completed"]
    assert inference["parentSpanId"] == root["spanId"]
    assert int(inference["endTimeUnixNano"]) - int(inference["startTimeUnixNano"]) == (
        12 * 1_000_000
    )
    assert {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}} in (
        inference["attributes"]
    )
    assert {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "42"}} in (
        inference["attributes"]
    )


async def test_otlp_export_posts_the_payload_to_the_endpoint(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(EXPORT_URL, status=200)
    store = TraceStore(hass, exporter=_otlp_exporter(hass))
    store.async_record(_trace(TRACE_ID))
    await hass.async_block_till_done(wait_background_tasks=True)

    assert aioclient_mock.call_count == 1
    assert str(aioclient_mock.mock_calls[0][1]) == EXPORT_URL


@pytest.mark.parametrize(
    "response",
    [
        {"exc": aiohttp.ClientConnectionError("collector is down")},
        {"status": 500},
    ],
)
async def test_failing_endpoint_is_logged_at_debug_and_dropped(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    caplog: pytest.LogCaptureFixture,
    response: dict[str, Any],
) -> None:
    aioclient_mock.post(EXPORT_URL, **response)
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    store = TraceStore(hass, exporter=_otlp_exporter(hass))

    store.async_record(_trace(TRACE_ID))
    await hass.async_block_till_done(wait_background_tasks=True)

    assert aioclient_mock.call_count == 1
    assert "SaySo trace export" in caplog.text
    assert "dropped" in caplog.text
    assert all(
        record.levelno == logging.DEBUG
        for record in caplog.records
        if record.name == LOGGER_NAME
    )
    assert store.get(TRACE_ID) is not None


@pytest.mark.parametrize(
    ("endpoint", "error"),
    [
        ("", "trace_export_endpoint_required"),
        ("ftp://collector", "invalid_trace_export_endpoint"),
    ],
)
async def test_options_flow_requires_a_valid_endpoint_for_otlp(
    hass: HomeAssistant,
    mock_list_models: Any,
    mock_validate_model: Any,
    endpoint: str,
    error: str,
) -> None:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"], base_url=BASE_URL)
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    await hass.async_block_till_done(wait_background_tasks=True)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM
    user_input = {
        **entry.options,
        CONF_TRACE_EXPORT: TRACE_EXPORT_OTLP_HTTP,
        CONF_TRACE_EXPORT_ENDPOINT: endpoint,
    }

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {CONF_TRACE_EXPORT_ENDPOINT: error}

    user_input[CONF_TRACE_EXPORT_ENDPOINT] = ENDPOINT
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.options[CONF_TRACE_EXPORT] == TRACE_EXPORT_OTLP_HTTP
    assert entry.options[CONF_TRACE_EXPORT_ENDPOINT] == ENDPOINT
    assert entry.state == config_entries.ConfigEntryState.LOADED

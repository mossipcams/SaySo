from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import ulid as ulid_util

from .const import (
    CONF_TRACE_EXPORT,
    CONF_TRACE_EXPORT_ENDPOINT,
    DEFAULT_TRACE_EXPORT,
    TRACE_EXPORT_JSONL,
    TRACE_EXPORT_OFF,
    TRACE_EXPORT_OTLP_HTTP,
)

_LOGGER = logging.getLogger(__name__)

JSONL_FILENAME = "sayso_traces.jsonl"
JSONL_ROTATED_FILENAME = "sayso_traces.jsonl.1"
JSONL_MAX_BYTES = 10 * 1024 * 1024
OTLP_TRACES_PATH = "/v1/traces"
OTLP_TIMEOUT = 10
SERVICE_NAME = "sayso"
ROOT_SPAN_NAME = "sayso.interaction"
ROOT_SEQUENCE = -1
SPAN_KIND_INTERNAL = 1
STATUS_CODE_ERROR = 2
NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MS = 1_000_000

_JSONL_FILE_LOCK = threading.Lock()


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def build_trace_exporter(
    hass: HomeAssistant, options: Mapping[str, Any]
) -> TraceExporter | None:
    mode = options.get(CONF_TRACE_EXPORT, DEFAULT_TRACE_EXPORT)
    endpoint = str(options.get(CONF_TRACE_EXPORT_ENDPOINT) or "").strip().rstrip("/")
    if mode == TRACE_EXPORT_JSONL or (mode == TRACE_EXPORT_OTLP_HTTP and endpoint):
        return TraceExporter(hass, mode, endpoint)
    return None


class TraceExporter:
    def __init__(self, hass: HomeAssistant, mode: str, endpoint: str) -> None:
        self.hass = hass
        self.mode = mode
        self.endpoint = endpoint
        self.path = Path(hass.config.path(JSONL_FILENAME))
        self._jsonl_lock = asyncio.Lock()

    @callback
    def async_export(self, record: dict[str, Any]) -> None:
        if self.mode == TRACE_EXPORT_JSONL:
            line = json.dumps(record, default=str)
            self.hass.async_create_background_task(
                self._async_write_line(line), "sayso trace export"
            )
        elif self.mode == TRACE_EXPORT_OTLP_HTTP:
            body = otlp_payload(record)
            self.hass.async_create_background_task(
                self._async_post_otlp(body), "sayso trace export"
            )

    async def _async_write_line(self, line: str) -> None:
        try:
            async with self._jsonl_lock:
                await self.hass.async_add_executor_job(_append_line, self.path, line)
        except OSError as err:
            _LOGGER.debug("SaySo trace export to JSONL failed: %s", err)

    async def _async_post_otlp(self, body: dict[str, Any]) -> None:
        session = async_get_clientsession(self.hass)
        try:
            async with session.post(
                f"{self.endpoint}{OTLP_TRACES_PATH}",
                json=body,
                timeout=aiohttp.ClientTimeout(total=OTLP_TIMEOUT),
            ) as response:
                if not 200 <= response.status < 300:
                    _LOGGER.debug(
                        "SaySo trace export got HTTP %s; dropped", response.status
                    )
        except (OSError, aiohttp.ClientError, TimeoutError) as err:
            _LOGGER.debug("SaySo trace export failed (%s); dropped", type(err).__name__)


def _append_line(path: Path, line: str) -> None:
    with _JSONL_FILE_LOCK:
        if path.exists() and path.stat().st_size >= JSONL_MAX_BYTES:
            path.replace(path.with_name(JSONL_ROTATED_FILENAME))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def otlp_payload(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    trace_id = _otlp_trace_id(str(summary["trace_id"]))
    root = _root_span(summary, trace_id)
    spans = [root]
    for event in record["events"]:
        span = _event_span(event, trace_id, root["spanId"])
        if span is not None:
            spans.append(span)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": SERVICE_NAME}}
                    ]
                },
                "scopeSpans": [{"scope": {"name": SERVICE_NAME}, "spans": spans}],
            }
        ]
    }


def _root_span(summary: dict[str, Any], trace_id: str) -> dict[str, Any]:
    start = _unix_nanos(summary.get("started_at")) or 0
    end = _unix_nanos(summary.get("completed_at"))
    if end is None:
        end = start + int(summary.get("total_ms") or 0) * NANOS_PER_MS
    return _span(
        trace_id,
        ROOT_SEQUENCE,
        ROOT_SPAN_NAME,
        start,
        end,
        {key: value for key, value in summary.items() if key != "trace_id"},
        success=summary.get("success") is not False,
    )


def _event_span(
    event: Mapping[str, Any], trace_id: str, parent_span_id: str
) -> dict[str, Any] | None:
    end = _unix_nanos(event.get("timestamp"))
    if end is None:
        return None
    duration = event.get("duration_ms")
    start = end - duration * NANOS_PER_MS if isinstance(duration, int) else end
    return _span(
        trace_id,
        int(event["sequence"]),
        str(event["stage"]),
        start,
        end,
        dict(event.get("metadata") or {}),
        success=bool(event.get("success", True)),
        parent_span_id=parent_span_id,
    )


def _span(
    trace_id: str,
    sequence: int,
    name: str,
    start: int,
    end: int,
    attributes: Mapping[str, Any],
    *,
    success: bool,
    parent_span_id: str | None = None,
) -> dict[str, Any]:
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": _span_id(trace_id, sequence),
        "name": name,
        "kind": SPAN_KIND_INTERNAL,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end),
        "attributes": [
            {"key": key, "value": _any_value(value)}
            for key, value in attributes.items()
            if value is not None
        ],
    }
    if parent_span_id is not None:
        span["parentSpanId"] = parent_span_id
    if not success:
        span["status"] = {"code": STATUS_CODE_ERROR}
    return span


def _otlp_trace_id(trace_id: str) -> str:
    try:
        return ulid_util.ulid_to_bytes(trace_id).hex()
    except ValueError:
        return hashlib.sha256(trace_id.encode()).hexdigest()[:32]


def _span_id(trace_id: str, sequence: int) -> str:
    return hashlib.sha256(f"{trace_id}:{sequence}".encode()).hexdigest()[:16]


def _unix_nanos(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp()) * NANOS_PER_SECOND + parsed.microsecond * 1_000


def _any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    return {"stringValue": json.dumps(value, default=str)}

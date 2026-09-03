"""Diagnostics support for SaySo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.core import HomeAssistant

from . import SaySoConfigEntry
from .exceptions import SaySoError

TO_REDACT = {CONF_API_KEY}


class BoundaryFailureCode(StrEnum):
    """Stable diagnostic codes for model-boundary failures."""

    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE_TOOL = "unavailable_tool"
    REQUEST_TIMEOUT = "request_timeout"
    ITERATION_LIMIT = "iteration_limit"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"


class BoundaryPhase(StrEnum):
    """Phase within a model turn when a boundary failure occurred."""

    INITIAL = "initial"
    CORRECTION = "correction"
    FOLLOW_UP = "follow_up"
    EXECUTION = "execution"


@dataclass
class _LastBoundaryFailure:
    code: BoundaryFailureCode
    phase: BoundaryPhase
    fingerprint: str | None
    timestamp: str
    ha_error: str | None = None


@dataclass
class BoundaryDiagnosticsState:
    """Runtime counters and last failure metadata for one config entry."""

    counts: dict[str, int] = field(default_factory=dict)
    last: _LastBoundaryFailure | None = None

    def record(
        self,
        code: BoundaryFailureCode,
        phase: BoundaryPhase,
        *,
        fingerprint: str | None = None,
        ha_error: str | None = None,
    ) -> None:
        """Increment a boundary counter and store safe last-failure metadata."""
        code_key = code.value
        self.counts[code_key] = self.counts.get(code_key, 0) + 1
        self.last = _LastBoundaryFailure(
            code=code,
            phase=phase,
            fingerprint=fingerprint,
            timestamp=datetime.now(tz=UTC).isoformat(),
            ha_error=ha_error,
        )


_ENTRY_BOUNDARY_DIAGNOSTICS: dict[str, BoundaryDiagnosticsState] = {}


def clear_boundary_diagnostics(entry_id: str | None = None) -> None:
    """Clear boundary diagnostics. Intended for tests."""
    if entry_id is None:
        _ENTRY_BOUNDARY_DIAGNOSTICS.clear()
        return
    _ENTRY_BOUNDARY_DIAGNOSTICS.pop(entry_id, None)


def record_boundary_failure(
    entry_id: str,
    code: BoundaryFailureCode,
    phase: BoundaryPhase,
    *,
    fingerprint: str | None = None,
    ha_error: str | None = None,
) -> None:
    """Record one boundary failure for a config entry."""
    state = _ENTRY_BOUNDARY_DIAGNOSTICS.setdefault(entry_id, BoundaryDiagnosticsState())
    state.record(code, phase, fingerprint=fingerprint, ha_error=ha_error)


def boundary_diagnostics_snapshot(entry_id: str) -> dict[str, Any]:
    """Return a redacted boundary diagnostics snapshot for export."""
    state = _ENTRY_BOUNDARY_DIAGNOSTICS.get(entry_id)
    if state is None:
        return {"counts": {}, "last": None}

    last_payload: dict[str, Any] | None = None
    if state.last is not None:
        last_payload = {
            "code": state.last.code.value,
            "phase": state.last.phase.value,
            "fingerprint": state.last.fingerprint,
            "timestamp": state.last.timestamp,
        }
        if state.last.ha_error is not None:
            last_payload["ha_error"] = state.last.ha_error

    return {
        "counts": dict(state.counts),
        "last": last_payload,
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: SaySoConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    connectivity: dict[str, Any] = {
        "api_key_configured": bool(entry.data.get(CONF_API_KEY)),
        "reachable": False,
        "models": [],
        "error": None,
    }

    if entry.runtime_data is not None:
        client = entry.runtime_data.client
        runtime = entry.runtime_data
        connectivity["base_url"] = client.base_url
        connectivity["chat_completions_url"] = client.chat_completions_url
        connectivity["models_url"] = client.models_url
        connectivity["timeout_seconds"] = client._timeout
        try:
            models = await client.list_models()
            connectivity["reachable"] = True
            connectivity["models"] = models
        except SaySoError as err:
            connectivity["error"] = type(err).__name__
        runtime_data: dict[str, Any] = {
            "loaded": True,
            "model": runtime.model,
            "llm_hass_api": runtime.llm_api,
            "temperature": runtime.temperature,
            "max_output_tokens": runtime.max_output_tokens,
            "max_tool_iterations": runtime.max_tool_iterations,
            "system_prompt_length": len(runtime.system_prompt),
        }
    else:
        connectivity["base_url"] = entry.data.get(CONF_URL)
        runtime_data = {"loaded": False}

    return {
        "entry": {
            "title": entry.title,
            "unique_id": entry.unique_id,
            "data": async_redact_data(entry.data, TO_REDACT),
            "options": dict(entry.options),
        },
        "runtime": runtime_data,
        "connectivity": connectivity,
        "boundary": boundary_diagnostics_snapshot(entry.entry_id),
    }

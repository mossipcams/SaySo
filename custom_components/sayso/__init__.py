"""The SaySo integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_URL,
    Platform,
)
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.llm import LLM_API_ASSIST

from .client import LlamaCppClient
from .const import (
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_STORE_UTTERANCES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TRACE_MAX_INTERACTIONS,
    DEFAULT_TRACE_RETENTION_DAYS,
    DEFAULT_TRACE_STORE_UTTERANCES,
    DOMAIN,
    SERVICE_GET_TRACE,
    SERVICE_LIST_TRACES,
)
from .exceptions import SaySoError
from .trace_store import TraceRecorder, TraceStore

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

type SaySoConfigEntry = ConfigEntry[SaySoRuntimeData]

GET_TRACE_SCHEMA = vol.Schema({vol.Required("trace_id"): cv.string})

LIST_TRACES_SCHEMA = vol.Schema(
    {
        vol.Optional("limit", default=50): vol.All(int, vol.Range(min=1, max=1000)),
        vol.Optional("only_failures", default=False): cv.boolean,
        vol.Optional("error_stage"): cv.string,
        vol.Optional("start_time"): cv.datetime,
        vol.Optional("end_time"): cv.datetime,
    }
)


@dataclass
class SaySoRuntimeData:
    """Runtime data stored on the config entry."""

    client: LlamaCppClient
    model: str
    llm_api: str
    system_prompt: str
    temperature: float
    max_output_tokens: int
    max_tool_iterations: int
    traces: TraceStore
    tracer: TraceRecorder


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the SaySo integration."""
    hass.data.setdefault(DOMAIN, {})
    _async_register_trace_services(hass)
    return True


def _async_trace_stores(hass: HomeAssistant) -> list[TraceStore]:
    """Return the trace store of every loaded SaySo config entry."""
    stores: list[TraceStore] = []
    for entry in hass.config_entries.async_entries(DOMAIN):
        runtime = getattr(entry, "runtime_data", None)
        if runtime is not None:
            stores.append(runtime.traces)
    return stores


def _async_register_trace_services(hass: HomeAssistant) -> None:
    """Register the trace retrieval services.

    Services keep retrieval inside Home Assistant's own APIs: no extra HTTP
    server and no frontend.
    """
    if hass.services.has_service(DOMAIN, SERVICE_GET_TRACE):
        return

    async def async_get_trace(call: ServiceCall) -> ServiceResponse:
        """Return one full trace by id."""
        trace_id = call.data["trace_id"]
        for store in _async_trace_stores(hass):
            if (trace := store.get(trace_id)) is not None:
                return trace
        return {"summary": None, "events": []}

    async def async_list_traces(call: ServiceCall) -> ServiceResponse:
        """Return recent interaction summaries, newest first."""
        limit: int = call.data["limit"]
        start_time: datetime | None = call.data.get("start_time")
        end_time: datetime | None = call.data.get("end_time")
        traces: list[dict[str, Any]] = []
        for store in _async_trace_stores(hass):
            traces.extend(
                store.query(
                    limit=limit,
                    only_failures=call.data["only_failures"],
                    error_stage=call.data.get("error_stage"),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
        traces.sort(key=lambda summary: summary.get("started_at") or "", reverse=True)
        return {"traces": traces[:limit]}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_TRACE,
        async_get_trace,
        schema=GET_TRACE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_LIST_TRACES,
        async_list_traces,
        schema=LIST_TRACES_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


async def async_setup_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Set up SaySo from a config entry."""
    options = entry.options
    client = LlamaCppClient.from_hass(
        hass,
        entry.data[CONF_URL],
        api_key=entry.data.get(CONF_API_KEY),
        timeout=options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
    )

    try:
        await client.validate_model(options[CONF_MODEL])
    except SaySoError:
        # Allow setup so options can be corrected without removing the entry.
        pass

    traces = TraceStore(
        hass,
        retention_days=int(
            options.get(CONF_TRACE_RETENTION_DAYS, DEFAULT_TRACE_RETENTION_DAYS)
        ),
        max_interactions=int(
            options.get(CONF_TRACE_MAX_INTERACTIONS, DEFAULT_TRACE_MAX_INTERACTIONS)
        ),
        store_utterances=bool(
            options.get(CONF_TRACE_STORE_UTTERANCES, DEFAULT_TRACE_STORE_UTTERANCES)
        ),
    )
    await traces.async_load()
    traces.async_prune()

    entry.runtime_data = SaySoRuntimeData(
        client=client,
        model=options[CONF_MODEL],
        llm_api=options.get(CONF_LLM_HASS_API, LLM_API_ASSIST),
        system_prompt=options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT),
        temperature=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        max_output_tokens=options.get(CONF_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS),
        max_tool_iterations=options.get(
            CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
        ),
        traces=traces,
        tracer=TraceRecorder(hass, traces),
    )

    _async_register_trace_services(hass)

    if PLATFORMS:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Unload a SaySo config entry."""
    if PLATFORMS:
        if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
            return False
    if entry.runtime_data is not None:
        entry.runtime_data.tracer.async_shutdown()
    entry.runtime_data = None  # type: ignore[assignment]
    return True

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
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
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.llm import LLM_API_ASSIST

from .client import LlamaCppClient
from .const import (
    BACKEND_EMBEDDED,
    BACKEND_EXTERNAL,
    CONF_DUCK_VOLUME,
    CONF_GRAMMAR_DECODING,
    CONF_MODEL_PATH,
    CONF_N_CTX,
    CONF_N_THREADS,
    CONF_TIMEOUT,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_STORE_UTTERANCES,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_DUCK_VOLUME,
    DEFAULT_GRAMMAR_DECODING,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_N_CTX,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TRACE_MAX_INTERACTIONS,
    DEFAULT_TRACE_RETENTION_DAYS,
    DEFAULT_TRACE_STORE_UTTERANCES,
    DOMAIN,
    EMBEDDED_MODELS,
    SERVICE_GET_TRACE,
    SERVICE_LIST_TRACES,
    VAD_SENSITIVITY_UNIQUE_ID,
)
from .core.exceptions import SaySoDependencyError, SaySoError, SaySoModelLoadError
from .inference import (
    EmbeddedEngine,
    ExternalEngine,
    SaySoInferenceEngine,
    entry_backend,
)
from .model_store import (
    SaySoModelMissingError,
    async_ensure_llama_cpp,
    async_verify_model,
    models_dir,
)
from .repairs import (
    ISSUE_LLAMA_CPP_MISSING,
    ISSUE_MODEL_CORRUPT,
    ISSUE_MODEL_MISSING,
    ISSUE_MODEL_PATH_MISSING,
    async_clear_issues,
    async_raise_issue,
)
from .trace_store import TraceRecorder, TraceStore

if TYPE_CHECKING:
    from .ducking import MediaDucker

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.SELECT]

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
    engine: SaySoInferenceEngine
    client: LlamaCppClient | None
    model: str
    traces: TraceStore
    tracer: TraceRecorder
    ducker: MediaDucker
    grammar_decoding: bool = False
    llm_api: str = LLM_API_ASSIST
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    hass.data.setdefault(DOMAIN, {})
    _async_register_trace_services(hass)
    _async_register_volume_step(hass)
    return True


def _async_register_volume_step(hass: HomeAssistant) -> None:
    try:
        from .volume_step import async_register_volume_step

        async_register_volume_step(hass)
    except Exception:
        _LOGGER.exception(
            "Volume step fallback disabled; Home Assistant's default relative volume handler stays active"
        )


def _async_trace_stores(hass: HomeAssistant) -> list[TraceStore]:
    stores: list[TraceStore] = []
    for entry in hass.config_entries.async_entries(DOMAIN):
        runtime = getattr(entry, "runtime_data", None)
        if runtime is not None:
            stores.append(runtime.traces)
    return stores


def _async_register_trace_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_GET_TRACE):
        return

    async def async_get_trace(call: ServiceCall) -> ServiceResponse:
        trace_id = call.data["trace_id"]
        for store in _async_trace_stores(hass):
            if (trace := store.get(trace_id)) is not None:
                return trace
        return {"summary": None, "events": []}

    async def async_list_traces(call: ServiceCall) -> ServiceResponse:
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


def _synced_embedded_options(entry: SaySoConfigEntry) -> dict[str, Any] | None:
    options = entry.options
    if entry_backend(entry) != BACKEND_EMBEDDED or options.get(CONF_MODEL_PATH):
        return None
    model = options.get(CONF_MODEL)
    if model in EMBEDDED_MODELS or (model and Path(model).is_absolute()):
        return None
    return {**options, CONF_MODEL: DEFAULT_MODEL_FILENAME}


async def async_migrate_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    if entry.version > 2:
        return False
    if entry.version == 1:
        options = dict(entry.options)
        if entry.minor_version < 2:
            options = _migrated_embedded_options(hass, entry) or options
        _async_rekey_vad_select(hass, entry)
        hass.config_entries.async_update_entry(
            entry, options=options, version=2, minor_version=1
        )
    return True


def _async_rekey_vad_select(hass: HomeAssistant, entry: SaySoConfigEntry) -> None:
    registry = er.async_get(hass)
    entity_id = registry.async_get_entity_id(
        Platform.SELECT, DOMAIN, VAD_SENSITIVITY_UNIQUE_ID
    )
    if entity_id is None:
        return
    registry.async_update_entity(
        entity_id,
        new_unique_id=f"{entry.entry_id}-vad-sensitivity",
        config_entry_id=entry.entry_id,
    )


def _migrated_embedded_options(
    hass: HomeAssistant, entry: SaySoConfigEntry
) -> dict[str, Any] | None:
    options = entry.options
    legacy_path = options.get(CONF_MODEL_PATH)
    if entry_backend(entry) != BACKEND_EMBEDDED or not legacy_path:
        return None
    path = Path(legacy_path)
    if path.parent == models_dir(hass) and path.name in EMBEDDED_MODELS:
        model = path.name
    else:
        model = legacy_path
    return {**options, CONF_MODEL: model, CONF_MODEL_PATH: ""}


def _readable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.R_OK)


def _async_fail(
    hass: HomeAssistant,
    entry: SaySoConfigEntry,
    key: str,
    message: str,
    *,
    fixable: bool,
    discard_model: bool = False,
    placeholders: dict[str, str] | None = None,
) -> None:
    _LOGGER.error("SaySo setup failed: %s", message)
    async_raise_issue(
        hass,
        entry.entry_id,
        key,
        fixable=fixable,
        discard_model=discard_model,
        placeholders=placeholders,
    )


async def _async_embedded_engine(
    hass: HomeAssistant, entry: SaySoConfigEntry
) -> EmbeddedEngine | None:
    options = entry.options
    try:
        await async_ensure_llama_cpp(hass)
    except SaySoDependencyError as err:
        _async_fail(hass, entry, ISSUE_LLAMA_CPP_MISSING, str(err), fixable=True)
        return None

    model = options[CONF_MODEL]
    if model in EMBEDDED_MODELS:
        try:
            model_path = await async_verify_model(
                hass, filename=model, sha256=EMBEDDED_MODELS[model].sha256
            )
        except SaySoModelMissingError as err:
            _async_fail(hass, entry, ISSUE_MODEL_MISSING, str(err), fixable=True)
            return None
        except SaySoModelLoadError as err:
            _async_fail(hass, entry, ISSUE_MODEL_CORRUPT, str(err), fixable=True)
            return None
    else:
        model_path = Path(model)
        if not await hass.async_add_executor_job(_readable_file, model_path):
            _async_fail(
                hass,
                entry,
                ISSUE_MODEL_PATH_MISSING,
                f"Model file not found or unreadable: {model_path}",
                fixable=False,
                placeholders={"path": str(model_path)},
            )
            return None

    return EmbeddedEngine(
        model_path,
        n_ctx=int(options.get(CONF_N_CTX, DEFAULT_N_CTX)),
        n_threads=int(options.get(CONF_N_THREADS) or 0) or None,
        timeout=options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
    )


def _default_conversation_model(entry: SaySoConfigEntry) -> str:
    for subentry in entry.subentries.values():
        if subentry.subentry_type == CONVERSATION_SUBENTRY_TYPE:
            return subentry.data.get(CONF_MODEL, "")
    return ""


async def _async_reload_on_update(hass: HomeAssistant, entry: SaySoConfigEntry) -> None:
    hass.config_entries.async_schedule_reload(entry.entry_id)


async def _async_build_engine(
    hass: HomeAssistant, entry: SaySoConfigEntry
) -> tuple[SaySoInferenceEngine, LlamaCppClient | None] | None:
    options = entry.options
    timeout = options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
    client: LlamaCppClient | None = None

    if entry_backend(entry) == BACKEND_EXTERNAL:
        client = LlamaCppClient.from_hass(
            hass,
            entry.data[CONF_URL],
            api_key=entry.data.get(CONF_API_KEY),
            timeout=timeout,
        )
        engine: SaySoInferenceEngine = ExternalEngine(
            client, _default_conversation_model(entry)
        )
    else:
        embedded = await _async_embedded_engine(hass, entry)
        if embedded is None:
            return None
        engine = embedded

    try:
        await engine.async_start()
    except SaySoModelLoadError as err:
        await engine.async_shutdown()
        _async_fail(
            hass,
            entry,
            ISSUE_MODEL_CORRUPT,
            str(err),
            fixable=True,
            discard_model=True,
        )
        return None
    except SaySoError as err:
        await engine.async_shutdown()
        raise ConfigEntryNotReady(str(err)) from err
    return engine, client


async def async_setup_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    synced_options = _synced_embedded_options(entry)
    if synced_options is not None:
        hass.config_entries.async_update_entry(entry, options=synced_options)
    entry.async_on_unload(entry.add_update_listener(_async_reload_on_update))
    options = entry.options
    built = await _async_build_engine(hass, entry)
    if built is None:
        return False
    engine, client = built
    async_clear_issues(hass, entry.entry_id)

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

    from .ducking import async_setup_ducking

    ducker = await async_setup_ducking(
        hass, float(options.get(CONF_DUCK_VOLUME, DEFAULT_DUCK_VOLUME))
    )
    entry.async_on_unload(ducker.async_teardown)

    entry.runtime_data = SaySoRuntimeData(
        engine=engine,
        client=client,
        model=engine.model_name,
        traces=traces,
        tracer=TraceRecorder(hass, traces),
        ducker=ducker,
        grammar_decoding=bool(
            options.get(CONF_GRAMMAR_DECODING, DEFAULT_GRAMMAR_DECODING)
        ),
    )

    _async_register_trace_services(hass)

    if PLATFORMS:
        try:
            await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        except Exception:
            await engine.async_shutdown()
            raise
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    if PLATFORMS:
        if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
            return False
    if entry.runtime_data is not None:
        entry.runtime_data.tracer.async_shutdown()
        await entry.runtime_data.engine.async_shutdown()
    entry.runtime_data = None
    return True

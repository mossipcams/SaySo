from __future__ import annotations

import logging
from contextlib import suppress
from types import MappingProxyType
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.assist_pipeline import (
    async_get_pipelines,
    async_update_pipeline,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry, UnknownEntry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_URL,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store

from .client import normalize_base_url
from .config_flow import _default_conversation_subentry
from .const import (
    BACKEND_EMBEDDED,
    BACKEND_EXTERNAL,
    CONF_DUCK_VOLUME,
    CONF_GRAMMAR_DECODING,
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_STORE_UTTERANCES,
    CONVERSATION_SUBENTRY_TYPE,
    DOMAIN,
)
from .inference import entry_backend

_LOGGER = logging.getLogger(__name__)

SNAPSHOT_STORAGE_KEY = "sayso.v1_backup"
SNAPSHOT_STORAGE_VERSION = 1

CONVERSATION_KEYS = (
    CONF_LLM_HASS_API,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
)
ENGINE_KEYS = (
    CONF_TIMEOUT,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_STORE_UTTERANCES,
    CONF_GRAMMAR_DECODING,
    CONF_DUCK_VOLUME,
)


async def async_snapshot_v1_entries(hass: HomeAssistant) -> None:
    store: Store[dict[str, Any]] = Store(
        hass, SNAPSHOT_STORAGE_VERSION, SNAPSHOT_STORAGE_KEY
    )
    if await store.async_load() is not None:
        return
    legacy = [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.version == 1
    ]
    await store.async_save({"entries": [_snapshot(hass, entry) for entry in legacy]})


def _snapshot(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    registry = er.async_get(hass)
    devices = dr.async_get(hass)
    return {
        "entry_id": entry.entry_id,
        "title": entry.title,
        "data": dict(entry.data),
        "options": dict(entry.options),
        "entities": [
            {
                "entity_id": row.entity_id,
                "unique_id": row.unique_id,
                "platform": row.platform,
            }
            for row in er.async_entries_for_config_entry(registry, entry.entry_id)
        ],
        "devices": [
            {"id": device.id, "identifiers": [list(i) for i in device.identifiers]}
            for device in dr.async_entries_for_config_entry(devices, entry.entry_id)
        ],
    }


async def async_merge_v1_entry(
    hass: HomeAssistant, entry: ConfigEntry, options: dict[str, Any]
) -> bool:
    group = _merge_group(hass, entry)
    main = group[0]
    if main.entry_id != entry.entry_id:
        _adopt(hass, main, entry, entry.options)
        await _async_rewrite_pipelines(hass, [entry.entry_id])
        hass.async_create_task(_async_remove_merged(hass, entry.entry_id))
        return False

    _adopt(hass, main, main, options)
    siblings = group[1:]
    for sibling in siblings:
        _adopt(hass, main, sibling, sibling.options)
    await _async_rewrite_pipelines(hass, [sibling.entry_id for sibling in siblings])
    for sibling in siblings:
        await _async_remove_merged(hass, sibling.entry_id)
    _async_finish_main(hass, main, options)
    return True


def _group_key(entry: ConfigEntry) -> str:
    if entry_backend(entry) == BACKEND_EMBEDDED:
        return BACKEND_EMBEDDED
    return normalize_base_url(entry.data[CONF_URL])


def _merge_group(hass: HomeAssistant, entry: ConfigEntry) -> list[ConfigEntry]:
    key = _group_key(entry)
    members = [
        member
        for member in hass.config_entries.async_entries(DOMAIN)
        if member.version <= 2 and _group_key(member) == key
    ]
    return sorted(members, key=lambda member: (member.version != 2, member.created_at))


def _adopt(
    hass: HomeAssistant,
    main: ConfigEntry,
    source: ConfigEntry,
    source_options: dict[str, Any],
) -> None:
    subentry_id = _ensure_conversation_subentry(hass, main, source, source_options)
    _relink(hass, source.entry_id, main.entry_id, subentry_id)


def _ensure_conversation_subentry(
    hass: HomeAssistant,
    main: ConfigEntry,
    source: ConfigEntry,
    source_options: dict[str, Any],
) -> str:
    for subentry in main.subentries.values():
        if subentry.unique_id == source.entry_id:
            return subentry.subentry_id
    if source.entry_id != main.entry_id:
        _log_merge_differences(main, source)
    external = entry_backend(source) == BACKEND_EXTERNAL
    template = _default_conversation_subentry(
        source.title, source_options.get(CONF_MODEL) if external else None
    )
    data = {
        **template["data"],
        **{
            key: source_options[key]
            for key in CONVERSATION_KEYS
            if key in source_options
        },
    }
    subentry = ConfigSubentry(
        data=MappingProxyType(data),
        subentry_type=CONVERSATION_SUBENTRY_TYPE,
        title=source.title,
        unique_id=source.entry_id,
    )
    hass.config_entries.async_add_subentry(main, subentry)
    return subentry.subentry_id


def _log_merge_differences(main: ConfigEntry, source: ConfigEntry) -> None:
    differing = [
        key for key in ENGINE_KEYS if source.options.get(key) != main.options.get(key)
    ]
    if differing:
        _LOGGER.warning(
            "Merging SaySo entry %s into %s: engine options differ (%s); %s keeps its values",
            source.title,
            main.title,
            ", ".join(differing),
            main.title,
        )
    if source.data.get(CONF_API_KEY) != main.data.get(CONF_API_KEY):
        _LOGGER.warning(
            "Merging SaySo entry %s into %s: API keys differ; %s keeps its key",
            source.title,
            main.title,
            main.title,
        )


def _vad_unique_id(entry_id: str) -> str:
    return f"{entry_id}-vad-sensitivity"


def _relink(
    hass: HomeAssistant, source_id: str, target_id: str, subentry_id: str
) -> None:
    registry = er.async_get(hass)
    for row in er.async_entries_for_config_entry(registry, source_id):
        if row.domain == conversation.DOMAIN and row.unique_id == source_id:
            if (row.config_entry_id, row.config_subentry_id) != (
                target_id,
                subentry_id,
            ):
                registry.async_update_entity(
                    row.entity_id,
                    config_entry_id=target_id,
                    config_subentry_id=subentry_id,
                )
        elif source_id != target_id:
            changes: dict[str, Any] = {"config_entry_id": target_id}
            if row.unique_id == _vad_unique_id(source_id):
                changes["new_unique_id"] = _vad_unique_id(target_id)
            registry.async_update_entity(row.entity_id, **changes)

    device = dr.async_get(hass).async_get_device(identifiers={(DOMAIN, source_id)})
    if device is not None and device.config_entries_subentries.get(target_id) != {
        subentry_id
    }:
        dr.async_get(hass).async_update_device(
            device.id,
            new_config_entry_id=target_id,
            new_config_subentry_id=subentry_id,
        )


async def _async_rewrite_pipelines(hass: HomeAssistant, merged_ids: list[str]) -> None:
    if not merged_ids:
        return
    if "assist_pipeline" not in hass.config.components:
        _LOGGER.info(
            "assist_pipeline is not loaded; pipelines naming merged SaySo entries were not rewritten"
        )
        return
    registry = er.async_get(hass)
    for pipeline in async_get_pipelines(hass):
        if pipeline.conversation_engine not in merged_ids:
            continue
        entity_id = registry.async_get_entity_id(
            conversation.DOMAIN, DOMAIN, pipeline.conversation_engine
        )
        if entity_id is None:
            _LOGGER.warning(
                "Pipeline %s still names merged SaySo entry %s; no conversation entity found",
                pipeline.name,
                pipeline.conversation_engine,
            )
            continue
        await async_update_pipeline(hass, pipeline, conversation_engine=entity_id)


async def _async_remove_merged(hass: HomeAssistant, entry_id: str) -> None:
    with suppress(UnknownEntry):
        await hass.config_entries.async_remove(entry_id)


def _async_finish_main(
    hass: HomeAssistant, entry: ConfigEntry, options: dict[str, Any]
) -> None:
    external = entry_backend(entry) == BACKEND_EXTERNAL
    kept = {
        key: value
        for key, value in options.items()
        if key not in CONVERSATION_KEYS and not (external and key == CONF_MODEL)
    }
    changes: dict[str, Any] = {"options": kept, "version": 2, "minor_version": 1}
    if external:
        changes["unique_id"] = normalize_base_url(entry.data[CONF_URL])
    hass.config_entries.async_update_entry(entry, **changes)

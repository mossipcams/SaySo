from __future__ import annotations

import logging
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.assist_pipeline import (
    async_create_default_pipeline,
    async_get_pipelines,
    async_update_pipeline,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_URL,
    Platform,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.llm import LLM_API_ASSIST
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso import async_migrate_entry
from custom_components.sayso.client import normalize_base_url
from custom_components.sayso.const import (
    BACKEND_EMBEDDED,
    CONF_BACKEND,
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
    DEFAULT_MODEL_FILENAME,
    DOMAIN,
    VAD_SENSITIVITY_UNIQUE_ID,
)
from custom_components.sayso.migration import SNAPSHOT_STORAGE_KEY

URL_A = "http://llm-a.local:8080/v1"
URL_B = "http://llm-b.local:8080/v1"
URL_C = "http://llm-c.local:8080/v1"
EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


def test_migration_module_does_not_import_assist_pipeline_eagerly() -> None:
    script = (
        "import sys; import custom_components.sayso.migration; "
        "print('homeassistant.components.assist_pipeline' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def _engine_options(**overrides: Any) -> dict[str, Any]:
    return {
        CONF_TIMEOUT: 30,
        CONF_TRACE_RETENTION_DAYS: 30,
        CONF_TRACE_MAX_INTERACTIONS: 500,
        CONF_TRACE_STORE_UTTERANCES: False,
        CONF_GRAMMAR_DECODING: False,
        CONF_DUCK_VOLUME: 0.02,
        **overrides,
    }


def _conversation_options(
    prompt: str = "Be brief.", temperature: float = 0.3, max_tokens: int = 200
) -> dict[str, Any]:
    return {
        CONF_LLM_HASS_API: LLM_API_ASSIST,
        CONF_PROMPT: prompt,
        CONF_TEMPERATURE: temperature,
        CONF_MAX_OUTPUT_TOKENS: max_tokens,
        CONF_MAX_TOOL_ITERATIONS: 2,
    }


def _legacy(
    hass: HomeAssistant,
    *,
    data: dict[str, Any],
    options: dict[str, Any],
    title: str,
    created_at: datetime = EPOCH,
    version: int = 1,
    minor_version: int = 1,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data=data,
        options=options,
        version=version,
        minor_version=minor_version,
    )
    entry.created_at = created_at
    entry.add_to_hass(hass)
    return entry


def _external(
    hass: HomeAssistant,
    *,
    url: str = URL_A,
    model: str = "model-a",
    created_at: datetime = EPOCH,
    engine: dict[str, Any] | None = None,
    version: int = 1,
    minor_version: int = 1,
) -> MockConfigEntry:
    return _legacy(
        hass,
        data={CONF_URL: url},
        options={
            CONF_MODEL: model,
            **(engine if engine is not None else _engine_options()),
            **_conversation_options(),
        },
        title=f"{model} @ {url}",
        created_at=created_at,
        version=version,
        minor_version=minor_version,
    )


def _embedded(hass: HomeAssistant) -> MockConfigEntry:
    return _legacy(
        hass,
        data={CONF_BACKEND: BACKEND_EMBEDDED},
        options={
            CONF_MODEL: DEFAULT_MODEL_FILENAME,
            **_engine_options(),
            **_conversation_options(),
        },
        title="SaySo (local)",
    )


def _link_rows(hass: HomeAssistant, entry: ConfigEntry) -> tuple[str, str]:
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
    )
    row = er.async_get(hass).async_get_or_create(
        conversation.DOMAIN,
        DOMAIN,
        entry.entry_id,
        config_entry=entry,
        device_id=device.id,
        suggested_object_id=f"sayso_{entry.entry_id[:8].lower()}",
    )
    return row.entity_id, device.id


def _legacy_vad_row(hass: HomeAssistant) -> str:
    row = er.async_get(hass).async_get_or_create(
        Platform.SELECT,
        DOMAIN,
        VAD_SENSITIVITY_UNIQUE_ID,
        suggested_object_id=f"{DOMAIN}_vad_sensitivity",
    )
    return row.entity_id


def _conversation_subentries(entry: ConfigEntry) -> list[Any]:
    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == CONVERSATION_SUBENTRY_TYPE
    ]


def _subentry_for(entry: ConfigEntry, legacy_id: str) -> Any:
    [subentry] = [
        s for s in _conversation_subentries(entry) if s.unique_id == legacy_id
    ]
    return subentry


def _row_for(hass: HomeAssistant, entity_id: str) -> er.RegistryEntry:
    row = er.async_get(hass).async_get(entity_id)
    assert row is not None
    return row


async def test_single_embedded_entry_becomes_main_with_conversation_subentry(
    hass: HomeAssistant,
) -> None:
    entry = _embedded(hass)
    entity_id, device_id = _link_rows(hass, entry)
    vad_entity_id = _legacy_vad_row(hass)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.version == 2
    assert entry.minor_version == 1
    [subentry] = _conversation_subentries(entry)
    assert subentry.unique_id == entry.entry_id
    assert subentry.title == "SaySo (local)"
    assert subentry.data[CONF_PROMPT] == "Be brief."
    assert subentry.data[CONF_TEMPERATURE] == 0.3
    assert subentry.data[CONF_MAX_OUTPUT_TOKENS] == 200
    assert CONF_MODEL not in subentry.data

    row = _row_for(hass, entity_id)
    assert row.entity_id == entity_id
    assert row.config_entry_id == entry.entry_id
    assert row.config_subentry_id == subentry.subentry_id

    device = dr.async_get(hass).async_get(device_id)
    assert device.identifiers == {(DOMAIN, entry.entry_id)}
    assert device.config_entries_subentries == {entry.entry_id: {subentry.subentry_id}}

    vad = _row_for(hass, vad_entity_id)
    assert vad.unique_id == f"{entry.entry_id}-vad-sensitivity"
    assert vad.config_entry_id == entry.entry_id

    assert CONF_PROMPT not in entry.options
    assert CONF_TEMPERATURE not in entry.options
    assert CONF_LLM_HASS_API not in entry.options
    assert entry.options[CONF_MODEL] == DEFAULT_MODEL_FILENAME
    assert entry.options[CONF_TIMEOUT] == 30


async def test_single_external_entry_moves_model_and_normalizes_unique_id(
    hass: HomeAssistant,
) -> None:
    entry = _external(hass, url=f"{URL_A}/", model="model-a")
    entity_id, _device_id = _link_rows(hass, entry)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.version == 2
    assert entry.minor_version == 1
    assert entry.unique_id == normalize_base_url(URL_A)
    [subentry] = _conversation_subentries(entry)
    assert subentry.data[CONF_MODEL] == "model-a"
    assert CONF_MODEL not in entry.options
    assert entry.options[CONF_TIMEOUT] == 30
    row = _row_for(hass, entity_id)
    assert row.config_subentry_id == subentry.subentry_id
    assert row.config_entry_id == entry.entry_id


async def test_entry_already_at_version_two_is_untouched(hass: HomeAssistant) -> None:
    options = {CONF_MODEL: "model-a", **_engine_options(), **_conversation_options()}
    entry = _legacy(
        hass,
        data={CONF_URL: URL_A},
        options=options,
        title="model-a",
        version=2,
        minor_version=1,
    )
    entity_id, _device_id = _link_rows(hass, entry)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.version == 2
    assert entry.minor_version == 1
    assert dict(entry.options) == options
    assert entry.subentries == {}
    assert _row_for(hass, entity_id).config_subentry_id is None


async def test_future_version_returns_false(hass: HomeAssistant) -> None:
    entry = _external(hass, version=3)

    assert await async_migrate_entry(hass, entry) is False

    assert entry.version == 3
    assert entry.subentries == {}


async def test_same_url_entries_merge_into_earliest_main(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    main = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    sibling = _external(
        hass,
        url=f"{URL_A}/",
        model="model-b",
        created_at=EPOCH + timedelta(days=1),
        engine=_engine_options(**{CONF_DUCK_VOLUME: 0.05}),
    )
    main_entity, main_device = _link_rows(hass, main)
    sibling_entity, sibling_device = _link_rows(hass, sibling)

    with caplog.at_level(logging.WARNING):
        assert await async_migrate_entry(hass, main) is True
    await hass.async_block_till_done()

    assert main.version == 2
    assert main.minor_version == 1
    assert main.unique_id == normalize_base_url(URL_A)
    assert hass.config_entries.async_get_entry(sibling.entry_id) is None
    assert {s.unique_id for s in _conversation_subentries(main)} == {
        main.entry_id,
        sibling.entry_id,
    }
    main_sub = _subentry_for(main, main.entry_id)
    sibling_sub = _subentry_for(main, sibling.entry_id)
    assert main_sub.data[CONF_MODEL] == "model-a"
    assert sibling_sub.data[CONF_MODEL] == "model-b"
    assert sibling_sub.title == sibling.title

    main_row = _row_for(hass, main_entity)
    assert main_row.config_entry_id == main.entry_id
    assert main_row.config_subentry_id == main_sub.subentry_id
    sibling_row = _row_for(hass, sibling_entity)
    assert sibling_row.config_entry_id == main.entry_id
    assert sibling_row.config_subentry_id == sibling_sub.subentry_id

    device = dr.async_get(hass).async_get(sibling_device)
    assert device.config_entries_subentries == {
        main.entry_id: {sibling_sub.subentry_id}
    }
    assert device.identifiers == {(DOMAIN, sibling.entry_id)}
    assert main_device != sibling_device

    assert CONF_MODEL not in main.options
    assert main.options[CONF_DUCK_VOLUME] == 0.02
    assert "engine options differ (duck_volume)" in caplog.text


async def test_non_main_entry_migrated_first_is_merged_then_removed(
    hass: HomeAssistant,
) -> None:
    main = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    sibling = _external(
        hass, url=URL_A, model="model-b", created_at=EPOCH + timedelta(days=1)
    )
    main_entity, _ = _link_rows(hass, main)
    sibling_entity, _ = _link_rows(hass, sibling)

    assert await async_migrate_entry(hass, sibling) is False
    await hass.async_block_till_done()

    assert hass.config_entries.async_get_entry(sibling.entry_id) is None
    assert main.version == 1
    assert {s.unique_id for s in _conversation_subentries(main)} == {sibling.entry_id}
    assert _row_for(hass, sibling_entity).config_entry_id == main.entry_id

    assert await async_migrate_entry(hass, sibling) is False

    assert await async_migrate_entry(hass, main) is True

    assert main.version == 2
    assert len(_conversation_subentries(main)) == 2
    assert _row_for(hass, main_entity).config_entry_id == main.entry_id
    assert _row_for(hass, sibling_entity).config_entry_id == main.entry_id
    assert hass.config_entries.async_get_entry(sibling.entry_id) is None


async def test_two_urls_stay_two_mains(hass: HomeAssistant) -> None:
    first = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    second = _external(hass, url=URL_B, model="model-b", created_at=EPOCH)
    _link_rows(hass, first)
    _link_rows(hass, second)

    assert await async_migrate_entry(hass, first) is True
    assert await async_migrate_entry(hass, second) is True

    assert first.version == 2
    assert second.version == 2
    assert first.unique_id == normalize_base_url(URL_A)
    assert second.unique_id == normalize_base_url(URL_B)
    assert [s.unique_id for s in _conversation_subentries(first)] == [first.entry_id]
    assert [s.unique_id for s in _conversation_subentries(second)] == [second.entry_id]
    assert hass.config_entries.async_get_entry(first.entry_id) is not None
    assert hass.config_entries.async_get_entry(second.entry_id) is not None


async def test_snapshot_is_written_once_before_any_change(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    first = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    second = _external(hass, url=URL_B, model="model-b", created_at=EPOCH)
    first_entity, first_device = _link_rows(hass, first)
    second_entity, _ = _link_rows(hass, second)
    original_options = dict(first.options)

    assert await async_migrate_entry(hass, first) is True

    stored = hass_storage[SNAPSHOT_STORAGE_KEY]["data"]["entries"]
    by_id = {item["entry_id"]: item for item in stored}
    assert set(by_id) == {first.entry_id, second.entry_id}
    assert by_id[first.entry_id]["title"] == first.title
    assert by_id[first.entry_id]["options"] == original_options
    assert by_id[first.entry_id]["data"] == {CONF_URL: URL_A}
    assert {row["entity_id"] for row in by_id[first.entry_id]["entities"]} == {
        first_entity
    }
    assert [device["id"] for device in by_id[first.entry_id]["devices"]] == [
        first_device
    ]
    assert {row["entity_id"] for row in by_id[second.entry_id]["entities"]} == {
        second_entity
    }

    third = _external(hass, url=URL_C, model="model-c", created_at=EPOCH)

    assert await async_migrate_entry(hass, third) is True

    stored_again = hass_storage[SNAPSHOT_STORAGE_KEY]["data"]["entries"]
    assert {item["entry_id"] for item in stored_again} == {
        first.entry_id,
        second.entry_id,
    }


async def test_snapshot_exists_when_merge_fails_before_changes(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    entry = _external(hass, url=URL_A, model="model-a")
    original_options = dict(entry.options)

    with (
        patch(
            "custom_components.sayso.async_merge_v1_entry",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        pytest.raises(RuntimeError),
    ):
        await async_migrate_entry(hass, entry)

    assert entry.version == 1
    assert dict(entry.options) == original_options
    stored = hass_storage[SNAPSHOT_STORAGE_KEY]["data"]["entries"]
    assert [item["entry_id"] for item in stored] == [entry.entry_id]


async def test_rerun_after_crash_between_subentry_add_and_relink(
    hass: HomeAssistant,
) -> None:
    entry = _external(hass, url=URL_A, model="model-a")
    entity_id, _device_id = _link_rows(hass, entry)
    registry = er.async_get(hass)

    with (
        patch.object(
            registry,
            "async_update_entity",
            side_effect=RuntimeError("crash"),
        ),
        pytest.raises(RuntimeError),
    ):
        await async_migrate_entry(hass, entry)

    assert entry.version == 1
    [first] = _conversation_subentries(entry)
    assert _row_for(hass, entity_id).config_subentry_id is None

    assert await async_migrate_entry(hass, entry) is True

    [subentry] = _conversation_subentries(entry)
    assert subentry.subentry_id == first.subentry_id
    assert entry.version == 2
    row = _row_for(hass, entity_id)
    assert row.config_subentry_id == subentry.subentry_id
    assert row.config_entry_id == entry.entry_id
    assert len(er.async_entries_for_config_entry(registry, entry.entry_id)) == 1


async def test_rerun_after_crash_between_relink_and_removal(
    hass: HomeAssistant,
) -> None:
    main = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    sibling = _external(
        hass, url=URL_A, model="model-b", created_at=EPOCH + timedelta(days=1)
    )
    main_entity, _ = _link_rows(hass, main)
    sibling_entity, _ = _link_rows(hass, sibling)

    with (
        patch.object(
            hass.config_entries,
            "async_remove",
            new=AsyncMock(side_effect=RuntimeError("crash")),
        ),
        pytest.raises(RuntimeError),
    ):
        await async_migrate_entry(hass, main)

    assert main.version == 1
    assert hass.config_entries.async_get_entry(sibling.entry_id) is not None
    assert _row_for(hass, sibling_entity).config_entry_id == main.entry_id

    assert await async_migrate_entry(hass, main) is True
    await hass.async_block_till_done()

    assert main.version == 2
    assert hass.config_entries.async_get_entry(sibling.entry_id) is None
    assert len(_conversation_subentries(main)) == 2
    assert (
        _row_for(hass, main_entity).config_subentry_id
        == _subentry_for(main, main.entry_id).subentry_id
    )
    assert (
        _row_for(hass, sibling_entity).config_subentry_id
        == _subentry_for(main, sibling.entry_id).subentry_id
    )


async def test_rerun_after_crash_before_version_bump_adds_no_duplicates(
    hass: HomeAssistant,
) -> None:
    entry = _external(hass, url=URL_A, model="model-a")
    entity_id, _device_id = _link_rows(hass, entry)
    registry = er.async_get(hass)

    with (
        patch.object(
            hass.config_entries,
            "async_update_entry",
            side_effect=RuntimeError("crash"),
        ),
        pytest.raises(RuntimeError),
    ):
        await async_migrate_entry(hass, entry)

    assert entry.version == 1
    [first] = _conversation_subentries(entry)
    assert _row_for(hass, entity_id).config_subentry_id == first.subentry_id

    assert await async_migrate_entry(hass, entry) is True

    [subentry] = _conversation_subentries(entry)
    assert subentry.subentry_id == first.subentry_id
    assert entry.version == 2
    row = _row_for(hass, entity_id)
    assert row.entity_id == entity_id
    assert row.config_subentry_id == subentry.subentry_id
    assert len(er.async_entries_for_config_entry(registry, entry.entry_id)) == 1


async def test_pipeline_naming_merged_entry_is_rewritten(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "assist_pipeline", {})
    main = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    sibling = _external(
        hass, url=URL_A, model="model-b", created_at=EPOCH + timedelta(days=1)
    )
    _link_rows(hass, main)
    sibling_entity, _ = _link_rows(hass, sibling)
    merged = await async_create_default_pipeline(
        hass, stt_engine_id=None, tts_engine_id=None, pipeline_name="Office"
    )
    assert merged is not None
    await async_update_pipeline(hass, merged, conversation_engine=sibling.entry_id)
    alias = await async_create_default_pipeline(
        hass, stt_engine_id=None, tts_engine_id=None, pipeline_name="Kitchen"
    )
    assert alias is not None
    await async_update_pipeline(hass, alias, conversation_engine=main.entry_id)

    assert await async_migrate_entry(hass, main) is True
    await hass.async_block_till_done()

    pipelines = {p.name: p for p in async_get_pipelines(hass)}
    assert pipelines["Office"].conversation_engine == sibling_entity
    assert pipelines["Kitchen"].conversation_engine == main.entry_id


async def test_pipeline_rewrite_is_skipped_when_assist_pipeline_is_not_loaded(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    main = _external(hass, url=URL_A, model="model-a", created_at=EPOCH)
    sibling = _external(
        hass, url=URL_A, model="model-b", created_at=EPOCH + timedelta(days=1)
    )
    _link_rows(hass, main)
    _link_rows(hass, sibling)
    assert "assist_pipeline" not in hass.config.components

    with caplog.at_level(logging.INFO):
        assert await async_migrate_entry(hass, main) is True
    await hass.async_block_till_done()

    assert main.version == 2
    assert hass.config_entries.async_get_entry(sibling.entry_id) is None
    assert "assist_pipeline is not loaded" in caplog.text

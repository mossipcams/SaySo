from __future__ import annotations

import pytest
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso import async_migrate_entry
from custom_components.sayso.const import (
    BACKEND_EMBEDDED,
    CONF_BACKEND,
    DOMAIN,
    VAD_SENSITIVITY_UNIQUE_ID,
)
from tests.test_init import mock_llama_client as mock_llama_client
from tests.test_vad_sensitivity import SELECT_ENTITY_ID, _create_entry


def _v1_entry(
    hass: HomeAssistant, *, version: int = 1, minor_version: int = 1
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=version,
        minor_version=minor_version,
        data={CONF_BACKEND: BACKEND_EMBEDDED},
        options={},
    )
    entry.add_to_hass(hass)
    return entry


def _legacy_vad_entity_id(hass: HomeAssistant) -> str:
    row = er.async_get(hass).async_get_or_create(
        Platform.SELECT,
        DOMAIN,
        VAD_SENSITIVITY_UNIQUE_ID,
        suggested_object_id=f"{DOMAIN}_vad_sensitivity",
    )
    return row.entity_id


@pytest.mark.usefixtures("mock_llama_client")
async def test_select_unique_id_and_device(hass: HomeAssistant) -> None:
    entry = await _create_entry(hass)
    await hass.async_block_till_done()

    row = er.async_get(hass).async_get(SELECT_ENTITY_ID)
    assert row is not None
    assert row.unique_id == f"{entry.entry_id}-vad-sensitivity"
    assert row.config_entry_id == entry.entry_id
    assert row.platform == DOMAIN

    device = dr.async_get(hass).async_get_device(
        identifiers={(DOMAIN, f"{entry.entry_id}-engine")}
    )
    assert device is not None
    assert row.device_id == device.id
    assert device.name == "SaySo"
    assert device.manufacturer == "SaySo"
    assert device.entry_type == dr.DeviceEntryType.SERVICE


async def test_legacy_v1_entry_rekeys_vad_row_and_keeps_entity_id(
    hass: HomeAssistant,
) -> None:
    registry = er.async_get(hass)
    legacy_entity_id = _legacy_vad_entity_id(hass)
    registry.async_update_entity(legacy_entity_id, name="Kitchen VAD")
    entry = _v1_entry(hass)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.version == 2
    assert entry.minor_version == 1
    assert legacy_entity_id == SELECT_ENTITY_ID
    row = registry.async_get(SELECT_ENTITY_ID)
    assert row.unique_id == f"{entry.entry_id}-vad-sensitivity"
    assert row.config_entry_id == entry.entry_id
    assert row.name == "Kitchen VAD"
    assert (
        registry.async_get_entity_id(Platform.SELECT, DOMAIN, VAD_SENSITIVITY_UNIQUE_ID)
        is None
    )


async def test_migration_rerun_is_idempotent(hass: HomeAssistant) -> None:
    registry = er.async_get(hass)
    _legacy_vad_entity_id(hass)
    entry = _v1_entry(hass)

    assert await async_migrate_entry(hass, entry) is True
    row_after_first = registry.async_get(SELECT_ENTITY_ID)
    snapshot = (row_after_first.unique_id, row_after_first.config_entry_id)

    assert await async_migrate_entry(hass, entry) is True

    row = registry.async_get(SELECT_ENTITY_ID)
    assert (row.unique_id, row.config_entry_id) == snapshot
    assert entry.version == 2
    assert entry.minor_version == 1


async def test_rerun_after_rekey_before_version_bump_is_safe(
    hass: HomeAssistant,
) -> None:
    registry = er.async_get(hass)
    legacy_entity_id = _legacy_vad_entity_id(hass)
    entry = _v1_entry(hass)
    registry.async_update_entity(
        legacy_entity_id,
        new_unique_id=f"{entry.entry_id}-vad-sensitivity",
        config_entry_id=entry.entry_id,
    )

    assert await async_migrate_entry(hass, entry) is True

    row = registry.async_get(SELECT_ENTITY_ID)
    assert row.unique_id == f"{entry.entry_id}-vad-sensitivity"
    assert row.config_entry_id == entry.entry_id
    assert entry.version == 2
    assert entry.minor_version == 1


async def test_v2_entry_is_untouched_by_migration(hass: HomeAssistant) -> None:
    registry = er.async_get(hass)
    _legacy_vad_entity_id(hass)
    entry = _v1_entry(hass, version=2, minor_version=1)

    assert await async_migrate_entry(hass, entry) is True

    row = registry.async_get(SELECT_ENTITY_ID)
    assert row.unique_id == VAD_SENSITIVITY_UNIQUE_ID
    assert row.config_entry_id is None
    assert entry.version == 2
    assert entry.minor_version == 1


@pytest.mark.usefixtures("mock_llama_client")
async def test_setup_rekeys_legacy_row_and_keeps_entity_id(hass: HomeAssistant) -> None:
    registry = er.async_get(hass)
    entry = await _create_entry(hass)
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    registry.async_remove(SELECT_ENTITY_ID)
    legacy_entity_id = _legacy_vad_entity_id(hass)
    registry.async_update_entity(legacy_entity_id, name="Kitchen VAD")
    hass.config_entries.async_update_entry(entry, version=1, minor_version=1)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state.value == "loaded"
    assert entry.version == 2
    assert entry.minor_version == 1
    row = registry.async_get(SELECT_ENTITY_ID)
    assert row.unique_id == f"{entry.entry_id}-vad-sensitivity"
    assert row.config_entry_id == entry.entry_id
    assert row.name == "Kitchen VAD"
    assert hass.states.get(SELECT_ENTITY_ID) is not None

"""Options flow tests for SaySo using pytest-homeassistant-custom-component."""

from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import (
    CONF_ACTION_ALLOWLIST,
    CONF_AREA_IDS,
    CONF_DOMAIN_ALLOWLIST,
    CONF_ENTITY_IDS,
    CONF_EXPOSURE_MODE,
    CONF_TOKEN,
    CONF_URL,
    DEFAULT_OPTIONS,
    DOMAIN,
    EXPOSURE_MODE_ALL,
    EXPOSURE_MODE_AREA,
    EXPOSURE_MODE_ENTITY,
)


def _mock_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_URL: "http://127.0.0.1:8765",
            CONF_TOKEN: "good-token",
        },
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_options_flow_shows_form(hass: HomeAssistant) -> None:
    entry = _mock_entry(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"


@pytest.mark.asyncio
async def test_options_flow_persists_allowlists_and_all_exposure(hass: HomeAssistant) -> None:
    entry = _mock_entry(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_DOMAIN_ALLOWLIST: ["light", "switch"],
            CONF_ACTION_ALLOWLIST: ["on", "off"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ALL,
        },
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options == {
        CONF_DOMAIN_ALLOWLIST: ["light", "switch"],
        CONF_ACTION_ALLOWLIST: ["on", "off"],
        CONF_EXPOSURE_MODE: EXPOSURE_MODE_ALL,
        CONF_AREA_IDS: [],
        CONF_ENTITY_IDS: [],
    }


@pytest.mark.asyncio
async def test_options_flow_area_mode_persists_area_ids(hass: HomeAssistant) -> None:
    entry = _mock_entry(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create("Living Room")
    kitchen = area_reg.async_create("Kitchen")

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_DOMAIN_ALLOWLIST: [],
            CONF_ACTION_ALLOWLIST: [],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_AREA,
            CONF_AREA_IDS: [living_room.id, kitchen.id],
        },
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_EXPOSURE_MODE] == EXPOSURE_MODE_AREA
    assert entry.options[CONF_AREA_IDS] == [living_room.id, kitchen.id]
    assert entry.options[CONF_ENTITY_IDS] == []


@pytest.mark.asyncio
async def test_options_flow_entity_mode_persists_entity_ids(hass: HomeAssistant) -> None:
    entry = _mock_entry(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    entity_reg = er.async_get(hass)
    light = entity_reg.async_get_or_create(
        "light",
        "test",
        "desk",
        suggested_object_id="desk_lamp",
    )

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_DOMAIN_ALLOWLIST: [],
            CONF_ACTION_ALLOWLIST: [],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ENTITY,
            CONF_ENTITY_IDS: [light.entity_id],
        },
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_EXPOSURE_MODE] == EXPOSURE_MODE_ENTITY
    assert entry.options[CONF_ENTITY_IDS] == [light.entity_id]
    assert entry.options[CONF_AREA_IDS] == []


@pytest.mark.asyncio
async def test_options_update_reloads_and_updates_hass_data(hass: HomeAssistant) -> None:
    entry = _mock_entry(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_DOMAIN_ALLOWLIST: ["climate"],
            CONF_ACTION_ALLOWLIST: ["set_temperature"],
            CONF_EXPOSURE_MODE: EXPOSURE_MODE_ALL,
        },
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    stored = hass.data[DOMAIN][entry.entry_id]["options"]
    assert stored[CONF_DOMAIN_ALLOWLIST] == ["climate"]
    assert stored[CONF_ACTION_ALLOWLIST] == ["set_temperature"]
    assert stored[CONF_EXPOSURE_MODE] == EXPOSURE_MODE_ALL


@pytest.mark.asyncio
async def test_setup_entry_applies_default_options_when_missing(hass: HomeAssistant) -> None:
    entry = _mock_entry(hass)

    assert await hass.config_entries.async_setup(entry.entry_id) is True

    assert hass.data[DOMAIN][entry.entry_id]["options"] == DEFAULT_OPTIONS

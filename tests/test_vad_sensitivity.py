from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
import pytest
from homeassistant.const import ATTR_ENTITY_ID, ATTR_OPTION
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers.restore_state import (
    StoredState,
    async_get as async_get_restore_state,
)
from homeassistant.helpers.state import State

from custom_components.sayso.const import (
    VAD_SENSITIVITY_DEFAULT,
    VAD_SENSITIVITY_OPTIONS,
    DOMAIN,
)
from custom_components.sayso.vad_sensitivity import (
    SATELLITE_DOMAIN,
    async_setup_vad_sensitivity,
    patch_satellite_entities,
)
from tests.test_config_flow import (
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)
from tests.test_init import mock_llama_client  # noqa: F401  (fixture re-export)

SELECT_ENTITY_ID = f"select.{DOMAIN}_vad_sensitivity"


async def _create_entry(hass: HomeAssistant) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    return hass.config_entries.async_entries(DOMAIN)[0]


def _add_mock_satellite(hass: HomeAssistant, entity_id: str, with_attr: bool = True) -> Any:
    platform = EntityPlatform(
        hass=hass,
        logger=logging.getLogger(__name__),
        domain=SATELLITE_DOMAIN,
        platform_name=SATELLITE_DOMAIN,
        platform=None,
        scan_interval=timedelta(seconds=15),
        entity_namespace=None,
    )
    platform.async_prepare()
    satellite = SimpleNamespace()
    if with_attr:
        satellite._attr_vad_sensitivity_entity_id = None
    platform.entities[entity_id] = satellite
    return satellite


def _fire_registry_created(hass: HomeAssistant, entity_id: str) -> None:
    hass.bus.async_fire(
        er.EVENT_ENTITY_REGISTRY_UPDATED,
        {"action": "create", "entity_id": entity_id},
    )


@pytest.mark.usefixtures("mock_llama_client")
async def test_select_entity_options_and_default(hass: HomeAssistant) -> None:
    entry = await _create_entry(hass)
    await hass.async_block_till_done()
    assert entry.state.value == "loaded"

    state = hass.states.get(SELECT_ENTITY_ID)
    assert state is not None
    assert state.state == VAD_SENSITIVITY_DEFAULT
    assert state.attributes["options"] == VAD_SENSITIVITY_OPTIONS


@pytest.mark.usefixtures("mock_llama_client")
async def test_select_option_service_updates_state(hass: HomeAssistant) -> None:
    await _create_entry(hass)
    await hass.async_block_till_done()

    await hass.services.async_call(
        "select",
        "select_option",
        {ATTR_ENTITY_ID: SELECT_ENTITY_ID, ATTR_OPTION: "aggressive"},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert hass.states.get(SELECT_ENTITY_ID).state == "aggressive"


async def test_patch_applied_to_mocked_satellite(hass: HomeAssistant) -> None:
    setup = await async_setup_vad_sensitivity(hass)
    await hass.async_block_till_done()

    satellite = _add_mock_satellite(hass, "assist_satellite.test")
    _fire_registry_created(hass, "assist_satellite.test")
    await hass.async_block_till_done()

    assert satellite._attr_vad_sensitivity_entity_id == setup.entity_id
    assert setup.entity_id == SELECT_ENTITY_ID


async def test_patch_noop_when_attribute_missing(hass: HomeAssistant) -> None:
    setup = await async_setup_vad_sensitivity(hass)
    await hass.async_block_till_done()

    satellite = _add_mock_satellite(hass, "assist_satellite.old", with_attr=False)
    _fire_registry_created(hass, "assist_satellite.old")
    await hass.async_block_till_done()

    assert not hasattr(satellite, "_attr_vad_sensitivity_entity_id")
    assert patch_satellite_entities(hass, setup.entity_id) == 0


async def test_patch_idempotent_on_repeated_registry_events(hass: HomeAssistant) -> None:
    setup = await async_setup_vad_sensitivity(hass)
    await hass.async_block_till_done()

    satellite = _add_mock_satellite(hass, "assist_satellite.test")
    _fire_registry_created(hass, "assist_satellite.test")
    await hass.async_block_till_done()
    assert satellite._attr_vad_sensitivity_entity_id == setup.entity_id

    _fire_registry_created(hass, "assist_satellite.test")
    await hass.async_block_till_done()
    assert satellite._attr_vad_sensitivity_entity_id == setup.entity_id
    assert patch_satellite_entities(hass, setup.entity_id) == 0


async def test_no_satellites_is_a_noop(hass: HomeAssistant) -> None:
    setup = await async_setup_vad_sensitivity(hass)
    await hass.async_block_till_done()
    assert patch_satellite_entities(hass, setup.entity_id) == 0
    _fire_registry_created(hass, "assist_satellite.nope")
    await hass.async_block_till_done()
    assert hass.states.get(SELECT_ENTITY_ID).state == VAD_SENSITIVITY_DEFAULT


async def test_setup_unload_entry_removes_select(hass: HomeAssistant, mock_llama_client: None) -> None:
    entry = await _create_entry(hass)
    await hass.async_block_till_done()
    assert hass.states.get(SELECT_ENTITY_ID) is not None

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get(SELECT_ENTITY_ID) is None


@pytest.mark.usefixtures("mock_llama_client")
async def test_selected_option_restored_after_reload(hass: HomeAssistant) -> None:
    entry = await _create_entry(hass)
    await hass.async_block_till_done()

    await hass.services.async_call(
        "select",
        "select_option",
        {ATTR_ENTITY_ID: SELECT_ENTITY_ID, ATTR_OPTION: "relaxed"},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert hass.states.get(SELECT_ENTITY_ID).state == "relaxed"

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get(SELECT_ENTITY_ID).state == "relaxed"


@pytest.mark.usefixtures("mock_llama_client")
async def test_invalid_restored_option_falls_back_to_default(hass: HomeAssistant) -> None:
    restore_store = async_get_restore_state(hass)
    restore_store.last_states[SELECT_ENTITY_ID] = StoredState(
        State(SELECT_ENTITY_ID, "bogus"),
        None,
        datetime.now(),
    )

    entry = await _create_entry(hass)
    await hass.async_block_till_done()
    assert entry.state.value == "loaded"
    assert hass.states.get(SELECT_ENTITY_ID).state == VAD_SENSITIVITY_DEFAULT

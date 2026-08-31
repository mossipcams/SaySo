"""Integration setup tests for SaySo."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DEFAULT_OPTIONS, DOMAIN

PROBE_PATH = "custom_components.sayso.config_flow.probe_connection"


@pytest.mark.asyncio
async def test_setup_entry_stores_credentials_and_loads(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_URL: "http://127.0.0.1:8765",
            CONF_TOKEN: "good-token",
        },
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id) is True
    assert entry.state is ConfigEntryState.LOADED
    stored = hass.data[DOMAIN][entry.entry_id]
    assert stored[CONF_URL] == "http://127.0.0.1:8765"
    assert stored[CONF_TOKEN] == "good-token"
    assert stored["options"] == DEFAULT_OPTIONS
    assert stored["coordinator"].connected is False


@pytest.mark.asyncio
async def test_unload_entry_removes_credentials(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_URL: "http://127.0.0.1:8765",
            CONF_TOKEN: "good-token",
        },
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    assert await hass.config_entries.async_unload(entry.entry_id) is True
    assert entry.entry_id not in hass.data.get(DOMAIN, {})
    assert DOMAIN not in hass.data


@pytest.mark.asyncio
async def test_unload_entry_stops_coordinator_and_clears_connected(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_URL: "http://127.0.0.1:8765",
            CONF_TOKEN: "good-token",
        },
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)

    assert await hass.config_entries.async_unload(entry.entry_id) is True
    assert coordinator.connected is False


@pytest.mark.asyncio
async def test_successful_flow_leaves_entry_loaded(hass: HomeAssistant) -> None:
    from unittest.mock import patch

    from homeassistant import config_entries
    from homeassistant.data_entry_flow import FlowResultType

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )

    with patch(PROBE_PATH, return_value=None):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_URL: "http://127.0.0.1:8765",
                CONF_TOKEN: "good-token",
            },
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert entry.state is ConfigEntryState.LOADED
    assert hass.data[DOMAIN][entry.entry_id][CONF_TOKEN] == "good-token"

"""Device registry and connection entity tests for SaySo."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DEVICE_NAME, DOMAIN


@pytest.mark.asyncio
async def test_setup_creates_device_and_connection_entity(
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

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    device = device_registry.async_get_device(identifiers={(DOMAIN, entry.entry_id)})
    assert device is not None
    assert device.name == DEVICE_NAME

    entity = entity_registry.async_get_entity_id(
        "binary_sensor",
        DOMAIN,
        f"{entry.entry_id}-connection",
    )
    assert entity is not None

    registry_entry = entity_registry.async_get(entity)
    assert registry_entry is not None
    assert registry_entry.device_id == device.id

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    state = hass.states.get(entity)
    assert state is not None
    assert state.state == ("on" if coordinator.connected else "off")

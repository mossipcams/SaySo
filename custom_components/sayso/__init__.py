"""The SaySo integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

PLATFORMS: list[Platform] = []

type SaySoConfigEntry = ConfigEntry[SaySoRuntimeData]


@dataclass
class SaySoRuntimeData:
    """Runtime data stored on the config entry."""

    entry_id: str


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the SaySo integration."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Set up SaySo from a config entry."""
    entry.runtime_data = SaySoRuntimeData(entry_id=entry.entry_id)
    if PLATFORMS:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Unload a SaySo config entry."""
    if PLATFORMS:
        return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return True

"""SaySo Home Assistant integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_TOKEN, CONF_URL, DOMAIN, get_entry_options
from .coordinator import SaySoConnectionCoordinator

type SaySoConfigEntry = ConfigEntry


async def async_setup_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Set up SaySo from a config entry."""

    coordinator = SaySoConnectionCoordinator(hass, entry)
    await coordinator.async_start()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        CONF_URL: entry.data[CONF_URL],
        CONF_TOKEN: entry.data[CONF_TOKEN],
        "options": get_entry_options(entry),
        "coordinator": coordinator,
    }
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Unload a SaySo config entry."""

    entry_data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    if entry_data is not None:
        coordinator: SaySoConnectionCoordinator | None = entry_data.get("coordinator")
        if coordinator is not None:
            await coordinator.async_stop()
    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)
    return True

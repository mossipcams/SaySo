"""SaySo Home Assistant integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.service import async_register_admin_service

from .const import CONF_TOKEN, CONF_URL, DOMAIN, get_entry_options
from .coordinator import SaySoConnectionCoordinator

type SaySoConfigEntry = ConfigEntry

PLATFORMS: list[str] = ["binary_sensor"]
SERVICE_SYNC_HOME_GRAPH = "sync_home_graph"


async def async_setup_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Set up SaySo from a config entry."""

    coordinator = SaySoConnectionCoordinator(hass, entry)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        CONF_URL: entry.data[CONF_URL],
        CONF_TOKEN: entry.data[CONF_TOKEN],
        "options": get_entry_options(entry),
        "coordinator": coordinator,
    }
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    await coordinator.async_start()

    async def handle_sync_home_graph(_call: ServiceCall) -> None:
        await coordinator.async_sync_home_graph()

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_SYNC_HOME_GRAPH,
        handle_sync_home_graph,
    )
    entry.async_on_unload(
        lambda: hass.services.async_remove(DOMAIN, SERVICE_SYNC_HOME_GRAPH),
    )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Unload a SaySo config entry."""

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if not unload_ok:
        return False

    entry_data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    if entry_data is not None:
        coordinator: SaySoConnectionCoordinator | None = entry_data.get("coordinator")
        if coordinator is not None:
            await coordinator.async_stop()
    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)
    return True

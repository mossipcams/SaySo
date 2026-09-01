"""Diagnostics support for SaySo."""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_registry import async_get as er_async_get

from .const import (
    API_VERSION,
    CONF_EXPOSURE_MODE,
    CONF_TOKEN,
    DOMAIN,
    WS_PATH,
    get_entry_options,
)
from .coordinator import SaySoConnectionCoordinator
from .exposure import is_entity_exposed

REDACT_KEYS = {CONF_TOKEN}


def _count_exposed_entities(
    hass: HomeAssistant,
    options: dict[str, list[str] | str],
) -> int:
    return sum(
        1
        for entry in er_async_get(hass).entities.values()
        if is_entity_exposed(hass, entry, options)
    )


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""

    entry_data = hass.data.get(DOMAIN, {}).get(config_entry.entry_id, {})
    coordinator: SaySoConnectionCoordinator | None = entry_data.get("coordinator")
    options = get_entry_options(config_entry)
    connected = coordinator.connected if coordinator is not None else False
    sequence = coordinator.sequence if coordinator is not None else 0

    diag: dict[str, Any] = {
        "config": config_entry.as_dict(),
        "health": {
            "connected": connected,
        },
        "exposure": {
            "mode": options[CONF_EXPOSURE_MODE],
            "entity_count": _count_exposed_entities(hass, options),
        },
        "protocol": {
            "api_version": API_VERSION,
            "ws_path": WS_PATH,
            "connected": connected,
            "sequence": sequence,
        },
    }

    return async_redact_data(diag, REDACT_KEYS)

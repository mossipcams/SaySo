"""Entity exposure filtering for SaySo Home Graph payloads."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_registry import RegistryEntry, async_get as er_async_get

from .const import (
    CONF_AREA_IDS,
    CONF_ENTITY_IDS,
    CONF_EXPOSURE_MODE,
    EXPOSURE_MODE_ALL,
    EXPOSURE_MODE_AREA,
    EXPOSURE_MODE_ENTITY,
)


def is_entity_exposed(
    hass: HomeAssistant,
    entry: RegistryEntry,
    options: dict[str, list[str] | str],
) -> bool:
    """Return whether an entity registry entry is exposed to SaySo."""

    mode = options[CONF_EXPOSURE_MODE]
    if mode == EXPOSURE_MODE_ALL:
        return True
    if mode == EXPOSURE_MODE_ENTITY:
        return entry.entity_id in options[CONF_ENTITY_IDS]
    if mode == EXPOSURE_MODE_AREA:
        allowed_areas = set(options[CONF_AREA_IDS])
        if entry.area_id is not None and entry.area_id in allowed_areas:
            return True
        if entry.device_id is not None:
            device = dr.async_get(hass).async_get(entry.device_id)
            if device is not None and device.area_id in allowed_areas:
                return True
        return False
    return True


def is_entity_id_exposed(
    hass: HomeAssistant,
    entity_id: str,
    options: dict[str, list[str] | str],
    *,
    entry: RegistryEntry | None = None,
) -> bool:
    """Return whether an entity id is exposed, resolving registry metadata when needed."""

    if entry is None:
        entry = er_async_get(hass).async_get(entity_id)
    if entry is None:
        return options[CONF_EXPOSURE_MODE] == EXPOSURE_MODE_ALL
    return is_entity_exposed(hass, entry, options)

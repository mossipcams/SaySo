from __future__ import annotations

import logging
from collections.abc import Callable

from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import async_get_platforms

_LOGGER = logging.getLogger(__name__)

SATELLITE_DOMAIN = "assist_satellite"

REGISTRY_EVENTS = tuple(
    event
    for event in (
        getattr(er, "EVENT_ENTITY_CREATED", "entity_registry_created"),
        getattr(er, "EVENT_ENTITY_UPDATED", "entity_registry_updated"),
        getattr(er, "EVENT_ENTITY_REGISTRY_UPDATED", "entity_registry_updated"),
    )
    if event
)


def patch_satellite_entities(hass: HomeAssistant, select_entity_id: str) -> int:
    patched = 0
    for platform in async_get_platforms(hass, SATELLITE_DOMAIN):
        for entity in platform.entities.values():
            if not hasattr(entity, "_attr_vad_sensitivity_entity_id"):
                continue
            if entity._attr_vad_sensitivity_entity_id == select_entity_id:
                continue
            entity._attr_vad_sensitivity_entity_id = select_entity_id
            patched += 1
    return patched


@callback
def async_watch_satellites(hass: HomeAssistant, select_entity_id: str) -> Callable[[], None]:
    @callback
    def _async_handle_registry_event(event: Event) -> None:
        data = event.data
        action = data.get("action")
        if action is not None and action not in ("create", "update"):
            return
        entity_id = data.get("entity_id", "")
        if not entity_id.startswith(f"{SATELLITE_DOMAIN}."):
            return
        try:
            patched = patch_satellite_entities(hass, select_entity_id)
        except Exception:
            _LOGGER.exception("Failed to patch satellite VAD sensitivity")
            return
        if patched:
            _LOGGER.debug("Patched %d satellite entit%s", patched, "y" if patched == 1 else "ies")

    unsubs = [
        hass.bus.async_listen(event, _async_handle_registry_event)
        for event in REGISTRY_EVENTS
    ]
    patch_satellite_entities(hass, select_entity_id)

    @callback
    def _async_unsub() -> None:
        for unsub in unsubs:
            unsub()

    return _async_unsub

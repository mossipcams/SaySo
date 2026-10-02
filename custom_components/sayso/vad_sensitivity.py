from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from homeassistant.components.select import DOMAIN as SELECT_DOMAIN
from homeassistant.components.select import SelectEntity
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import (
    EntityPlatform,
    async_get_platforms,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.setup import async_setup_component

from .const import (
    VAD_SENSITIVITY_DEFAULT,
    VAD_SENSITIVITY_OPTIONS,
    VAD_SENSITIVITY_UNIQUE_ID,
)

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


class SaySoVadSensitivitySelect(SelectEntity, RestoreEntity):
    _attr_entity_category = EntityCategory.CONFIG
    _attr_name = "SaySo VAD sensitivity"
    _attr_should_poll = False
    _attr_options = list(VAD_SENSITIVITY_OPTIONS)
    _attr_current_option = VAD_SENSITIVITY_DEFAULT
    _attr_unique_id = VAD_SENSITIVITY_UNIQUE_ID

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()
        if last_state is not None and last_state.state in VAD_SENSITIVITY_OPTIONS:
            self._attr_current_option = last_state.state

    async def async_select_option(self, option: str) -> None:
        self._attr_current_option = option
        self.async_write_ha_state()


@dataclass
class VadSensitivitySetup:
    platform: EntityPlatform
    entity_id: str
    unsub: object


def patch_satellite_entities(hass: HomeAssistant, select_entity_id: str) -> int:
    patched = 0
    for platform in async_get_platforms(hass, SATELLITE_DOMAIN):
        for entity in platform.entities.values():
            if not hasattr(entity, "_attr_vad_sensitivity_entity_id"):
                continue
            if getattr(entity, "_attr_vad_sensitivity_entity_id") == select_entity_id:
                continue
            entity._attr_vad_sensitivity_entity_id = select_entity_id
            patched += 1
    return patched


async def async_setup_vad_sensitivity(hass: HomeAssistant) -> VadSensitivitySetup:
    if not await async_setup_component(hass, SELECT_DOMAIN, {SELECT_DOMAIN: {}}):
        raise RuntimeError(f"Could not set up the {SELECT_DOMAIN} component")

    platform = EntityPlatform(
        hass=hass,
        logger=_LOGGER,
        domain=SELECT_DOMAIN,
        platform_name="sayso",
        platform=None,
        scan_interval=timedelta(seconds=15),
        entity_namespace=None,
    )
    entity = SaySoVadSensitivitySelect()
    await platform.async_add_entities([entity])
    select_entity_id = entity.entity_id
    _LOGGER.info("VAD sensitivity select entity: %s", select_entity_id)

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
        except Exception:  # noqa: BLE001 - a patch failure must not break the bus
            _LOGGER.exception("Failed to patch satellite VAD sensitivity")
            return
        if patched:
            _LOGGER.debug("Patched %d satellite entit%s", patched, "y" if patched == 1 else "ies")

    unsubs = [
        hass.bus.async_listen(event, _async_handle_registry_event)
        for event in REGISTRY_EVENTS
    ]

    @callback
    def _async_unsub() -> None:
        for unsub in unsubs:
            unsub()

    unsub = _async_unsub
    patch_satellite_entities(hass, select_entity_id)
    return VadSensitivitySetup(platform=platform, entity_id=select_entity_id, unsub=unsub)


async def async_teardown_vad_sensitivity(hass: HomeAssistant, setup: VadSensitivitySetup) -> None:
    setup.unsub()
    registry = er.async_get(hass)
    if registry.async_get(setup.entity_id) is not None:
        registry.async_remove(setup.entity_id)
        await hass.async_block_till_done()
    else:
        await setup.platform.async_remove_entity(setup.entity_id)

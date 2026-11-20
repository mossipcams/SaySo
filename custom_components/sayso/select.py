from __future__ import annotations

from functools import partial

from homeassistant.components.select import SelectEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from . import SaySoConfigEntry
from .const import (
    DOMAIN,
    VAD_SENSITIVITY_DEFAULT,
    VAD_SENSITIVITY_OPTIONS,
)
from .vad_sensitivity import async_watch_satellites


@callback
def _async_drop_state(hass: HomeAssistant, entity_id: str) -> None:
    hass.states.async_remove(entity_id)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SaySoConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([SaySoVadSensitivitySelect(entry)])


class SaySoVadSensitivitySelect(SelectEntity, RestoreEntity):
    _attr_entity_category = EntityCategory.CONFIG
    _attr_name = "SaySo VAD sensitivity"
    _attr_should_poll = False
    _attr_options = list(VAD_SENSITIVITY_OPTIONS)
    _attr_current_option = VAD_SENSITIVITY_DEFAULT

    def __init__(self, entry: SaySoConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}-vad-sensitivity"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}-engine")},
            name="SaySo",
            manufacturer="SaySo",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()
        if last_state is not None and last_state.state in VAD_SENSITIVITY_OPTIONS:
            self._attr_current_option = last_state.state
        self.async_on_remove(async_watch_satellites(self.hass, self.entity_id))
        self._entry.async_on_unload(partial(_async_drop_state, self.hass, self.entity_id))

    async def async_select_option(self, option: str) -> None:
        self._attr_current_option = option
        self.async_write_ha_state()

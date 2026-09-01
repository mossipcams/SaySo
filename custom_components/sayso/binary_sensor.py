"""Binary sensor platform for SaySo."""

from __future__ import annotations

from collections.abc import Callable

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DEVICE_NAME, DOMAIN
from .coordinator import SaySoConnectionCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up SaySo connection binary sensors."""

    coordinator: SaySoConnectionCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    async_add_entities([SaySoConnectionBinarySensor(coordinator, entry)])


class SaySoConnectionBinarySensor(BinarySensorEntity):
    """Report whether the SaySo outbound WebSocket is connected."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_has_entity_name = True
    _attr_name = "Connection"

    def __init__(
        self,
        coordinator: SaySoConnectionCoordinator,
        entry: ConfigEntry,
    ) -> None:
        self.coordinator = coordinator
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}-connection"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=DEVICE_NAME,
            manufacturer="SaySo",
            model="Voice Assistant",
        )
        self._unsub_connection: Callable[[], None] | None = None

    async def async_added_to_hass(self) -> None:
        """Register for coordinator connection updates."""

        await super().async_added_to_hass()
        self._unsub_connection = self.coordinator.async_add_connection_listener(
            self._handle_connection_update,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Drop coordinator listener."""

        if self._unsub_connection is not None:
            self._unsub_connection()
            self._unsub_connection = None
        await super().async_will_remove_from_hass()

    @callback
    def _handle_connection_update(self) -> None:
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool:
        return self.coordinator.connected

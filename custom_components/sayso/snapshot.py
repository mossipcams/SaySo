"""Build Home Graph snapshots from Home Assistant registries."""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers.entity_registry import COMPUTED_NAME, RegistryEntry

from .capabilities import (
    entity_capabilities_from_state,
    scene_capabilities,
    script_capabilities,
)
from .const import API_VERSION, SNAPSHOT_OMIT_ATTRIBUTES
from .exposure import is_entity_exposed


def build_home_graph_snapshot(
    hass: HomeAssistant,
    *,
    home_id: str,
    sequence: int,
    options: dict[str, list[str] | str] | None = None,
    version: int = API_VERSION,
) -> dict[str, Any]:
    """Serialize floor/area/device/entity registries into a Home Graph snapshot."""

    floor_reg = fr.async_get(hass)
    area_reg = ar.async_get(hass)
    device_reg = dr.async_get(hass)
    entity_reg = er.async_get(hass)

    floors = [
        {
            "id": floor.floor_id,
            "name": floor.name,
            "aliases": sorted(floor.aliases),
        }
        for floor in sorted(floor_reg.async_list_floors(), key=lambda item: item.floor_id)
    ]

    areas = [
        {
            "id": area.id,
            "name": area.name,
            "floor_id": area.floor_id,
            "aliases": sorted(area.aliases),
        }
        for area in sorted(area_reg.async_list_areas(), key=lambda item: item.id)
    ]

    devices = [
        {
            "id": device.id,
            "name": device.name_by_user or device.name or device.id,
            **(
                {"manufacturer": device.manufacturer}
                if device.manufacturer is not None
                else {}
            ),
            **({"model": device.model} if device.model is not None else {}),
            **({"area_id": device.area_id} if device.area_id is not None else {}),
        }
        for device in sorted(device_reg.devices.values(), key=lambda item: item.id)
    ]

    entities: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    scripts: list[dict[str, Any]] = []

    for entry in sorted(entity_reg.entities.values(), key=lambda item: item.entity_id):
        if options is not None and not is_entity_exposed(hass, entry, options):
            continue
        if entry.domain == "scene":
            scenes.append(_serialize_scene_or_script(hass, entry))
        elif entry.domain == "script":
            scripts.append(_serialize_scene_or_script(hass, entry))
        else:
            state = hass.states.get(entry.entity_id)
            entities.append(
                {
                    **_serialize_registry_entity(hass, entry),
                    "state": _serialize_state(state),
                },
            )

    return {
        "version": version,
        "sequence": sequence,
        "home_id": home_id,
        "floors": floors,
        "areas": areas,
        "devices": devices,
        "entities": entities,
        "scenes": scenes,
        "scripts": scripts,
    }


def _effective_area_id(hass: HomeAssistant, entry: RegistryEntry) -> str | None:
    """Return entity area, else inherit from the linked device."""

    if entry.area_id is not None:
        return entry.area_id
    if entry.device_id is None:
        return None
    device = dr.async_get(hass).async_get(entry.device_id)
    if device is None or device.area_id is None:
        return None
    return device.area_id


def _serialize_registry_entity(
    hass: HomeAssistant,
    entry: RegistryEntry,
) -> dict[str, Any]:
    """Serialize entity registry fields for standard entities."""

    payload: dict[str, Any] = {
        "entity_id": entry.entity_id,
        "domain": entry.domain,
        "name": _entity_display_name(hass, entry),
        "aliases": _entity_aliases(hass, entry),
        "capabilities": entity_capabilities_from_state(
            entry.domain,
            hass.states.get(entry.entity_id),
        ),
    }
    area_id = _effective_area_id(hass, entry)
    if area_id is not None:
        payload["area_id"] = area_id
    if entry.device_id is not None:
        payload["device_id"] = entry.device_id
    return payload


def _serialize_scene_or_script(
    hass: HomeAssistant,
    entry: RegistryEntry,
) -> dict[str, Any]:
    """Serialize scene/script registry fields (no domain or state)."""

    payload: dict[str, Any] = {
        "entity_id": entry.entity_id,
        "name": _entity_display_name(hass, entry),
        "aliases": _entity_aliases(hass, entry),
        "capabilities": (
            scene_capabilities()
            if entry.domain == "scene"
            else script_capabilities()
        ),
    }
    area_id = _effective_area_id(hass, entry)
    if area_id is not None:
        payload["area_id"] = area_id
    return payload


def _entity_aliases(hass: HomeAssistant, entry: RegistryEntry) -> list[str]:
    display_name = _entity_display_name(hass, entry)
    return [
        alias.strip()
        for alias in entry.aliases
        if alias is not COMPUTED_NAME
        and alias.strip()
        and alias.strip() != display_name
    ]


def _entity_display_name(hass: HomeAssistant, entry: RegistryEntry) -> str:
    if entry.name:
        return entry.name
    if entry.original_name:
        return entry.original_name
    state = hass.states.get(entry.entity_id)
    if state is not None and state.name:
        return state.name
    return entry.entity_id


def _json_safe(value: Any) -> Any:
    """Coerce HA state attribute values into JSON-serializable data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _serialize_state(state: State | None) -> dict[str, Any]:
    if state is None:
        return {"value": "unavailable", "attributes": {}}

    attributes: dict[str, Any] = {}
    for key, value in state.attributes.items():
        if key in SNAPSHOT_OMIT_ATTRIBUTES:
            continue
        attributes[key] = _json_safe(value)

    return {
        "value": state.state,
        "attributes": attributes,
    }

"""Build incremental Home Graph delta payloads."""

from __future__ import annotations

from typing import Any, Literal

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.entity_registry import RegistryEntry

from .snapshot import (
    _serialize_registry_entity,
    _serialize_scene_or_script,
    _serialize_state,
)

RegistryChange = Literal["create", "update", "remove"]


def build_state_delta(
    *,
    home_id: str,
    sequence: int,
    entity_id: str,
    state: State | None,
    version: int = 1,
) -> dict[str, Any]:
    """Serialize a single entity state change."""

    return {
        "version": version,
        "home_id": home_id,
        "sequence": sequence,
        "entity_id": entity_id,
        "state": _serialize_state(state),
    }


def build_registry_delta(
    hass: HomeAssistant,
    *,
    home_id: str,
    sequence: int,
    change: RegistryChange,
    entry: RegistryEntry | None,
    entity_id: str,
    version: int = 1,
) -> dict[str, Any]:
    """Serialize a single entity registry change."""

    payload: dict[str, Any] = {
        "version": version,
        "home_id": home_id,
        "sequence": sequence,
        "change": change,
        "entity_id": entity_id,
    }
    if change != "remove" and entry is not None:
        if entry.domain in {"scene", "script"}:
            payload["entity"] = _serialize_scene_or_script(hass, entry)
        else:
            state = hass.states.get(entry.entity_id)
            payload["entity"] = {
                **_serialize_registry_entity(hass, entry),
                "state": _serialize_state(state),
            }
    return payload

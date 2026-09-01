"""Pure Home Graph mutation helpers."""

from __future__ import annotations

from sayso_server.deltas import RegistryChange
from sayso_server.home_graph import Entity, HomeGraphSnapshot, Scene, Script, State

EntityRecord = Entity | Scene | Script


def apply_state_delta(
    snapshot: HomeGraphSnapshot,
    *,
    entity_id: str,
    state: State | None,
) -> bool:
    """Update one entity state in-place. Returns False when the entity is missing."""

    if state is None:
        return False

    for entity in snapshot.entities:
        if entity.entity_id == entity_id:
            entity.state = state
            return True
    return False


def _replace_in_list(items: list[EntityRecord], entity_id: str, replacement: EntityRecord) -> None:
    for index, item in enumerate(items):
        if item.entity_id == entity_id:
            items[index] = replacement
            return
    items.append(replacement)


def _remove_from_list(items: list[EntityRecord], entity_id: str) -> None:
    items[:] = [item for item in items if item.entity_id != entity_id]


def apply_registry_delta(
    snapshot: HomeGraphSnapshot,
    *,
    change: RegistryChange,
    entity_id: str,
    entity: EntityRecord | None,
) -> bool:
    """Apply a registry create/update/remove in-place."""

    if change == "remove":
        _remove_from_list(snapshot.entities, entity_id)
        _remove_from_list(snapshot.scenes, entity_id)
        _remove_from_list(snapshot.scripts, entity_id)
        return True

    if entity is None:
        return False

    if entity_id.startswith("scene."):
        if change == "create":
            snapshot.scenes.append(entity)  # type: ignore[arg-type]
        else:
            _replace_in_list(snapshot.scenes, entity_id, entity)  # type: ignore[arg-type]
        return True

    if entity_id.startswith("script."):
        if change == "create":
            snapshot.scripts.append(entity)  # type: ignore[arg-type]
        else:
            _replace_in_list(snapshot.scripts, entity_id, entity)  # type: ignore[arg-type]
        return True

    if not isinstance(entity, Entity):
        return False

    if change == "create":
        snapshot.entities.append(entity)
    else:
        _replace_in_list(snapshot.entities, entity_id, entity)
    return True

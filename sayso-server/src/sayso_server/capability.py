"""Per-target capability and range validation against Home Graph state."""

from __future__ import annotations

from sayso_server.home_graph import (
    Capability,
    CapabilityKind,
    Entity,
    HomeGraphSnapshot,
    Scene,
    Script,
)
from sayso_server.models import ActionState, ClimateMode

GraphItem = Entity | Scene | Script


class CapabilityValidationError(Exception):
    """Raised when a target entity lacks a required capability or value is out of range."""

    def __init__(self, entity_id: str, message: str) -> None:
        self.entity_id = entity_id
        super().__init__(f"{entity_id}: {message}")


def validate_target_capabilities(
    snapshot: HomeGraphSnapshot,
    entity_ids: frozenset[str],
    *,
    value: float | int | None = None,
    state: ActionState | None = None,
    mode: ClimateMode | None = None,
) -> None:
    """Validate every target supports the requested action before execution.

    Raises CapabilityValidationError when any target is invalid; the caller must
    reject the whole target set atomically.
    """
    items_by_id = _items_by_entity_id(snapshot)
    for entity_id in sorted(entity_ids):
        item = items_by_id.get(entity_id)
        if item is None:
            raise CapabilityValidationError(entity_id, "entity not found in home graph")
        _validate_item(item, value=value, state=state, mode=mode)


def _validate_item(
    item: GraphItem,
    *,
    value: float | int | None,
    state: ActionState | None,
    mode: ClimateMode | None,
) -> None:
    if value is not None:
        _validate_value(item, float(value))
    if state is not None:
        _validate_state(item, state)
    if mode is not None:
        _validate_mode(item)


def _validate_value(item: GraphItem, value: float) -> None:
    if isinstance(item, Entity) and item.domain == "light":
        capability = _capability(item, CapabilityKind.BRIGHTNESS)
        if capability is None:
            raise CapabilityValidationError(item.entity_id, "entity does not support brightness")
        _validate_range(item.entity_id, capability, value)
        return

    if isinstance(item, Entity) and item.domain == "climate":
        capability = _capability(item, CapabilityKind.TEMPERATURE)
        if capability is None:
            raise CapabilityValidationError(item.entity_id, "entity does not support temperature")
        _validate_range(item.entity_id, capability, value)
        return

    raise CapabilityValidationError(item.entity_id, "entity does not support numeric values")


def _validate_state(item: GraphItem, state: ActionState) -> None:
    if state is ActionState.ACTIVATE:
        if isinstance(item, Scene):
            _require_capability(item, CapabilityKind.SCENE)
            return
        if isinstance(item, Script):
            _require_capability(item, CapabilityKind.SCRIPT)
            return

    _require_capability(item, CapabilityKind.POWER)


def _validate_mode(item: GraphItem) -> None:
    if not isinstance(item, Entity) or item.domain != "climate":
        raise CapabilityValidationError(item.entity_id, "entity does not support climate mode")
    _require_capability(item, CapabilityKind.TEMPERATURE)


def _require_capability(item: GraphItem, kind: CapabilityKind) -> None:
    if _capability(item, kind) is None:
        raise CapabilityValidationError(item.entity_id, f"entity does not support {kind.value}")


def _capability(item: GraphItem, kind: CapabilityKind) -> Capability | None:
    for capability in item.capabilities:
        if capability.kind == kind:
            return capability
    return None


def _validate_range(entity_id: str, capability: Capability, value: float) -> None:
    if capability.min_value is not None and value < capability.min_value:
        raise CapabilityValidationError(
            entity_id,
            f"value {value} is below minimum {capability.min_value}",
        )
    if capability.max_value is not None and value > capability.max_value:
        raise CapabilityValidationError(
            entity_id,
            f"value {value} is above maximum {capability.max_value}",
        )


def _items_by_entity_id(snapshot: HomeGraphSnapshot) -> dict[str, GraphItem]:
    return {
        item.entity_id: item
        for item in (*snapshot.entities, *snapshot.scenes, *snapshot.scripts)
    }

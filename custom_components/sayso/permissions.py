"""Action permission checks for inbound SaySo action requests."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.core import HomeAssistant

from .capabilities import action_capability_kind, entity_capabilities_for_entity
from .const import (
    CONF_ACTION_ALLOWLIST,
    CONF_DOMAIN_ALLOWLIST,
    REJECT_ACTION_NOT_ALLOWED,
    REJECT_CAPABILITY_NOT_SUPPORTED,
    REJECT_DOMAIN_MISMATCH,
    REJECT_DOMAIN_NOT_ALLOWED,
    REJECT_ENTITY_NOT_EXPOSED,
)
from .exposure import is_entity_id_exposed


@dataclass(frozen=True, slots=True)
class ActionPermissionResult:
    """Outcome of validating an action request against integration policy."""

    allowed: bool
    reason: str | None = None


def entity_domain_from_id(entity_id: str) -> str | None:
    """Return the Home Assistant domain prefix from an entity_id, if present."""

    domain, _, _object_id = entity_id.partition(".")
    if not domain or not _object_id:
        return None
    return domain


def validate_action_permission(
    hass: HomeAssistant,
    options: dict[str, list[str] | str],
    *,
    entity_id: str,
    domain: str,
    action: str,
) -> ActionPermissionResult:
    """Return whether an action request may proceed to Home Assistant execution."""

    if not is_entity_id_exposed(hass, entity_id, options):
        return ActionPermissionResult(False, REJECT_ENTITY_NOT_EXPOSED)

    entity_domain = entity_domain_from_id(entity_id)
    if entity_domain is None:
        return ActionPermissionResult(False, "invalid_request")

    if domain != entity_domain:
        return ActionPermissionResult(False, REJECT_DOMAIN_MISMATCH)

    domain_allowlist = options[CONF_DOMAIN_ALLOWLIST]
    if domain_allowlist and entity_domain not in domain_allowlist:
        return ActionPermissionResult(False, REJECT_DOMAIN_NOT_ALLOWED)

    action_allowlist = options[CONF_ACTION_ALLOWLIST]
    if action_allowlist and action not in action_allowlist:
        return ActionPermissionResult(False, REJECT_ACTION_NOT_ALLOWED)

    required_capability = action_capability_kind(action)
    if required_capability is None:
        return ActionPermissionResult(False, REJECT_CAPABILITY_NOT_SUPPORTED)

    exposed_kinds = {
        capability["kind"]
        for capability in entity_capabilities_for_entity(
            hass,
            entity_id,
            entity_domain,
        )
    }
    if required_capability not in exposed_kinds:
        return ActionPermissionResult(False, REJECT_CAPABILITY_NOT_SUPPORTED)

    return ActionPermissionResult(True)

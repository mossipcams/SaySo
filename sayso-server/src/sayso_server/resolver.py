"""Resolve scopes and explicit targets to entity-id sets."""

from __future__ import annotations

from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope
from sayso_server.scope import expand_scope


def resolve_entity_ids(
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    scope: Scope | None = None,
    entity_ids: list[str] | None = None,
) -> frozenset[str]:
    """Return the exact entity-id set for explicit targets or scope expansion."""
    if entity_ids:
        return frozenset(sorted(entity_ids))
    if scope is None:
        return frozenset()
    return expand_scope(snapshot, origin_area_id, scope)

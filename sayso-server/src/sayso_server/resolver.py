"""Resolve scopes and explicit targets to entity-id sets."""

from __future__ import annotations

from sayso_server.exclusions import apply_inclusions_exclusions, filter_entity_ids_by_domain
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope
from sayso_server.scope import expand_scope


def resolve_entity_ids(
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    scope: Scope | None = None,
    entity_ids: list[str] | None = None,
    domain: str | None = None,
    targets: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> frozenset[str]:
    """Return the exact entity-id set for scope expansion with include/exclude names."""
    if entity_ids:
        base = frozenset(sorted(entity_ids))
    elif scope is None:
        base = frozenset()
    else:
        base = expand_scope(snapshot, origin_area_id, scope)

    if domain is not None:
        base = filter_entity_ids_by_domain(snapshot, base, domain)

    return apply_inclusions_exclusions(
        snapshot,
        base,
        targets=targets,
        include=include,
        exclude=exclude,
    )

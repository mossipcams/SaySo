"""Deterministic Home Graph scope expansion."""

from __future__ import annotations

from sayso_server.home_graph import Area, Floor, HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.normalize import normalize_tokens
from sayso_server.scoring import CandidateItem, lookup_origin


def expand_scope(
    snapshot: HomeGraphSnapshot,
    origin_area_id: str,
    scope: Scope,
) -> frozenset[str]:
    """Expand a scope to the exact entity-id set for entities, scenes, and scripts."""
    if scope.kind == ScopeKind.CURRENT_AREA:
        origin_area, _ = lookup_origin(snapshot, origin_area_id)
        if origin_area is None:
            return frozenset()
        return _entity_ids_in_areas(snapshot, {origin_area.id})

    if scope.kind == ScopeKind.NAMED_AREA:
        assert scope.name is not None
        area = _find_area(snapshot, scope.name)
        if area is None:
            return frozenset()
        return _entity_ids_in_areas(snapshot, {area.id})

    if scope.kind == ScopeKind.FLOOR:
        assert scope.name is not None
        floor = _find_floor(snapshot, scope.name)
        if floor is None:
            return frozenset()
        area_ids = {area.id for area in snapshot.areas if area.floor_id == floor.id}
        return _entity_ids_in_areas(snapshot, area_ids)

    if scope.kind == ScopeKind.ALL:
        return _all_entity_ids(snapshot)

    return frozenset()


def _normalized_label(label: str) -> str:
    return " ".join(normalize_tokens(label))


def _find_area(snapshot: HomeGraphSnapshot, name: str) -> Area | None:
    for area in snapshot.areas:
        if area.id == name:
            return area
    needle = _normalized_label(name)
    for area in snapshot.areas:
        for label in (area.name, *area.aliases):
            if _normalized_label(label) == needle:
                return area
    return None


def _find_floor(snapshot: HomeGraphSnapshot, name: str) -> Floor | None:
    for floor in snapshot.floors:
        if floor.id == name:
            return floor
    needle = _normalized_label(name)
    for floor in snapshot.floors:
        for label in (floor.name, *floor.aliases):
            if _normalized_label(label) == needle:
                return floor
    return None


def _iter_candidates(snapshot: HomeGraphSnapshot) -> list[CandidateItem]:
    return [*snapshot.entities, *snapshot.scenes, *snapshot.scripts]


def _entity_ids_in_areas(snapshot: HomeGraphSnapshot, area_ids: set[str]) -> frozenset[str]:
    ids = sorted(
        item.entity_id
        for item in _iter_candidates(snapshot)
        if item.area_id is not None and item.area_id in area_ids
    )
    return frozenset(ids)


def _all_entity_ids(snapshot: HomeGraphSnapshot) -> frozenset[str]:
    return frozenset(sorted(item.entity_id for item in _iter_candidates(snapshot)))

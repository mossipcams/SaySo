"""Scope expansion tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.resolver import resolve_entity_ids

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"

LIVING_ROOM_ENTITY_IDS = frozenset(
    {
        "light.living_room_ceiling",
        "light.floor_lamp",
        "climate.downstairs",
        "binary_sensor.front_door",
        "scene.movie_time",
    }
)

GROUND_FLOOR_ENTITY_IDS = LIVING_ROOM_ENTITY_IDS

UPPER_FLOOR_ENTITY_IDS = frozenset({"script.good_night"})


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def test_current_area_resolves_origin_entities() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
    )

    assert result == LIVING_ROOM_ENTITY_IDS


def test_named_area_resolves_by_name() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.NAMED_AREA, name="Living Room"),
    )

    assert result == LIVING_ROOM_ENTITY_IDS


def test_named_area_resolves_by_alias() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.NAMED_AREA, name="lounge"),
    )

    assert result == LIVING_ROOM_ENTITY_IDS


def test_named_area_empty_when_area_has_no_entities() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.NAMED_AREA, name="kitchen"),
    )

    assert result == frozenset()


def test_floor_resolves_by_alias_upstairs() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.FLOOR, name="upstairs"),
    )

    assert result == UPPER_FLOOR_ENTITY_IDS


def test_floor_resolves_by_alias_downstairs() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.FLOOR, name="downstairs"),
    )

    assert result == GROUND_FLOOR_ENTITY_IDS


def test_floor_resolves_by_floor_id() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.FLOOR, name="floor_upper"),
    )

    assert result == UPPER_FLOOR_ENTITY_IDS


def test_explicit_entity_ids_return_exact_set() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        entity_ids=["light.floor_lamp", "script.good_night"],
    )

    assert result == frozenset({"light.floor_lamp", "script.good_night"})


def test_explicit_entity_ids_ignore_scope() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        entity_ids=["script.good_night"],
    )

    assert result == frozenset({"script.good_night"})


def test_scope_expansion_is_deterministic() -> None:
    graph = _load_graph()
    scope = Scope(kind=ScopeKind.FLOOR, name="downstairs")

    first = resolve_entity_ids(graph, origin_area_id="area_living_room", scope=scope)
    second = resolve_entity_ids(graph, origin_area_id="area_living_room", scope=scope)

    assert first == second
    assert tuple(sorted(first)) == tuple(sorted(first))


def test_unknown_named_area_returns_empty_set() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.NAMED_AREA, name="garage"),
    )

    assert result == frozenset()

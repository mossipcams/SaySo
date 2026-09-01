"""Include/exclude resolution tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.resolver import resolve_entity_ids

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"

LIVING_ROOM_LIGHTS = frozenset({"light.living_room_ceiling", "light.floor_lamp"})


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def test_exclude_floor_lamp_from_lights_in_current_area() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        domain="light",
        exclude=["floor lamp"],
    )

    assert result == frozenset({"light.living_room_ceiling"})


def test_exclude_by_alias_does_not_select_excluded_device() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        domain="light",
        exclude=["lamp"],
    )

    assert "light.floor_lamp" not in result
    assert result == frozenset({"light.living_room_ceiling"})


def test_include_narrows_scope_to_named_entities() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        include=["ceiling lights"],
    )

    assert result == frozenset({"light.living_room_ceiling"})


def test_targets_resolve_within_scope() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        targets=["floor lamp", "movie time"],
    )

    assert result == frozenset({"light.floor_lamp", "scene.movie_time"})


def test_exclude_unknown_name_leaves_set_unchanged() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        domain="light",
        exclude=["garage door"],
    )

    assert result == LIVING_ROOM_LIGHTS


def test_exclude_applies_to_explicit_entity_ids() -> None:
    graph = _load_graph()

    result = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        entity_ids=["light.living_room_ceiling", "light.floor_lamp"],
        exclude=["floor lamp"],
    )

    assert result == frozenset({"light.living_room_ceiling"})


def test_include_exclude_resolution_is_deterministic() -> None:
    graph = _load_graph()
    kwargs = {
        "origin_area_id": "area_living_room",
        "scope": Scope(kind=ScopeKind.CURRENT_AREA),
        "domain": "light",
        "exclude": ["lamp"],
    }

    first = resolve_entity_ids(graph, **kwargs)
    second = resolve_entity_ids(graph, **kwargs)

    assert first == second
    assert tuple(sorted(first)) == tuple(sorted(first))

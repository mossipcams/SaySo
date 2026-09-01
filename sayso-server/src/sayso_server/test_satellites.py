"""Tests for satellite registration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.const import (
    DEFAULT_SATELLITE_AREA_ID,
    DEFAULT_SATELLITE_ID,
    SATELLITE_AREA_ID_ENV_VAR,
)
from sayso_server.home_graph import Area, HomeGraphSnapshot
from sayso_server.satellites import (
    SatelliteRegistry,
    default_satellite_area_id,
    register_default_satellites,
)

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _ha_living_room_graph() -> HomeGraphSnapshot:
    graph = _load_graph()
    areas = [
        Area(
            id="living_room",
            name="Living Room",
            floor_id=area.floor_id,
            aliases=area.aliases,
        )
        if area.id == "area_living_room"
        else area
        for area in graph.areas
    ]
    return graph.model_copy(update={"areas": areas})


def test_register_default_satellites_maps_macbook_to_living_room() -> None:
    registry = SatelliteRegistry()
    register_default_satellites(registry)

    registration = registry.get(DEFAULT_SATELLITE_ID)
    assert registration is not None
    assert registration.satellite_id == DEFAULT_SATELLITE_ID
    assert registration.area_id == DEFAULT_SATELLITE_AREA_ID


def test_default_satellite_area_id_uses_env_override() -> None:
    assert default_satellite_area_id(environ={}) == DEFAULT_SATELLITE_AREA_ID
    assert default_satellite_area_id(
        environ={SATELLITE_AREA_ID_ENV_VAR: "living_room"},
    ) == "living_room"
    assert default_satellite_area_id(
        environ={SATELLITE_AREA_ID_ENV_VAR: "  "},
    ) == DEFAULT_SATELLITE_AREA_ID


def test_register_default_satellites_uses_env_override() -> None:
    registry = SatelliteRegistry()
    register_default_satellites(
        registry,
        environ={SATELLITE_AREA_ID_ENV_VAR: "living_room"},
    )

    registration = registry.get(DEFAULT_SATELLITE_ID)
    assert registration is not None
    assert registration.area_id == "living_room"


def test_resolve_area_id_returns_snapshot_id_for_exact_match() -> None:
    registry = SatelliteRegistry()
    registry.register(DEFAULT_SATELLITE_ID, "area_living_room")

    area_id, error = registry.resolve_area_id(
        DEFAULT_SATELLITE_ID,
        snapshot=_load_graph(),
    )

    assert error is None
    assert area_id == "area_living_room"


def test_resolve_area_id_matches_name_and_alias_case_insensitively() -> None:
    registry = SatelliteRegistry()
    snapshot = _load_graph()

    registry.register("by-name", "living room")
    area_id, error = registry.resolve_area_id("by-name", snapshot=snapshot)
    assert error is None
    assert area_id == "area_living_room"

    registry.register("by-alias", "FAMILY ROOM")
    area_id, error = registry.resolve_area_id("by-alias", snapshot=snapshot)
    assert error is None
    assert area_id == "area_living_room"


def test_resolve_area_id_maps_default_area_id_to_ha_living_room() -> None:
    registry = SatelliteRegistry()
    register_default_satellites(registry)

    area_id, error = registry.resolve_area_id(
        DEFAULT_SATELLITE_ID,
        snapshot=_ha_living_room_graph(),
    )

    assert error is None
    assert area_id == "living_room"


def test_resolve_area_id_unknown_when_no_match() -> None:
    registry = SatelliteRegistry()
    registry.register(DEFAULT_SATELLITE_ID, "area_missing")

    area_id, error = registry.resolve_area_id(
        DEFAULT_SATELLITE_ID,
        snapshot=_load_graph(),
    )

    assert area_id is None
    assert error == "unknown_area"


def test_resolve_area_id_unknown_when_multiple_matches() -> None:
    snapshot = _load_graph()
    ambiguous = snapshot.model_copy(
        update={
            "areas": [
                *snapshot.areas,
                Area(
                    id="area_lounge",
                    name="Lounge",
                    floor_id="floor_ground",
                    aliases=["family room"],
                ),
            ],
        },
    )
    registry = SatelliteRegistry()
    registry.register(DEFAULT_SATELLITE_ID, "family room")

    area_id, error = registry.resolve_area_id(
        DEFAULT_SATELLITE_ID,
        snapshot=ambiguous,
    )

    assert area_id is None
    assert error == "unknown_area"

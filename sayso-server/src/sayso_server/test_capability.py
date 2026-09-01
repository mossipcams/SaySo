"""Capability validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sayso_server.capability import CapabilityValidationError, validate_target_capabilities
from sayso_server.home_graph import (
    Area,
    Capability,
    CapabilityKind,
    Entity,
    Floor,
    HomeGraphSnapshot,
    State,
)
from sayso_server.models import ActionState

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _graph_with_non_dimmable_light() -> HomeGraphSnapshot:
    graph = _load_graph()
    non_dimmable = Entity(
        entity_id="light.porch",
        domain="light",
        name="Porch Light",
        area_id="area_living_room",
        capabilities=[Capability(kind=CapabilityKind.POWER)],
        state=State(value="off"),
    )
    return graph.model_copy(update={"entities": [*graph.entities, non_dimmable]})


def test_brightness_within_range_on_dimmable_entity_passes() -> None:
    graph = _load_graph()

    validate_target_capabilities(
        graph,
        frozenset({"light.living_room_ceiling"}),
        value=50,
    )


def test_brightness_on_non_dimmable_entity_raises() -> None:
    graph = _graph_with_non_dimmable_light()

    with pytest.raises(CapabilityValidationError, match="light.porch"):
        validate_target_capabilities(
            graph,
            frozenset({"light.porch"}),
            value=50,
        )


def test_brightness_out_of_range_raises() -> None:
    graph = _load_graph()

    with pytest.raises(CapabilityValidationError, match="light.living_room_ceiling"):
        validate_target_capabilities(
            graph,
            frozenset({"light.living_room_ceiling"}),
            value=101,
        )


def test_mixed_dimmable_and_non_dimmable_rejects_atomically() -> None:
    graph = _graph_with_non_dimmable_light()

    with pytest.raises(CapabilityValidationError, match="light.porch"):
        validate_target_capabilities(
            graph,
            frozenset({"light.living_room_ceiling", "light.porch"}),
            value=50,
        )


def test_temperature_within_range_on_climate_entity_passes() -> None:
    graph = _load_graph()

    validate_target_capabilities(
        graph,
        frozenset({"climate.downstairs"}),
        value=72,
    )


def test_temperature_out_of_range_raises() -> None:
    graph = _load_graph()

    with pytest.raises(CapabilityValidationError, match="climate.downstairs"):
        validate_target_capabilities(
            graph,
            frozenset({"climate.downstairs"}),
            value=40,
        )


def test_power_state_on_entity_without_power_raises() -> None:
    graph = _load_graph()

    with pytest.raises(CapabilityValidationError, match="binary_sensor.front_door"):
        validate_target_capabilities(
            graph,
            frozenset({"binary_sensor.front_door"}),
            state=ActionState.ON,
        )


def test_power_state_on_dimmable_light_passes() -> None:
    graph = _load_graph()

    validate_target_capabilities(
        graph,
        frozenset({"light.living_room_ceiling"}),
        state=ActionState.OFF,
    )

"""Candidate retrieval tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.candidates import CandidateRequest, retrieve_candidates
from sayso_server.conversation import LastTarget, SatelliteConversationState
from sayso_server.home_graph import Capability, CapabilityKind, Entity, HomeGraphSnapshot, State
from sayso_server.normalize import normalize_tokens

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _entity_ids(candidates: list[object]) -> list[str]:
    return [item.item.entity_id for item in candidates]


def test_normalize_tokens_lowercases_and_strips_punctuation() -> None:
    assert normalize_tokens("Turn OFF the lamp!") == ["turn", "off", "the", "lamp"]


def test_alias_match_ranks_gold_target_in_top_candidates() -> None:
    graph = _load_graph()

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="Turn off the lamp"),
        limit=3,
    )

    assert "light.floor_lamp" in _entity_ids(results)


def test_current_room_entity_ranks_in_top_candidates() -> None:
    graph = _load_graph()

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(tokens=["light"], domain="light"),
        limit=3,
    )

    top_ids = _entity_ids(results)
    assert "light.living_room_ceiling" in top_ids or "light.floor_lamp" in top_ids
    assert "script.good_night" not in top_ids


def test_structured_query_tokens_retrieve_candidates() -> None:
    graph = _load_graph()

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(tokens=["thermostat"], domain="climate"),
    )

    assert _entity_ids(results)[0] == "climate.downstairs"


def test_referent_boosts_last_target_in_top_candidates() -> None:
    graph = _load_graph()
    conversation = SatelliteConversationState(
        last_target=LastTarget(entity_ids=["light.living_room_ceiling"]),
    )

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="turn it off"),
        conversation=conversation,
        limit=3,
    )

    assert _entity_ids(results)[0] == "light.living_room_ceiling"


def test_inferred_light_domain_includes_switch_plug_lamps() -> None:
    graph = _load_graph()
    corner_lamp = Entity(
        entity_id="switch.corner_lamp",
        domain="switch",
        name="Corner Lamp",
        aliases=["corner lamp"],
        area_id="area_living_room",
        capabilities=[Capability(kind=CapabilityKind.POWER)],
        state=State(value="on"),
    )
    graph = graph.model_copy(update={"entities": [*graph.entities, corner_lamp]})

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="turn off corner lamp"),
        limit=5,
    )

    assert "switch.corner_lamp" in _entity_ids(results)


def test_inferred_light_domain_excludes_floor_sensor_at_limit_one() -> None:
    graph = _load_graph()
    floor_sensor = Entity(
        entity_id="sensor.alyssa_iphone_ble_floor",
        domain="sensor",
        name="Floor",
        aliases=["floor"],
        area_id="area_living_room",
        capabilities=[Capability(kind=CapabilityKind.QUERY)],
        state=State(value="42"),
    )
    corner_lamp = Entity(
        entity_id="switch.corner_lamp",
        domain="switch",
        name="Corner Lamp",
        aliases=["corner lamp"],
        area_id="area_living_room",
        capabilities=[Capability(kind=CapabilityKind.POWER)],
        state=State(value="on"),
    )
    graph = graph.model_copy(
        update={"entities": [*graph.entities, floor_sensor, corner_lamp]},
    )

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="turn off the floor lamp"),
        limit=1,
    )

    assert _entity_ids(results) == ["light.floor_lamp"]


def test_scoring_breakdown_includes_all_signals() -> None:
    graph = _load_graph()

    results = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="turn off the lamp"),
        limit=1,
    )

    breakdown = results[0].breakdown
    assert breakdown.domain >= 0
    assert breakdown.area >= 0
    assert breakdown.floor >= 0
    assert breakdown.alias >= 0
    assert breakdown.capability >= 0
    assert breakdown.state >= 0
    assert breakdown.referent >= 0
    assert breakdown.total == (
        breakdown.domain
        + breakdown.area
        + breakdown.floor
        + breakdown.alias
        + breakdown.capability
        + breakdown.state
        + breakdown.referent
    )

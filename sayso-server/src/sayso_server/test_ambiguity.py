"""Ambiguity handling tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.ambiguity import CandidateSelection, resolve_candidate_selection, resolve_candidates_for_request
from sayso_server.candidates import CandidateRequest, ScoredCandidate, retrieve_candidates
from sayso_server.control_plan import ClarificationPlan
from sayso_server.home_graph import (
    Area,
    Capability,
    CapabilityKind,
    Entity,
    Floor,
    HomeGraphSnapshot,
    State,
)
from sayso_server.scoring import DEFAULT_AMBIGUITY_MARGIN, ScoreBreakdown

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _two_lamp_graph() -> HomeGraphSnapshot:
    def _light(entity_id: str, name: str) -> Entity:
        return Entity(
            entity_id=entity_id,
            domain="light",
            name=name,
            aliases=["lamp"],
            area_id="area_living_room",
            capabilities=[
                Capability(kind=CapabilityKind.POWER),
                Capability(kind=CapabilityKind.BRIGHTNESS, min_value=1, max_value=100),
            ],
            state=State(value="off"),
        )

    return HomeGraphSnapshot(
        version=1,
        sequence=1,
        home_id="two-lamp-home",
        floors=[Floor(id="floor_ground", name="Ground Floor")],
        areas=[Area(id="area_living_room", name="Living Room", floor_id="floor_ground")],
        devices=[],
        entities=[
            _light("light.floor_lamp", "Floor Lamp"),
            _light("light.table_lamp", "Table Lamp"),
        ],
        scenes=[],
        scripts=[],
    )


def _scored(entity_id: str, name: str, score: float) -> ScoredCandidate:
    item = Entity(
        entity_id=entity_id,
        domain="light",
        name=name,
        area_id="area_living_room",
        capabilities=[Capability(kind=CapabilityKind.POWER)],
        state=State(value="off"),
    )
    breakdown = ScoreBreakdown(alias=score)
    return ScoredCandidate(item=item, score=score, breakdown=breakdown)


def test_equal_scores_return_clarification() -> None:
    candidates = [
        _scored("light.a", "Lamp A", 10.0),
        _scored("light.b", "Lamp B", 10.0),
    ]

    selection = resolve_candidate_selection(
        candidates,
        intent="turn on the lamp",
    )

    assert selection.outcome == "clarification"
    assert selection.candidate is None
    assert isinstance(selection.clarification, ClarificationPlan)
    assert selection.clarification.outcome == "clarification"
    assert len(selection.tied_candidates) == 2


def test_near_equal_scores_within_margin_return_clarification() -> None:
    candidates = [
        _scored("light.a", "Lamp A", 10.0),
        _scored("light.b", "Lamp B", 9.8),
    ]

    selection = resolve_candidate_selection(
        candidates,
        intent="turn on the lamp",
        margin=DEFAULT_AMBIGUITY_MARGIN,
    )

    assert selection.outcome == "clarification"
    assert selection.candidate is None
    assert isinstance(selection.clarification, ClarificationPlan)


def test_clear_winner_outside_margin_returns_candidate() -> None:
    graph = _load_graph()
    candidates = retrieve_candidates(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="turn off the lamp"),
        limit=3,
    )

    selection = resolve_candidate_selection(
        candidates,
        intent="turn off the lamp",
    )

    assert selection.outcome == "selected"
    assert selection.candidate is not None
    assert selection.candidate.item.entity_id == "light.floor_lamp"
    assert selection.clarification is None


def test_ambiguous_lamp_request_performs_no_action() -> None:
    graph = _two_lamp_graph()

    selection = resolve_candidates_for_request(
        graph,
        origin_area_id="area_living_room",
        request=CandidateRequest(utterance="turn on the lamp"),
        intent="turn on the lamp",
    )

    assert selection.outcome == "clarification"
    assert selection.candidate is None
    assert isinstance(selection.clarification, ClarificationPlan)
    assert selection.clarification.intent == "turn on the lamp"
    assert "lamp" in selection.clarification.reason.lower()

"""State query evaluator tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.control_plan import ControlPlan, QueryPlan
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.orchestrator import execute_control_plan
from sayso_server.queries import QueryOutcome, evaluate_query
from sayso_server.results import ExecutionCategory

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _query_plan(**overrides: object) -> QueryPlan:
    payload = {
        "outcome": "query",
        "intent": "is the door closed",
        "domain": "binary_sensor",
        "scope": {"kind": "current_area"},
        "targets": ["door"],
    }
    payload.update(overrides)
    return ControlPlan.model_validate(payload)  # type: ignore[return-value]


def test_single_door_query_returns_closed_state() -> None:
    graph = _load_graph()
    plan = _query_plan()

    result = evaluate_query(
        plan,
        graph,
        origin_area_id="area_living_room",
    )

    assert isinstance(result, QueryOutcome)
    assert result.answer == "closed"
    assert result.entity_ids == frozenset({"binary_sensor.front_door"})


def test_any_lights_on_aggregate_query_returns_yes() -> None:
    graph = _load_graph()
    plan = _query_plan(
        intent="check if any lights are on",
        domain="light",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        targets=[],
    )

    result = evaluate_query(
        plan,
        graph,
        origin_area_id="area_living_room",
    )

    assert isinstance(result, QueryOutcome)
    assert result.answer == "yes"
    assert result.entity_ids == frozenset({"light.floor_lamp", "light.living_room_ceiling"})


def test_query_orchestrator_path_never_sends_action_request() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _query_plan(
        intent="check if any lights are on",
        domain="light",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        targets=[],
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-query-1",
    )

    assert outcome.category == ExecutionCategory.COMPLETED
    assert isinstance(outcome.plan, QueryOutcome)
    assert outcome.plan.answer == "yes"
    assert ha_client.action_requests == []
    assert ha_client.calls == []

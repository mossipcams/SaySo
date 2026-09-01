"""Execution orchestrator pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.control_plan import (
    ActionPlan,
    ClarificationPlan,
    ControlPlan,
    NoActionPlan,
    UnsupportedPlan,
)
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import ActionState, Scope, ScopeKind
from sayso_server.orchestrator import execute_control_plan
from sayso_server.results import ActionResultStatus, ExecutionCategory

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _action_plan(**overrides: object) -> ActionPlan:
    payload = {
        "outcome": "action",
        "intent": "turn off the floor lamp",
        "domain": "light",
        "targets": ["floor lamp"],
        "state": "off",
    }
    payload.update(overrides)
    return ControlPlan.model_validate(payload)  # type: ignore[return-value]


def test_successful_pipeline_emits_completed_category() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    ha_client.queue_results(
        [
            ("req-1", ActionResultStatus.ACCEPTED, None),
            ("req-1", ActionResultStatus.COMPLETED, "state_changed"),
        ],
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.COMPLETED
    assert outcome.request_id == "req-1"
    assert len(outcome.results) == 2
    assert [result.status for result in outcome.results] == [
        ActionResultStatus.ACCEPTED,
        ActionResultStatus.COMPLETED,
    ]
    assert len(ha_client.action_requests) == 1
    assert ha_client.action_requests[0].entity_id == "light.floor_lamp"
    assert ha_client.action_requests[0].action == "off"


def test_validation_barrier_emits_no_action_without_request() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = UnsupportedPlan(
        intent="play music",
        reason="media playback is not supported",
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.NO_ACTION
    assert isinstance(outcome.plan, NoActionPlan)
    assert ha_client.action_requests == []


def test_clarification_plan_emits_no_action_without_request() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = ClarificationPlan(intent="turn on the lamp", reason="multiple lamps match")

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.NO_ACTION
    assert ha_client.action_requests == []


def test_rejected_result_emits_rejected_category() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    ha_client.queue_results([("req-1", ActionResultStatus.REJECTED, "domain_mismatch")])

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.REJECTED
    assert outcome.reason == "domain_mismatch"
    assert len(ha_client.action_requests) == 1


def test_failed_result_emits_failed_category() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    ha_client.queue_results(
        [
            ("req-1", ActionResultStatus.ACCEPTED, None),
            ("req-1", ActionResultStatus.FAILED, "execution_failed"),
        ],
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.FAILED
    assert outcome.reason == "execution_failed"


def test_partial_results_emit_incomplete_category() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    ha_client.queue_results([("req-1", ActionResultStatus.ACCEPTED, None)])

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.INCOMPLETE_RESULTS
    assert len(outcome.results) == 1


def test_misordered_results_emit_misordered_category() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    ha_client.queue_results(
        [
            ("req-1", ActionResultStatus.COMPLETED, "state_changed"),
            ("req-1", ActionResultStatus.ACCEPTED, None),
        ],
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.MISORDERED_RESULTS


def test_unrelated_request_id_is_not_treated_as_success() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    ha_client.queue_results(
        [
            ("other-req", ActionResultStatus.ACCEPTED, None),
            ("other-req", ActionResultStatus.COMPLETED, "state_changed"),
        ],
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.INCOMPLETE_RESULTS
    assert outcome.results == ()


def test_empty_target_set_is_no_action_before_request() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan(
        intent="turn off the lights",
        targets=[],
        scope=Scope(kind=ScopeKind.NAMED_AREA, name="Kitchen"),
        state=ActionState.OFF,
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.NO_ACTION
    assert isinstance(outcome.plan, NoActionPlan)
    assert ha_client.action_requests == []

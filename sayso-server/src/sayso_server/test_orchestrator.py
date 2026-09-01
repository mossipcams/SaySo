"""Execution orchestrator pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sayso_server.control_plan import (
    ActionPlan,
    ClarificationPlan,
    ControlPlan,
    NoActionPlan,
    UnsupportedPlan,
)
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import (
    Area,
    Capability,
    CapabilityKind,
    Entity,
    Floor,
    HomeGraphSnapshot,
    State,
)
from sayso_server.models import ActionState, Scope, ScopeKind
from sayso_server.orchestrator import execute_control_plan, execute_control_plan_async
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


def test_ambiguous_named_target_emits_no_action_without_request() -> None:
    graph = _two_lamp_graph()
    ha_client = FakeHaClient()
    plan = _action_plan(
        intent="turn on the lamp",
        targets=["lamp"],
        state=ActionState.ON,
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
    assert "clarification required" in (outcome.reason or "")
    assert ha_client.action_requests == []


def test_unique_named_target_still_executes() -> None:
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
    assert len(ha_client.action_requests) == 1
    assert ha_client.action_requests[0].entity_id == "light.floor_lamp"


@pytest.mark.asyncio
async def test_ambiguous_named_target_async_emits_no_action_without_request() -> None:
    graph = _two_lamp_graph()
    ha_client = FakeHaClient()

    async def _collect(_request_id: str) -> list:
        return []

    ha_client.collect_action_results = _collect  # type: ignore[attr-defined]
    plan = _action_plan(
        intent="turn on the lamp",
        targets=["lamp"],
        state=ActionState.ON,
    )

    outcome = await execute_control_plan_async(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
    )

    assert outcome.category == ExecutionCategory.NO_ACTION
    assert isinstance(outcome.plan, NoActionPlan)
    assert ha_client.action_requests == []


def test_switch_lamp_action_request_uses_resolved_entity_domain() -> None:
    graph = _load_graph()
    corner_lamp = Entity(
        entity_id="switch.corner_lamp",
        domain="switch",
        name="Corner Lamp",
        aliases=["corner plug"],
        area_id="area_living_room",
        capabilities=[
            Capability(kind=CapabilityKind.POWER),
        ],
        state=State(value="on"),
    )
    graph = graph.model_copy(update={"entities": [*graph.entities, corner_lamp]})
    ha_client = FakeHaClient()
    plan = _action_plan(
        intent="turn off corner plug",
        targets=["corner plug"],
    )
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
    assert len(ha_client.action_requests) == 1
    assert ha_client.action_requests[0].entity_id == "switch.corner_lamp"
    assert ha_client.action_requests[0].domain == "switch"


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

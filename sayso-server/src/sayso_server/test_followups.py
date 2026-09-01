"""Follow-up referent resolution tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.control_plan import ClarificationPlan, ControlPlan, NoActionPlan
from sayso_server.conversation import ConversationStore, LastTarget
from sayso_server.followups import is_follow_up_intent, resolve_follow_up
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import ActionState
from sayso_server.orchestrator import execute_control_plan
from sayso_server.results import ActionResultStatus, ExecutionCategory

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _action_plan(**overrides: object) -> object:
    payload = {
        "outcome": "action",
        "intent": "turn off the floor lamp",
        "domain": "light",
        "targets": ["floor lamp"],
        "state": "off",
    }
    payload.update(overrides)
    return ControlPlan.model_validate(payload)


def test_is_follow_up_intent_detects_pronoun_and_back_on_phrases() -> None:
    assert is_follow_up_intent("turn it back on") is True
    assert is_follow_up_intent("turn them off") is True
    assert is_follow_up_intent("turn off the floor lamp") is False


def test_active_follow_up_resolves_last_target() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=300.0, clock=clock)
    store.record_last_target("sat-1", LastTarget(entity_ids=["light.floor_lamp"]))
    plan = _action_plan(intent="turn it back on", targets=[], scope=None, state="on")

    resolution = resolve_follow_up(plan, store, satellite_id="sat-1")

    assert resolution.outcome == "resolved"
    assert resolution.entity_ids == frozenset({"light.floor_lamp"})


def test_expired_follow_up_returns_clarification() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    store.record_last_target("sat-1", LastTarget(entity_ids=["light.floor_lamp"]))
    clock.advance(61.0)
    plan = _action_plan(intent="turn it back on", targets=[], scope=None, state="on")

    resolution = resolve_follow_up(plan, store, satellite_id="sat-1")

    assert resolution.outcome == "clarification"
    assert isinstance(resolution.clarification, ClarificationPlan)
    assert "expired" in resolution.clarification.reason.lower()


def test_turn_it_back_on_executes_against_prior_target() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=300.0, clock=clock)

    ha_client.queue_results(
        [
            ("req-1", ActionResultStatus.ACCEPTED, None),
            ("req-1", ActionResultStatus.COMPLETED, "state_changed"),
        ],
    )
    first_plan = _action_plan()
    first_outcome = execute_control_plan(
        first_plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
        conversation_store=store,
        satellite_id="sat-1",
    )
    assert first_outcome.category == ExecutionCategory.COMPLETED

    ha_client.queue_results(
        [
            ("req-2", ActionResultStatus.ACCEPTED, None),
            ("req-2", ActionResultStatus.COMPLETED, "state_changed"),
        ],
    )
    follow_up_plan = _action_plan(
        intent="turn it back on",
        targets=[],
        scope=None,
        state=ActionState.ON,
    )
    follow_up_outcome = execute_control_plan(
        follow_up_plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-2",
        conversation_store=store,
        satellite_id="sat-1",
    )

    assert follow_up_outcome.category == ExecutionCategory.COMPLETED
    assert len(ha_client.action_requests) == 2
    assert ha_client.action_requests[-1].entity_id == "light.floor_lamp"
    assert ha_client.action_requests[-1].action == "on"


def test_expired_follow_up_clarifies_without_execution() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)

    ha_client.queue_results(
        [
            ("req-1", ActionResultStatus.ACCEPTED, None),
            ("req-1", ActionResultStatus.COMPLETED, "state_changed"),
        ],
    )
    first_plan = _action_plan()
    execute_control_plan(
        first_plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
        conversation_store=store,
        satellite_id="sat-1",
    )

    clock.advance(61.0)
    follow_up_plan = _action_plan(
        intent="turn it back on",
        targets=[],
        scope=None,
        state=ActionState.ON,
    )
    follow_up_outcome = execute_control_plan(
        follow_up_plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-2",
        conversation_store=store,
        satellite_id="sat-1",
    )

    assert follow_up_outcome.category == ExecutionCategory.NO_ACTION
    assert isinstance(follow_up_outcome.plan, NoActionPlan)
    assert "clarification" in follow_up_outcome.reason.lower()
    assert len(ha_client.action_requests) == 1

"""Safety validation barrier tests."""

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
from sayso_server.resolver import resolve_entity_ids
from sayso_server.safety import execute_if_safe

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


def _call(
    plan: object,
    graph: HomeGraphSnapshot,
    entity_ids: frozenset[str],
    ha_client: FakeHaClient,
) -> object:
    return execute_if_safe(
        plan,
        graph,
        entity_ids,
        ha_client,
        domain="light",
        service="turn_off",
        data={},
    )


def test_unsupported_plan_is_no_action_barrier() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = UnsupportedPlan(
        intent="play music on spotify",
        reason="media playback is not supported",
    )

    result = _call(plan, graph, frozenset({"light.floor_lamp"}), ha_client)

    assert isinstance(result, NoActionPlan)
    assert result.outcome == "no-action"
    assert result.intent == plan.intent
    assert "unsupported" in result.reason.lower()
    assert ha_client.calls == []


def test_unresolved_pronoun_is_no_action_barrier() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan(intent="turn it off", targets=[], scope=None, state="off")

    result = _call(plan, graph, frozenset(), ha_client)

    assert isinstance(result, NoActionPlan)
    assert result.outcome == "no-action"
    assert "pronoun" in result.reason.lower()
    assert ha_client.calls == []


def test_empty_target_set_is_no_action_barrier() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan(
        intent="turn off the lights",
        targets=[],
        scope=Scope(kind=ScopeKind.NAMED_AREA, name="Kitchen"),
        state="off",
    )
    resolved = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=plan.scope,
        domain=plan.domain,
    )

    result = _call(plan, graph, resolved, ha_client)

    assert isinstance(result, NoActionPlan)
    assert result.outcome == "no-action"
    assert "empty" in result.reason.lower() or "target" in result.reason.lower()
    assert resolved == frozenset()
    assert ha_client.calls == []


def test_hidden_entity_is_no_action_barrier() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    hidden_entity_ids = frozenset({"light.hidden_garage"})

    result = _call(plan, graph, hidden_entity_ids, ha_client)

    assert isinstance(result, NoActionPlan)
    assert result.outcome == "no-action"
    assert "hidden" in result.reason.lower()
    assert ha_client.calls == []


def test_clarification_plan_never_calls_ha_client() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = ClarificationPlan(intent="turn on the lamp", reason="multiple lamps match")

    result = _call(plan, graph, frozenset({"light.floor_lamp", "light.table_lamp"}), ha_client)

    assert isinstance(result, NoActionPlan)
    assert ha_client.calls == []


def test_no_action_plan_never_calls_ha_client() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = NoActionPlan(intent="unknown request", reason="could not interpret command")

    result = _call(plan, graph, frozenset({"light.floor_lamp"}), ha_client)

    assert result is plan
    assert ha_client.calls == []


def test_valid_action_calls_ha_client_once() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    plan = _action_plan()
    resolved = resolve_entity_ids(
        graph,
        origin_area_id="area_living_room",
        scope=Scope(kind=ScopeKind.CURRENT_AREA),
        domain=plan.domain,
        targets=plan.targets,
    )

    result = _call(plan, graph, resolved, ha_client)

    assert isinstance(result, ActionPlan)
    assert result == plan
    assert resolved == frozenset({"light.floor_lamp"})
    assert len(ha_client.calls) == 1
    assert ha_client.calls[0].entity_ids == resolved
    assert ha_client.calls[0].domain == "light"
    assert ha_client.calls[0].service == "turn_off"

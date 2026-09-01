"""Response policy matrix: earcon for completed control, short text otherwise."""

from __future__ import annotations

import pytest

from sayso_server.control_plan import ActionPlan, ClarificationPlan, ControlPlan, NoActionPlan
from sayso_server.queries import QueryOutcome
from sayso_server.response_policy import (
    EARCON_TOKEN,
    ResponseMode,
    resolve_response_policy,
)
from sayso_server.results import ExecutionCategory, ExecutionOutcome


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


def _outcome(
    *,
    category: ExecutionCategory,
    plan: object,
    reason: str | None = None,
) -> ExecutionOutcome:
    return ExecutionOutcome(category=category, plan=plan, reason=reason)


@pytest.mark.parametrize(
    ("category", "plan", "reason", "expected_mode", "expected_content"),
    [
        pytest.param(
            ExecutionCategory.COMPLETED,
            _action_plan(),
            "state_changed",
            ResponseMode.EARCON,
            EARCON_TOKEN,
            id="completed_action_earcon",
        ),
        pytest.param(
            ExecutionCategory.COMPLETED,
            QueryOutcome(answer="on", entity_ids=frozenset({"light.floor_lamp"})),
            "on",
            ResponseMode.TEXT,
            "on",
            id="completed_query_short_text",
        ),
        pytest.param(
            ExecutionCategory.NO_ACTION,
            ControlPlan.model_validate(
                {
                    "outcome": "clarification",
                    "intent": "turn off the lights",
                    "reason": "which lights?",
                }
            ),
            "clarification required: which lights?",
            ResponseMode.TEXT,
            "which lights?",
            id="clarification_short_text",
        ),
        pytest.param(
            ExecutionCategory.NO_ACTION,
            NoActionPlan(intent="turn on it", reason="missing referent"),
            "missing referent",
            ResponseMode.TEXT,
            "missing referent",
            id="no_action_short_text",
        ),
        pytest.param(
            ExecutionCategory.REJECTED,
            _action_plan(),
            "permission denied",
            ResponseMode.TEXT,
            "permission denied",
            id="rejected_error_short_text",
        ),
        pytest.param(
            ExecutionCategory.FAILED,
            _action_plan(),
            "entity unavailable",
            ResponseMode.TEXT,
            "entity unavailable",
            id="failed_error_short_text",
        ),
        pytest.param(
            ExecutionCategory.INCOMPLETE_RESULTS,
            _action_plan(),
            None,
            ResponseMode.TEXT,
            "action did not complete",
            id="incomplete_results_error",
        ),
        pytest.param(
            ExecutionCategory.MISORDERED_RESULTS,
            _action_plan(),
            None,
            ResponseMode.TEXT,
            "action did not complete",
            id="misordered_results_error",
        ),
    ],
)
def test_response_policy_matrix(
    category: ExecutionCategory,
    plan: object,
    reason: str | None,
    expected_mode: ResponseMode,
    expected_content: str,
) -> None:
    outcome = _outcome(category=category, plan=plan, reason=reason)
    policy = resolve_response_policy(outcome)

    assert policy.mode == expected_mode
    assert policy.content == expected_content


def test_response_policy_from_payload_dict() -> None:
    from sayso_server.response_policy import apply_response_policy

    payload = {
        "category": ExecutionCategory.COMPLETED.value,
        "reason": "state_changed",
        "plan": {
            "outcome": "action",
            "intent": "turn off the floor lamp",
            "domain": "light",
            "targets": ["floor lamp"],
            "state": "off",
        },
        "request_id": "req-1",
    }

    enriched = apply_response_policy(payload)

    assert enriched["response_mode"] == ResponseMode.EARCON.value
    assert enriched["response_content"] == EARCON_TOKEN
    assert enriched["category"] == "completed"


def test_clarification_barrier_strips_prefix() -> None:
    plan = ClarificationPlan(
        intent="turn off the lights",
        reason="which lights?",
    )
    barrier = NoActionPlan(
        intent=plan.intent,
        reason=f"clarification required: {plan.reason}",
    )
    outcome = _outcome(
        category=ExecutionCategory.NO_ACTION,
        plan=barrier,
        reason=barrier.reason,
    )

    policy = resolve_response_policy(outcome)

    assert policy.mode == ResponseMode.TEXT
    assert policy.content == "which lights?"

"""Map execution outcomes to satellite feedback: earcon vs short text."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sayso_server.control_plan import ActionPlan, NoActionPlan
from sayso_server.queries import QueryOutcome
from sayso_server.results import ExecutionCategory, ExecutionOutcome

EARCON_TOKEN = "\a"
_CLARIFICATION_PREFIX = "clarification required: "
_DEFAULT_INCOMPLETE_MESSAGE = "action did not complete"


class ResponseMode(StrEnum):
    EARCON = "earcon"
    TEXT = "text"


@dataclass(frozen=True)
class ResponsePolicy:
    mode: ResponseMode
    content: str


def resolve_response_policy(outcome: ExecutionOutcome) -> ResponsePolicy:
    """Choose earcon for a completed control action, short text for everything else."""

    if (
        outcome.category is ExecutionCategory.COMPLETED
        and isinstance(outcome.plan, ActionPlan)
    ):
        return ResponsePolicy(mode=ResponseMode.EARCON, content=EARCON_TOKEN)

    text = _short_text(outcome)
    return ResponsePolicy(mode=ResponseMode.TEXT, content=text)


def apply_response_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach response_mode and response_content to a text-controller payload."""

    outcome = _outcome_from_payload(payload)
    policy = resolve_response_policy(outcome)
    enriched = dict(payload)
    enriched["response_mode"] = policy.mode.value
    enriched["response_content"] = policy.content
    return enriched


def _outcome_from_payload(payload: dict[str, Any]) -> ExecutionOutcome:
    category = ExecutionCategory(payload["category"])
    plan = _plan_from_payload(payload.get("plan", {}))
    return ExecutionOutcome(
        category=category,
        plan=plan,
        reason=payload.get("reason"),
        request_id=payload.get("request_id"),
    )


def _plan_from_payload(plan_payload: dict[str, Any] | object) -> object:
    if not isinstance(plan_payload, dict):
        return plan_payload
    outcome = plan_payload.get("outcome")
    if outcome == "action":
        from sayso_server.control_plan import ControlPlan

        return ControlPlan.model_validate(plan_payload)
    if outcome in {"clarification", "unsupported", "no-action", "query"}:
        from sayso_server.control_plan import ControlPlan

        return ControlPlan.model_validate(plan_payload)
    return plan_payload


def _short_text(outcome: ExecutionOutcome) -> str:
    if outcome.category in {
        ExecutionCategory.INCOMPLETE_RESULTS,
        ExecutionCategory.MISORDERED_RESULTS,
    }:
        return outcome.reason or _DEFAULT_INCOMPLETE_MESSAGE

    if outcome.reason:
        return _normalize_reason(outcome.reason, plan=outcome.plan)

    if isinstance(outcome.plan, QueryOutcome):
        return outcome.plan.answer

    if isinstance(outcome.plan, NoActionPlan):
        return _normalize_reason(outcome.plan.reason, plan=outcome.plan)

    return "unable to complete request"


def _normalize_reason(reason: str, *, plan: object) -> str:
    if reason.startswith(_CLARIFICATION_PREFIX):
        return reason.removeprefix(_CLARIFICATION_PREFIX)
    if isinstance(plan, NoActionPlan) and plan.reason.startswith(_CLARIFICATION_PREFIX):
        return plan.reason.removeprefix(_CLARIFICATION_PREFIX)
    return reason


__all__ = [
    "EARCON_TOKEN",
    "ResponseMode",
    "ResponsePolicy",
    "apply_response_policy",
    "resolve_response_policy",
]

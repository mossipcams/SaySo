"""Plan → resolve → validate → request → verify execution pipeline."""

from __future__ import annotations

from pydantic import BaseModel

from sayso_server.control_plan import ActionPlan, NoActionPlan, QueryPlan
from sayso_server.ha_client import ActionRequestClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import ActionState, Scope, ScopeKind
from sayso_server.queries import evaluate_query
from sayso_server.resolver import resolve_entity_ids
from sayso_server.results import (
    ActionResult,
    ActionResultStatus,
    ExecutionCategory,
    ExecutionOutcome,
)
from sayso_server.safety import evaluate_safety_barrier

_SUCCESS_SEQUENCE = (
    ActionResultStatus.ACCEPTED,
    ActionResultStatus.COMPLETED,
)


def execute_control_plan(
    plan: BaseModel,
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    ha_client: ActionRequestClient,
    request_id: str,
) -> ExecutionOutcome:
    """Run the control-plan execution pipeline and emit an exact outcome category."""

    if isinstance(plan, QueryPlan):
        query_result = evaluate_query(
            plan,
            snapshot,
            origin_area_id=origin_area_id,
        )
        if isinstance(query_result, NoActionPlan):
            return ExecutionOutcome(
                category=ExecutionCategory.NO_ACTION,
                plan=query_result,
                reason=query_result.reason,
            )
        return ExecutionOutcome(
            category=ExecutionCategory.COMPLETED,
            plan=query_result,
            reason=query_result.answer,
        )

    if not isinstance(plan, ActionPlan):
        barrier = evaluate_safety_barrier(plan, snapshot, frozenset())
        if barrier is None:
            msg = "non-action plans must produce a safety barrier"
            raise RuntimeError(msg)
        return ExecutionOutcome(
            category=ExecutionCategory.NO_ACTION,
            plan=barrier,
            reason=barrier.reason,
        )

    scope = plan.scope
    if scope is None and (plan.targets or plan.include or plan.exclude):
        scope = Scope(kind=ScopeKind.CURRENT_AREA)

    resolved_entity_ids = resolve_entity_ids(
        snapshot,
        origin_area_id=origin_area_id,
        scope=scope,
        domain=plan.domain,
        targets=plan.targets,
        include=plan.include,
        exclude=plan.exclude,
    )

    barrier = evaluate_safety_barrier(plan, snapshot, resolved_entity_ids)
    if barrier is not None:
        return ExecutionOutcome(
            category=ExecutionCategory.NO_ACTION,
            plan=barrier,
            reason=barrier.reason,
        )

    entity_id = sorted(resolved_entity_ids)[0]
    action, payload = _semantic_action(plan)

    ha_client.send_action_request(
        request_id=request_id,
        entity_id=entity_id,
        domain=plan.domain,
        action=action,
        data=payload,
    )
    results = tuple(ha_client.take_action_results(request_id))
    category, reason = classify_action_results(request_id, results)

    return ExecutionOutcome(
        category=category,
        plan=plan,
        request_id=request_id,
        results=results,
        reason=reason,
    )


def classify_action_results(
    request_id: str,
    results: tuple[ActionResult, ...] | list[ActionResult],
) -> tuple[ExecutionCategory, str | None]:
    """Map correlated action results to an exact success or failure category."""

    matching = [result for result in results if result.request_id == request_id]
    if not matching:
        return ExecutionCategory.INCOMPLETE_RESULTS, None

    statuses = [result.status for result in matching]

    if statuses == [ActionResultStatus.REJECTED]:
        return ExecutionCategory.REJECTED, matching[0].reason

    if statuses == [ActionResultStatus.FAILED]:
        return ExecutionCategory.FAILED, matching[0].reason

    if ActionResultStatus.FAILED in statuses:
        failed = next(
            result for result in matching if result.status is ActionResultStatus.FAILED
        )
        return ExecutionCategory.FAILED, failed.reason

    if statuses == list(_SUCCESS_SEQUENCE):
        return ExecutionCategory.COMPLETED, matching[-1].reason

    if ActionResultStatus.ACCEPTED in statuses and ActionResultStatus.COMPLETED not in statuses:
        return ExecutionCategory.INCOMPLETE_RESULTS, None

    if (
        ActionResultStatus.COMPLETED in statuses
        and ActionResultStatus.ACCEPTED in statuses
        and statuses != list(_SUCCESS_SEQUENCE)
    ):
        return ExecutionCategory.MISORDERED_RESULTS, None

    return ExecutionCategory.INCOMPLETE_RESULTS, None


def _semantic_action(plan: ActionPlan) -> tuple[str, dict[str, object]]:
    if plan.value is not None:
        return "set_brightness", {"brightness": plan.value}

    if plan.state is ActionState.ACTIVATE:
        if plan.domain == "scene":
            return "scene", {}
        if plan.domain == "script":
            return "script", {}

    if plan.state is not None:
        return plan.state.value, {}

    msg = "action plan requires state, value, or supported activate target"
    raise ValueError(msg)

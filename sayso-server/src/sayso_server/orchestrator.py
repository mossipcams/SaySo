"""Plan → resolve → validate → request → verify execution pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from sayso_server.telemetry import InteractionTelemetry

from sayso_server.control_plan import ActionPlan, NoActionPlan, QueryPlan
from sayso_server.conversation import ConversationStore, LastIntent, LastTarget
from sayso_server.followups import resolve_follow_up
from sayso_server.ha_client import ActionRequestClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import ActionState, Scope, ScopeKind
from sayso_server.queries import evaluate_query
from sayso_server.resolver import resolve_action_entities
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
    conversation_store: ConversationStore | None = None,
    satellite_id: str | None = None,
    telemetry: InteractionTelemetry | None = None,
) -> ExecutionOutcome:
    """Run the control-plan execution pipeline and emit an exact outcome category."""

    if isinstance(plan, QueryPlan):
        with _telemetry_stage(telemetry, "resolve"):
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
        with _telemetry_stage(telemetry, "validate"):
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

    follow_up_entity_ids: list[str] | None = None
    with _telemetry_stage(telemetry, "resolve"):
        if conversation_store is not None and satellite_id is not None:
            follow_up = resolve_follow_up(plan, conversation_store, satellite_id=satellite_id)
            if follow_up.outcome == "clarification" and follow_up.clarification is not None:
                with _telemetry_stage(telemetry, "validate"):
                    barrier = evaluate_safety_barrier(
                        follow_up.clarification,
                        snapshot,
                        frozenset(),
                    )
                if barrier is None:
                    msg = "follow-up clarification must produce a safety barrier"
                    raise RuntimeError(msg)
                return ExecutionOutcome(
                    category=ExecutionCategory.NO_ACTION,
                    plan=barrier,
                    reason=barrier.reason,
                )
            if follow_up.outcome == "resolved":
                follow_up_entity_ids = sorted(follow_up.entity_ids)

        conversation = (
            conversation_store.get_state(satellite_id)
            if conversation_store is not None and satellite_id is not None
            else None
        )
        resolution = resolve_action_entities(
            snapshot,
            origin_area_id=origin_area_id,
            intent=plan.intent,
            scope=scope,
            entity_ids=follow_up_entity_ids,
            domain=plan.domain,
            targets=plan.targets,
            include=plan.include,
            exclude=plan.exclude,
            conversation=conversation,
        )
        if resolution.clarification is not None:
            with _telemetry_stage(telemetry, "validate"):
                barrier = evaluate_safety_barrier(
                    resolution.clarification,
                    snapshot,
                    frozenset(),
                )
            if barrier is None:
                msg = "ambiguity clarification must produce a safety barrier"
                raise RuntimeError(msg)
            return ExecutionOutcome(
                category=ExecutionCategory.NO_ACTION,
                plan=barrier,
                reason=barrier.reason,
            )
        resolved_entity_ids = resolution.entity_ids

    with _telemetry_stage(telemetry, "validate"):
        barrier = evaluate_safety_barrier(plan, snapshot, resolved_entity_ids)
    if barrier is not None:
        return ExecutionOutcome(
            category=ExecutionCategory.NO_ACTION,
            plan=barrier,
            reason=barrier.reason,
        )

    entity_id = sorted(resolved_entity_ids)[0]
    action, payload = _semantic_action(plan)
    entity_domain = _resolved_entity_domain(snapshot, entity_id)

    with _telemetry_stage(telemetry, "request"):
        ha_client.send_action_request(
            request_id=request_id,
            entity_id=entity_id,
            domain=entity_domain,
            action=action,
            data=payload,
        )
    with _telemetry_stage(telemetry, "verify"):
        results = tuple(ha_client.take_action_results(request_id))
        category, reason = classify_action_results(request_id, results)

    if (
        category is ExecutionCategory.COMPLETED
        and conversation_store is not None
        and satellite_id is not None
    ):
        _record_conversation_referents(
            conversation_store,
            satellite_id=satellite_id,
            plan=plan,
            resolved_entity_ids=resolved_entity_ids,
        )

    return ExecutionOutcome(
        category=category,
        plan=plan,
        request_id=request_id,
        results=results,
        reason=reason,
    )


async def execute_control_plan_async(
    plan: BaseModel,
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    ha_client: ActionRequestClient,
    request_id: str,
    conversation_store: ConversationStore | None = None,
    satellite_id: str | None = None,
    telemetry: InteractionTelemetry | None = None,
) -> ExecutionOutcome:
    """Async execution path that awaits live WebSocket action_result collection."""

    collect = getattr(ha_client, "collect_action_results", None)
    if collect is None:
        return execute_control_plan(
            plan,
            snapshot,
            origin_area_id=origin_area_id,
            ha_client=ha_client,
            request_id=request_id,
            conversation_store=conversation_store,
            satellite_id=satellite_id,
            telemetry=telemetry,
        )

    if isinstance(plan, QueryPlan):
        with _telemetry_stage(telemetry, "resolve"):
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
        with _telemetry_stage(telemetry, "validate"):
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

    follow_up_entity_ids: list[str] | None = None
    with _telemetry_stage(telemetry, "resolve"):
        if conversation_store is not None and satellite_id is not None:
            follow_up = resolve_follow_up(plan, conversation_store, satellite_id=satellite_id)
            if follow_up.outcome == "clarification" and follow_up.clarification is not None:
                with _telemetry_stage(telemetry, "validate"):
                    barrier = evaluate_safety_barrier(
                        follow_up.clarification,
                        snapshot,
                        frozenset(),
                    )
                if barrier is None:
                    msg = "follow-up clarification must produce a safety barrier"
                    raise RuntimeError(msg)
                return ExecutionOutcome(
                    category=ExecutionCategory.NO_ACTION,
                    plan=barrier,
                    reason=barrier.reason,
                )
            if follow_up.outcome == "resolved":
                follow_up_entity_ids = sorted(follow_up.entity_ids)

        conversation = (
            conversation_store.get_state(satellite_id)
            if conversation_store is not None and satellite_id is not None
            else None
        )
        resolution = resolve_action_entities(
            snapshot,
            origin_area_id=origin_area_id,
            intent=plan.intent,
            scope=scope,
            entity_ids=follow_up_entity_ids,
            domain=plan.domain,
            targets=plan.targets,
            include=plan.include,
            exclude=plan.exclude,
            conversation=conversation,
        )
        if resolution.clarification is not None:
            with _telemetry_stage(telemetry, "validate"):
                barrier = evaluate_safety_barrier(
                    resolution.clarification,
                    snapshot,
                    frozenset(),
                )
            if barrier is None:
                msg = "ambiguity clarification must produce a safety barrier"
                raise RuntimeError(msg)
            return ExecutionOutcome(
                category=ExecutionCategory.NO_ACTION,
                plan=barrier,
                reason=barrier.reason,
            )
        resolved_entity_ids = resolution.entity_ids

    with _telemetry_stage(telemetry, "validate"):
        barrier = evaluate_safety_barrier(plan, snapshot, resolved_entity_ids)
    if barrier is not None:
        return ExecutionOutcome(
            category=ExecutionCategory.NO_ACTION,
            plan=barrier,
            reason=barrier.reason,
        )

    entity_id = sorted(resolved_entity_ids)[0]
    action, payload = _semantic_action(plan)
    entity_domain = _resolved_entity_domain(snapshot, entity_id)

    with _telemetry_stage(telemetry, "request"):
        ha_client.send_action_request(
            request_id=request_id,
            entity_id=entity_id,
            domain=entity_domain,
            action=action,
            data=payload,
        )
    with _telemetry_stage(telemetry, "verify"):
        results = tuple(await collect(request_id))
        category, reason = classify_action_results(request_id, results)

    if (
        category is ExecutionCategory.COMPLETED
        and conversation_store is not None
        and satellite_id is not None
    ):
        _record_conversation_referents(
            conversation_store,
            satellite_id=satellite_id,
            plan=plan,
            resolved_entity_ids=resolved_entity_ids,
        )

    return ExecutionOutcome(
        category=category,
        plan=plan,
        request_id=request_id,
        results=results,
        reason=reason,
    )


def _resolved_entity_domain(snapshot: HomeGraphSnapshot, entity_id: str) -> str:
    for entity in snapshot.entities:
        if entity.entity_id == entity_id:
            return entity.domain
    for scene in snapshot.scenes:
        if scene.entity_id == entity_id:
            return "scene"
    for script in snapshot.scripts:
        if script.entity_id == entity_id:
            return "script"
    return entity_id.split(".", 1)[0]


def _telemetry_stage(
    telemetry: InteractionTelemetry | None,
    stage: str,
):
    if telemetry is None:
        return nullcontext()
    return telemetry.time_stage(stage)


def _record_conversation_referents(
    store: ConversationStore,
    *,
    satellite_id: str,
    plan: ActionPlan,
    resolved_entity_ids: frozenset[str],
) -> None:
    store.record_last_target(
        satellite_id,
        LastTarget(entity_ids=sorted(resolved_entity_ids)),
    )
    store.record_last_intent(
        satellite_id,
        LastIntent(intent=plan.intent, outcome=plan.outcome),
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

"""Safety validation barriers before Home Assistant execution."""

from __future__ import annotations

from pydantic import BaseModel

from sayso_server.capability import CapabilityValidationError, validate_target_capabilities
from sayso_server.control_plan import (
    ActionPlan,
    ClarificationPlan,
    NoActionPlan,
    QueryPlan,
    UnsupportedPlan,
)
from sayso_server.ha_client import HaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.normalize import normalize_tokens

_PRONOUN_TOKENS = frozenset({"it", "them", "that", "those"})


def _intent_uses_pronoun(intent: str) -> bool:
    return bool(set(normalize_tokens(intent)).intersection(_PRONOUN_TOKENS))


def _known_entity_ids(snapshot: HomeGraphSnapshot) -> frozenset[str]:
    return frozenset(
        item.entity_id
        for item in (*snapshot.entities, *snapshot.scenes, *snapshot.scripts)
    )


def evaluate_safety_barrier(
    plan: BaseModel,
    snapshot: HomeGraphSnapshot,
    resolved_entity_ids: frozenset[str],
) -> NoActionPlan | None:
    """Return a no-action barrier when execution must not proceed."""
    if isinstance(plan, UnsupportedPlan):
        return NoActionPlan(
            intent=plan.intent,
            reason=f"unsupported: {plan.reason}",
        )

    if isinstance(plan, NoActionPlan):
        return plan

    if isinstance(plan, ClarificationPlan):
        return NoActionPlan(
            intent=plan.intent,
            reason=f"clarification required: {plan.reason}",
        )

    if isinstance(plan, QueryPlan):
        return NoActionPlan(
            intent=plan.intent,
            reason="query requests do not execute service calls",
        )

    if not isinstance(plan, ActionPlan):
        return NoActionPlan(
            intent=getattr(plan, "intent", "unknown"),
            reason="unsupported control plan outcome",
        )

    if _intent_uses_pronoun(plan.intent) and not resolved_entity_ids:
        return NoActionPlan(
            intent=plan.intent,
            reason="unresolved pronoun reference",
        )

    if not resolved_entity_ids:
        return NoActionPlan(
            intent=plan.intent,
            reason="empty target set",
        )

    hidden = resolved_entity_ids - _known_entity_ids(snapshot)
    if hidden:
        hidden_id = sorted(hidden)[0]
        return NoActionPlan(
            intent=plan.intent,
            reason=f"hidden entity: {hidden_id}",
        )

    try:
        validate_target_capabilities(
            snapshot,
            resolved_entity_ids,
            value=plan.value,
            state=plan.state,
            mode=plan.mode,
        )
    except CapabilityValidationError as exc:
        return NoActionPlan(intent=plan.intent, reason=str(exc))

    return None


def execute_if_safe(
    plan: BaseModel,
    snapshot: HomeGraphSnapshot,
    resolved_entity_ids: frozenset[str],
    ha_client: HaClient,
    *,
    domain: str,
    service: str,
    data: dict[str, object] | None = None,
) -> BaseModel:
    """Apply safety barriers and call Home Assistant only when validation passes."""
    barrier = evaluate_safety_barrier(plan, snapshot, resolved_entity_ids)
    if barrier is not None:
        return barrier

    ha_client.call_service(
        domain=domain,
        service=service,
        data=data or {},
        entity_ids=resolved_entity_ids,
    )
    return plan

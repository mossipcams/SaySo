"""Read-only state query evaluation from Home Graph snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from sayso_server.control_plan import NoActionPlan, QueryPlan
from sayso_server.home_graph import Entity, HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.normalize import normalize_tokens
from sayso_server.resolver import resolve_entity_ids

_AGGREGATE_ANY_TOKENS = frozenset({"any"})
_AGGREGATE_ALL_TOKENS = frozenset({"all"})


@dataclass(frozen=True)
class QueryOutcome:
    answer: str
    entity_ids: frozenset[str]


def evaluate_query(
    plan: QueryPlan,
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
) -> QueryOutcome | NoActionPlan:
    """Evaluate a read-only state query against the Home Graph snapshot."""
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

    if not resolved_entity_ids:
        return NoActionPlan(
            intent=plan.intent,
            reason="empty target set",
        )

    intent_tokens = set(normalize_tokens(plan.intent))
    if intent_tokens.intersection(_AGGREGATE_ANY_TOKENS):
        answer = _aggregate_answer(resolved_entity_ids, snapshot, require_all=False)
    elif intent_tokens.intersection(_AGGREGATE_ALL_TOKENS):
        answer = _aggregate_answer(resolved_entity_ids, snapshot, require_all=True)
    else:
        if len(resolved_entity_ids) != 1:
            return NoActionPlan(
                intent=plan.intent,
                reason="single-state query requires exactly one target",
            )
        entity_id = next(iter(resolved_entity_ids))
        entity = _entity_by_id(snapshot, entity_id)
        if entity is None:
            return NoActionPlan(
                intent=plan.intent,
                reason=f"hidden entity: {entity_id}",
            )
        answer = _normalize_entity_state(entity)

    return QueryOutcome(answer=answer, entity_ids=resolved_entity_ids)


def _aggregate_answer(
    entity_ids: frozenset[str],
    snapshot: HomeGraphSnapshot,
    *,
    require_all: bool,
) -> str:
    active_count = 0
    for entity_id in entity_ids:
        entity = _entity_by_id(snapshot, entity_id)
        if entity is not None and entity.state.value == "on":
            active_count += 1

    if require_all:
        return "yes" if active_count == len(entity_ids) else "no"
    return "yes" if active_count else "no"


def _entity_by_id(snapshot: HomeGraphSnapshot, entity_id: str) -> Entity | None:
    for entity in snapshot.entities:
        if entity.entity_id == entity_id:
            return entity
    return None


def _normalize_entity_state(entity: Entity) -> str:
    raw = entity.state.value
    device_class = entity.state.attributes.get("device_class")
    if entity.domain == "binary_sensor" and device_class == "door":
        if raw == "on":
            return "open"
        if raw == "off":
            return "closed"
    return raw

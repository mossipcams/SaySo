"""Resolve scopes and explicit targets to entity-id sets."""

from __future__ import annotations

from dataclasses import dataclass

from sayso_server.ambiguity import resolve_candidate_selection
from sayso_server.candidates import CandidateRequest, retrieve_candidates
from sayso_server.control_plan import ClarificationPlan
from sayso_server.conversation import SatelliteConversationState
from sayso_server.exclusions import apply_inclusions_exclusions, filter_entity_ids_by_domain
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.scope import expand_scope


@dataclass(frozen=True)
class EntityResolution:
    entity_ids: frozenset[str]
    clarification: ClarificationPlan | None = None


def resolve_entity_ids(
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    scope: Scope | None = None,
    entity_ids: list[str] | None = None,
    domain: str | None = None,
    targets: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> frozenset[str]:
    """Return the exact entity-id set for scope expansion with include/exclude names."""
    if entity_ids:
        base = frozenset(sorted(entity_ids))
    elif scope is None:
        base = frozenset()
    else:
        base = expand_scope(snapshot, origin_area_id, scope)

    if domain is not None:
        base = filter_entity_ids_by_domain(snapshot, base, domain)

    result = apply_inclusions_exclusions(
        snapshot,
        base,
        targets=targets,
        include=include,
        exclude=exclude,
    )

    names = [*(targets or []), *(include or [])]
    if not result and names and not entity_ids and scope is not None:
        whole_base = expand_scope(snapshot, origin_area_id, Scope(kind=ScopeKind.ALL))
        if domain is not None:
            whole_base = filter_entity_ids_by_domain(snapshot, whole_base, domain)
        result = apply_inclusions_exclusions(
            snapshot,
            whole_base,
            targets=targets,
            include=include,
            exclude=exclude,
        )

    return result


def resolve_action_entities(
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    intent: str = "",
    scope: Scope | None = None,
    entity_ids: list[str] | None = None,
    domain: str | None = None,
    targets: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    conversation: SatelliteConversationState | None = None,
) -> EntityResolution:
    """Resolve action targets, applying score-margin ambiguity for named matches."""
    resolved = resolve_entity_ids(
        snapshot,
        origin_area_id=origin_area_id,
        scope=scope,
        entity_ids=entity_ids,
        domain=domain,
        targets=targets,
        include=include,
        exclude=exclude,
    )

    if entity_ids or len(resolved) <= 1:
        return EntityResolution(resolved)

    if targets and len(targets) > 1:
        return EntityResolution(resolved)
    if include and len(include) > 1:
        return EntityResolution(resolved)

    single_name = (targets or include or [None])[0]
    if single_name is None:
        return EntityResolution(resolved)

    candidates = retrieve_candidates(
        snapshot,
        origin_area_id=origin_area_id,
        request=CandidateRequest(
            utterance=intent or single_name,
            domain=domain,
        ),
        conversation=conversation,
    )
    filtered = [candidate for candidate in candidates if candidate.item.entity_id in resolved]
    selection = resolve_candidate_selection(filtered, intent=intent)

    if selection.outcome == "clarification" and selection.clarification is not None:
        return EntityResolution(frozenset(), selection.clarification)

    if selection.candidate is not None:
        return EntityResolution(frozenset([selection.candidate.item.entity_id]))

    return EntityResolution(resolved)

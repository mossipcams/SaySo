"""Follow-up intent detection and last-target referent resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sayso_server.control_plan import ActionPlan, ClarificationPlan
from sayso_server.conversation import ConversationStore, LastTarget
from sayso_server.normalize import normalize_tokens

_PRONOUN_TOKENS = frozenset({"it", "them", "that", "those"})
_BACK_TOKEN = "back"
_FOLLOWUP_STATE_TOKENS = frozenset({"on", "off", "open", "closed", "lock", "unlock"})


@dataclass(frozen=True)
class FollowUpResolution:
    outcome: Literal["not_follow_up", "resolved", "clarification"]
    entity_ids: frozenset[str] = frozenset()
    clarification: ClarificationPlan | None = None


def is_follow_up_intent(intent: str) -> bool:
    """True when the utterance refers to a prior target via pronoun or back-* phrasing."""
    tokens = set(normalize_tokens(intent))
    if tokens.intersection(_PRONOUN_TOKENS):
        return True
    return _BACK_TOKEN in tokens and bool(tokens.intersection(_FOLLOWUP_STATE_TOKENS))


def resolve_follow_up(
    plan: ActionPlan,
    store: ConversationStore,
    *,
    satellite_id: str,
) -> FollowUpResolution:
    """Resolve a follow-up action plan to prior entity ids or return clarification."""
    if not is_follow_up_intent(plan.intent):
        return FollowUpResolution(outcome="not_follow_up")

    last_target = store.active_last_target(satellite_id)
    if last_target is None:
        return FollowUpResolution(
            outcome="clarification",
            clarification=ClarificationPlan(
                intent=plan.intent,
                reason="prior reference expired or unavailable",
            ),
        )

    return FollowUpResolution(
        outcome="resolved",
        entity_ids=_entity_ids_from_target(last_target),
    )


def _entity_ids_from_target(last_target: LastTarget) -> frozenset[str]:
    return frozenset(last_target.entity_ids)

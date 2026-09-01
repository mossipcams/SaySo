"""Score-margin ambiguity detection for candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sayso_server.candidates import CandidateRequest, ScoredCandidate, retrieve_candidates
from sayso_server.control_plan import ClarificationPlan
from sayso_server.conversation import SatelliteConversationState
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.scoring import DEFAULT_AMBIGUITY_MARGIN


@dataclass(frozen=True)
class CandidateSelection:
    outcome: Literal["selected", "clarification"]
    candidate: ScoredCandidate | None = None
    clarification: ClarificationPlan | None = None
    tied_candidates: tuple[ScoredCandidate, ...] = ()


def candidates_within_score_margin(
    candidates: list[ScoredCandidate],
    margin: float,
) -> list[ScoredCandidate]:
    """Return top-scoring candidates within the score margin of the leader."""
    if not candidates:
        return []

    top_score = candidates[0].score
    if top_score <= 0:
        return []

    return [candidate for candidate in candidates if top_score - candidate.score <= margin]


def is_ambiguous(
    candidates: list[ScoredCandidate],
    *,
    margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> bool:
    """True when two or more distinct candidates tie within the score margin."""
    tied = candidates_within_score_margin(candidates, margin)
    if len(tied) < 2:
        return False
    entity_ids = {candidate.item.entity_id for candidate in tied}
    return len(entity_ids) >= 2


def resolve_candidate_selection(
    candidates: list[ScoredCandidate],
    *,
    intent: str,
    margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> CandidateSelection:
    """Select one candidate or return clarification when scores are too close."""
    tied = candidates_within_score_margin(candidates, margin)
    entity_ids = {candidate.item.entity_id for candidate in tied}

    if len(entity_ids) >= 2:
        names = ", ".join(candidate.item.name for candidate in tied[:3])
        reason = f"Multiple devices match: {names}"
        return CandidateSelection(
            outcome="clarification",
            clarification=ClarificationPlan(intent=intent, reason=reason),
            tied_candidates=tuple(tied),
        )

    if tied:
        return CandidateSelection(outcome="selected", candidate=tied[0])

    return CandidateSelection(
        outcome="clarification",
        clarification=ClarificationPlan(
            intent=intent,
            reason="No matching device found",
        ),
    )


def resolve_candidates_for_request(
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    request: CandidateRequest | str,
    intent: str,
    conversation: SatelliteConversationState | None = None,
    limit: int = 8,
    margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> CandidateSelection:
    """Retrieve candidates and apply the score-margin ambiguity rule."""
    candidates = retrieve_candidates(
        snapshot,
        origin_area_id=origin_area_id,
        request=request,
        conversation=conversation,
        limit=limit,
    )
    return resolve_candidate_selection(candidates, intent=intent, margin=margin)

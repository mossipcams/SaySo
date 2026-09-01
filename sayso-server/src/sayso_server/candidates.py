"""Retrieve scored Home Graph candidates for model prompting."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

from sayso_server.conversation import SatelliteConversationState
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.normalize import normalize_tokens
from sayso_server.scoring import CandidateItem, ScoreBreakdown, infer_domain, lookup_origin, score_candidate


class CandidateRequest(BaseModel):
    utterance: str | None = None
    tokens: list[str] = Field(default_factory=list)
    domain: str | None = None

    @model_validator(mode="after")
    def _require_query_input(self) -> CandidateRequest:
        if self.utterance is None and not self.tokens:
            msg = "CandidateRequest requires utterance or tokens"
            raise ValueError(msg)
        return self

    def query_tokens(self) -> list[str]:
        if self.tokens:
            return list(self.tokens)
        if self.utterance is None:
            return []
        return normalize_tokens(self.utterance)


@dataclass(frozen=True)
class ScoredCandidate:
    item: CandidateItem
    score: float
    breakdown: ScoreBreakdown


def retrieve_candidates(
    snapshot: HomeGraphSnapshot,
    *,
    origin_area_id: str,
    request: CandidateRequest | str,
    conversation: SatelliteConversationState | None = None,
    limit: int = 8,
) -> list[ScoredCandidate]:
    """Return top-scoring entities, scenes, and scripts for a query."""
    parsed_request = (
        CandidateRequest(utterance=request)
        if isinstance(request, str)
        else request
    )
    query_tokens = parsed_request.query_tokens()
    inferred_domain = infer_domain(query_tokens, parsed_request.domain)
    origin_area, origin_floor = lookup_origin(snapshot, origin_area_id)
    area_by_id = {area.id: area for area in snapshot.areas}

    candidates: list[CandidateItem] = [
        *snapshot.entities,
        *snapshot.scenes,
        *snapshot.scripts,
    ]
    scored = [
        ScoredCandidate(
            item=item,
            score=breakdown.total,
            breakdown=breakdown,
        )
        for item in candidates
        if (
            breakdown := score_candidate(
                item,
                query_tokens=query_tokens,
                inferred_domain=inferred_domain,
                origin_area=origin_area,
                origin_floor=origin_floor,
                area_by_id=area_by_id,
                conversation=conversation,
            )
        )
    ]

    positive = [candidate for candidate in scored if candidate.score > 0]
    if positive:
        scored = positive

    scored.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.item.name.lower(),
            candidate.item.entity_id,
        ),
    )
    return scored[:limit]

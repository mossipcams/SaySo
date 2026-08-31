"""LFM prompt builder for ControlPlan generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from sayso_server.conversation import LastTarget, SatelliteConversationState
from sayso_server.control_plan import ControlPlan
from sayso_server.home_graph import Area, Entity, Scene, Script

CandidateItem = Entity | Scene | Script


class PromptOrigin(BaseModel):
    satellite_id: str = Field(min_length=1)
    area_name: str = Field(min_length=1)
    area_aliases: list[str] = Field(default_factory=list)


def build_lfm_prompt(
    *,
    user_text: str,
    origin: PromptOrigin,
    conversation: SatelliteConversationState,
    candidates: Sequence[CandidateItem],
    areas: Sequence[Area],
    schema: dict[str, object] | None = None,
) -> str:
    """Build an LFM prompt from schema, origin, conversation state, and candidates only."""
    area_by_id = {area.id: area for area in areas}
    candidate_by_entity_id = {
        item.entity_id: item for item in candidates if hasattr(item, "entity_id")
    }
    control_plan_schema = schema if schema is not None else ControlPlan.json_schema()

    payload = {
        "origin": {
            "satellite_id": origin.satellite_id,
            "area": origin.area_name,
            "area_aliases": list(origin.area_aliases),
        },
        "conversation_state": _serialize_conversation(
            conversation,
            candidate_by_entity_id=candidate_by_entity_id,
        ),
        "candidate_entities": [
            _serialize_candidate(item, area_by_id=area_by_id) for item in candidates
        ],
        "control_plan_schema": control_plan_schema,
        "user_text": user_text,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _serialize_conversation(
    conversation: SatelliteConversationState,
    *,
    candidate_by_entity_id: dict[str, CandidateItem],
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if conversation.last_target is not None:
        state["last_target"] = _serialize_last_target(
            conversation.last_target,
            candidate_by_entity_id=candidate_by_entity_id,
        )
    if conversation.last_intent is not None:
        state["last_intent"] = {
            "intent": conversation.last_intent.intent,
            "outcome": conversation.last_intent.outcome,
        }
    return state


def _serialize_last_target(
    last_target: LastTarget,
    *,
    candidate_by_entity_id: dict[str, CandidateItem],
) -> dict[str, Any]:
    names: list[str] = []
    aliases: list[str] = []
    for entity_id in last_target.entity_ids:
        candidate = candidate_by_entity_id.get(entity_id)
        if candidate is None:
            continue
        names.append(candidate.name)
        aliases.extend(candidate.aliases)
    return {
        "names": names,
        "aliases": sorted(set(aliases)),
    }


def _serialize_candidate(
    item: CandidateItem,
    *,
    area_by_id: dict[str, Area],
) -> dict[str, Any]:
    area_name: str | None = None
    area_aliases: list[str] = []
    if item.area_id is not None:
        area = area_by_id.get(item.area_id)
        if area is not None:
            area_name = area.name
            area_aliases = list(area.aliases)

    serialized: dict[str, Any] = {
        "name": item.name,
        "aliases": list(item.aliases),
        "capabilities": [cap.model_dump(mode="json") for cap in item.capabilities],
    }
    if area_name is not None:
        serialized["area"] = area_name
    if area_aliases:
        serialized["area_aliases"] = area_aliases

    if isinstance(item, Entity):
        serialized["domain"] = item.domain
        serialized["state"] = item.state.model_dump(mode="json")
    elif isinstance(item, Scene):
        serialized["domain"] = "scene"
    else:
        serialized["domain"] = "script"

    return serialized

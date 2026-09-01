"""LFM prompt builder for ControlPlan generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from sayso_server.conversation import LastTarget, SatelliteConversationState
from sayso_server.home_graph import Area, Entity, Scene, Script

CandidateItem = Entity | Scene | Script

GENERATION_INSTRUCTION = (
    "Reply with only one ControlPlan JSON object. "
    "No prose, explanation, or markdown."
)

def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


LFM_FEW_SHOT_USER_JSON = _compact_json(
    {
        "candidate_entities": [
            {
                "aliases": ["lamp", "reading lamp"],
                "area": "Living Room",
                "capabilities": [
                    {"kind": "power"},
                    {"kind": "brightness", "max_value": 100, "min_value": 1},
                ],
                "domain": "light",
                "name": "Floor Lamp",
                "state": {"attributes": {"brightness": 0}, "value": "off"},
            }
        ],
        "conversation_state": {},
        "origin": {
            "area": "Living Room",
            "area_aliases": [],
            "satellite_id": "sat-1",
        },
        "user_text": "turn off the floor lamp",
    },
)

LFM_FEW_SHOT_ASSISTANT_JSON = _compact_json(
    {
        "domain": "light",
        "intent": "turn off the floor lamp",
        "outcome": "action",
        "state": "off",
        "targets": ["floor lamp"],
    },
)


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
) -> str:
    """Build an LFM prompt from origin, conversation state, and candidates only."""
    area_by_id = {area.id: area for area in areas}
    candidate_by_entity_id = {
        item.entity_id: item for item in candidates if hasattr(item, "entity_id")
    }

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
        "user_text": user_text,
    }
    return f"{GENERATION_INSTRUCTION}\n{_compact_json(payload)}"


def extract_lfm_prompt_user_json(prompt: str) -> str:
    """Return the user JSON body from a built LFM prompt, without the instruction prefix."""
    json_start = prompt.index("{")
    return prompt[json_start:]


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
        "capabilities": [
            cap.model_dump(mode="json", exclude_none=True) for cap in item.capabilities
        ],
    }
    if area_name is not None:
        serialized["area"] = area_name
    if area_aliases:
        serialized["area_aliases"] = area_aliases

    if isinstance(item, Entity):
        serialized["domain"] = item.domain
        serialized["state"] = {
            "value": item.state.value,
            "attributes": dict(item.state.attributes),
        }
    elif isinstance(item, Scene):
        serialized["domain"] = "scene"
    else:
        serialized["domain"] = "script"

    return serialized

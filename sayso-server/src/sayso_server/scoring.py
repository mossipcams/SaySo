"""Candidate scoring signals for Home Graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from sayso_server.conversation import SatelliteConversationState
from sayso_server.home_graph import Area, CapabilityKind, Entity, Floor, HomeGraphSnapshot, Scene, Script
from sayso_server.normalize import normalize_labels, normalize_tokens

CandidateItem = Entity | Scene | Script

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "light": ("light", "lights", "lamp", "lamps", "dim", "brightness"),
    "climate": ("climate", "thermostat", "temperature", "hvac", "heat", "cool"),
    "binary_sensor": ("door", "sensor", "motion", "window"),
    "scene": ("scene",),
    "script": ("script", "run"),
}

_CAPABILITY_KEYWORDS: dict[CapabilityKind, tuple[str, ...]] = {
    CapabilityKind.POWER: ("turn", "on", "off", "switch", "power"),
    CapabilityKind.BRIGHTNESS: ("dim", "brightness", "bright"),
    CapabilityKind.TEMPERATURE: ("temperature", "heat", "cool", "thermostat", "hvac"),
    CapabilityKind.QUERY: ("what", "is", "are", "check", "status", "open", "closed"),
    CapabilityKind.SCENE: ("scene", "activate"),
    CapabilityKind.SCRIPT: ("script", "run"),
}

DEFAULT_AMBIGUITY_MARGIN = 0.5

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "please",
        "can",
        "you",
        "my",
        "all",
        "it",
        "back",
    },
)


@dataclass(frozen=True)
class ScoreBreakdown:
    domain: float = 0.0
    area: float = 0.0
    floor: float = 0.0
    alias: float = 0.0
    capability: float = 0.0
    state: float = 0.0
    referent: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.domain
            + self.area
            + self.floor
            + self.alias
            + self.capability
            + self.state
            + self.referent
        )


def content_tokens(tokens: list[str]) -> list[str]:
    """Drop common command filler tokens before matching."""
    return [token for token in tokens if token not in _STOP_WORDS]


def infer_domain(tokens: list[str], explicit_domain: str | None) -> str | None:
    if explicit_domain:
        return explicit_domain
    token_set = set(tokens)
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if token_set.intersection(keywords):
            return domain
    return None


def score_candidate(
    item: CandidateItem,
    *,
    query_tokens: list[str],
    inferred_domain: str | None,
    origin_area: Area | None,
    origin_floor: Floor | None,
    area_by_id: dict[str, Area],
    conversation: SatelliteConversationState | None,
) -> ScoreBreakdown:
    """Score one entity, scene, or script against query and origin signals."""
    labels = [item.name, *item.aliases]
    label_tokens = normalize_labels(labels)
    meaningful_tokens = content_tokens(query_tokens)
    token_set = set(meaningful_tokens)

    domain_score = _score_domain(item, inferred_domain)
    area_score = _score_area(item, origin_area)
    floor_score = _score_floor(item, origin_area, origin_floor, area_by_id)
    alias_score = _score_alias(token_set, meaningful_tokens, label_tokens, labels)
    capability_score = _score_capability(item, token_set)
    state_score = _score_state(item, token_set)
    referent_score = _score_referent(item, conversation)

    return ScoreBreakdown(
        domain=domain_score,
        area=area_score,
        floor=floor_score,
        alias=alias_score,
        capability=capability_score,
        state=state_score,
        referent=referent_score,
    )


def _item_domain(item: CandidateItem) -> str:
    if isinstance(item, Entity):
        return item.domain
    if isinstance(item, Scene):
        return "scene"
    return "script"


def _score_domain(item: CandidateItem, inferred_domain: str | None) -> float:
    if inferred_domain is None:
        return 0.0
    return 2.0 if _item_domain(item) == inferred_domain else 0.0


def _score_area(item: CandidateItem, origin_area: Area | None) -> float:
    if origin_area is None or item.area_id is None:
        return 0.0
    return 3.0 if item.area_id == origin_area.id else 0.0


def _score_floor(
    item: CandidateItem,
    origin_area: Area | None,
    origin_floor: Floor | None,
    area_by_id: dict[str, Area],
) -> float:
    if origin_area is None or origin_floor is None or item.area_id is None:
        return 0.0
    item_area = area_by_id.get(item.area_id)
    if item_area is None or item_area.floor_id is None:
        return 0.0
    return 1.0 if item_area.floor_id == origin_floor.id else 0.0


def _score_alias(
    token_set: set[str],
    meaningful_tokens: list[str],
    label_tokens: set[str],
    labels: list[str],
) -> float:
    if not token_set:
        return 0.0

    overlap = token_set.intersection(label_tokens)
    if not overlap:
        return 0.0

    score = min(5.0, float(len(overlap)) * 2.0)

    normalized_labels = [normalize_tokens(label) for label in labels]
    for alias_tokens in normalized_labels:
        if not alias_tokens:
            continue
        alias_set = set(alias_tokens)
        if alias_set.issubset(token_set):
            score = max(score, 5.0)

    joined_query = " ".join(meaningful_tokens)
    for label in labels:
        normalized_label = " ".join(normalize_tokens(label))
        if normalized_label and normalized_label in joined_query:
            score = max(score, 5.0)

    return score


def _score_capability(item: CandidateItem, token_set: set[str]) -> float:
    if not item.capabilities:
        return 0.0

    score = 0.0
    item_kinds = {capability.kind for capability in item.capabilities}
    for kind in item_kinds:
        keywords = _CAPABILITY_KEYWORDS.get(kind, ())
        if token_set.intersection(keywords):
            score = max(score, 1.5)
    return score


def _score_state(item: CandidateItem, token_set: set[str]) -> float:
    if not isinstance(item, Entity):
        return 0.0

    value_tokens = normalize_tokens(item.state.value)
    state_tokens = {value_tokens[0]} if value_tokens else set()
    for attribute in item.state.attributes.values():
        if isinstance(attribute, str):
            state_tokens.update(normalize_tokens(attribute))

    if token_set.intersection(state_tokens):
        return 1.0
    return 0.0


def _score_referent(
    item: CandidateItem,
    conversation: SatelliteConversationState | None,
) -> float:
    if conversation is None or conversation.last_target is None:
        return 0.0
    if item.entity_id in conversation.last_target.entity_ids:
        return 4.0
    return 0.0


def lookup_origin(
    snapshot: HomeGraphSnapshot,
    origin_area_id: str,
) -> tuple[Area | None, Floor | None]:
    area = next((candidate for candidate in snapshot.areas if candidate.id == origin_area_id), None)
    if area is None or area.floor_id is None:
        return area, None
    floor = next(
        (candidate for candidate in snapshot.floors if candidate.id == area.floor_id),
        None,
    )
    return area, floor

"""Conservative command-domain routing hints from HA registry metadata.

Two independent kinds of evidence can narrow the tool schema, and only ever
when they agree unambiguously:

1. the command names an exposed entity or says a domain word outright, or
2. the command names an area or floor whose exposed contents share one domain.

Anything fuzzy, conflicting or absent means "unknown", and an unknown route
sends the complete schema. Filtering is an optimization; it can never grant a
capability Home Assistant did not offer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    llm,
)

from custom_components.sayso.schema import (
    CompiledToolSchema,
    extract_tool_routing_metadata,
    schema_fingerprint,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_CONTROL_VERBS = frozenset(
    {
        "activate", "arm", "brighten", "close", "decrease", "dim", "disable",
        "disarm", "enable", "flip", "increase", "lock", "lower", "off", "on",
        "open", "pause", "play", "raise", "set", "start", "stop", "switch",
        "toggle", "turn", "unlock",
    }
)

# Domains a Home Assistant Assist/SaySo tool can act on. Non-control entities
# (remote, update, select, sensor, ...) may share a friendly name with one of
# these; they must not suppress the control domain hint.
_CONTROL_DOMAINS = frozenset(
    {
        "alarm_control_panel", "button", "climate", "cover", "fan", "humidifier",
        "lawn_mower", "light", "lock", "media_player", "scene", "script", "siren",
        "switch", "todo", "vacuum", "valve", "water_heater",
    }
)


@dataclass(frozen=True, slots=True)
class RoutingArea:
    """One HA area registry entry used for routing hints."""

    area_id: str
    name: str
    floor_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutingFloor:
    """One HA floor registry entry used for routing hints."""

    floor_id: str
    name: str


@dataclass(frozen=True, slots=True)
class RoutingDevice:
    """One HA device registry entry used for routing hints."""

    device_id: str
    area_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutingEntity:
    """One exposed entity used for routing hints."""

    entity_id: str
    domain: str
    name: str
    aliases: tuple[str, ...] = ()
    area_id: str | None = None
    device_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutingCatalog:
    """HA-provided entity names and domains for routing."""

    entities: tuple[RoutingEntity, ...]

    @property
    def domains(self) -> frozenset[str]:
        return frozenset(entity.domain for entity in self.entities)


@dataclass(frozen=True, slots=True)
class RoutingRegistries:
    """HA area, floor, and device registry metadata for routing hints."""

    areas: tuple[RoutingArea, ...] = ()
    floors: tuple[RoutingFloor, ...] = ()
    devices: tuple[RoutingDevice, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutingPreferences:
    """Satellite-preferred area/floor supporting evidence."""

    area: str | None = None
    floor: str | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; punctuation and case never matter."""
    return _TOKEN_RE.findall(text.casefold())


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_matches_domain(token: str, domain: str) -> bool:
    return (
        token == domain
        or token in {f"{domain}s", f"{domain}es"}
        or _singularize(token) == domain
    )


def _phrase_in_tokens(phrase_tokens: list[str], command_tokens: list[str]) -> bool:
    """Return whether the phrase appears as a verbatim run of command tokens."""
    width = len(phrase_tokens)
    if not width:
        return False
    return any(
        command_tokens[index : index + width] == phrase_tokens
        for index in range(len(command_tokens) - width + 1)
    )


def _control_tokens(command: str, catalog: RoutingCatalog) -> list[str] | None:
    """Tokenize a command, or return None when routing must not even try."""
    tokens = _tokenize(command)
    if not tokens or not catalog.entities:
        return None
    if not any(token in _CONTROL_VERBS for token in tokens):
        return None
    return tokens


def _resolve_domain_hint(matched_domains: set[str]) -> str | None:
    """Return the domain hint, preferring a lone control domain.

    A non-control entity that shares a friendly name with a control entity
    (``remote.living_room_tv`` vs ``media_player.living_room_tv``) must not
    suppress the hint. Two control domains stay ambiguous.
    """
    if len(matched_domains) == 1:
        return next(iter(matched_domains))
    control_domains = matched_domains & _CONTROL_DOMAINS
    if len(control_domains) == 1:
        return next(iter(control_domains))
    return None


def _identify_from_entity_and_domain_terms(
    command: str, catalog: RoutingCatalog
) -> str | None:
    """Match exposed entity names, their aliases, and bare domain words."""
    command_tokens = _control_tokens(command, catalog)
    if command_tokens is None:
        return None

    matched = {
        entity.domain
        for entity in catalog.entities
        if any(
            _phrase_in_tokens(_tokenize(phrase), command_tokens)
            for phrase in (entity.name, *entity.aliases)
            if phrase.strip()
        )
    }
    matched |= {
        domain
        for domain in catalog.domains
        if any(_token_matches_domain(token, domain) for token in command_tokens)
    }
    return _resolve_domain_hint(matched)


def _named_matches[T](
    entries: Sequence[T],
    id_of: Callable[[T], str],
    command_tokens: list[str],
    preferred: str | None,
) -> list[T]:
    """Return registry entries the command names, resolving ties by preference.

    One match is unambiguous. Several stay ambiguous unless the satellite's
    preferred area or floor is one of them — supporting evidence can break a
    tie but never creates a match on its own.
    """
    matched = [
        entry
        for entry in entries
        if _phrase_in_tokens(_tokenize(entry.name), command_tokens)
    ]
    if len(matched) <= 1:
        return matched
    if preferred is None:
        return []

    folded = preferred.casefold()
    resolved = next(
        (
            id_of(entry)
            for entry in matched
            if id_of(entry) == preferred or entry.name.casefold() == folded
        ),
        None,
    )
    return [entry for entry in matched if id_of(entry) == resolved] if resolved else []


def _entity_area_id(
    entity: RoutingEntity, devices: dict[str, RoutingDevice]
) -> str | None:
    """An entity's own area, or the area of the device it belongs to."""
    if entity.area_id is not None:
        return entity.area_id
    device = devices.get(entity.device_id) if entity.device_id else None
    return device.area_id if device is not None else None


def _identify_from_area_and_floor_evidence(
    command: str,
    catalog: RoutingCatalog,
    registries: RoutingRegistries,
    *,
    preferences: RoutingPreferences | None,
) -> str | None:
    """Narrow by place: the exposed contents of a named area or floor."""
    command_tokens = _control_tokens(command, catalog)
    if command_tokens is None:
        return None

    preferred_area = preferences.area if preferences else None
    preferred_floor = preferences.floor if preferences else None
    matched_areas = _named_matches(
        registries.areas, lambda area: area.area_id, command_tokens, preferred_area
    )
    matched_floors = _named_matches(
        registries.floors, lambda floor: floor.floor_id, command_tokens, preferred_floor
    )

    area_ids = {area.area_id for area in matched_areas}
    for floor in matched_floors:
        area_ids |= {
            area.area_id
            for area in registries.areas
            if area.floor_id == floor.floor_id
        }
    if not area_ids:
        return None

    devices = {device.device_id: device for device in registries.devices}
    return _resolve_domain_hint(
        {
            entity.domain
            for entity in catalog.entities
            if _entity_area_id(entity, devices) in area_ids
        }
    )


def identify_command_domain(
    command: str,
    catalog: RoutingCatalog,
    *,
    registries: RoutingRegistries | None = None,
    preferences: RoutingPreferences | None = None,
) -> str | None:
    """Return a domain hint only for exact, unambiguous token or registry matches."""
    domain_hint = _identify_from_entity_and_domain_terms(command, catalog)
    if domain_hint is not None or registries is None:
        return domain_hint
    return _identify_from_area_and_floor_evidence(
        command, catalog, registries, preferences=preferences
    )


def select_tools_for_domain(
    compiled_tools: tuple[dict[str, Any], ...],
    source_tools: list[llm.Tool],
    domain_hint: str | None,
) -> tuple[dict[str, Any], ...]:
    """Return compiled tools compatible with a confident domain hint.

    A tool survives unless Home Assistant's own metadata says it acts on other
    domains. Unknown metadata, scripts and query tools are always retained: a
    routing guess must never be able to hide a tool the user needs.
    """
    if domain_hint is None:
        return compiled_tools

    metadata_by_name = {
        tool.name: extract_tool_routing_metadata(tool) for tool in source_tools
    }
    return tuple(
        compiled_tool
        for compiled_tool in compiled_tools
        if (metadata := metadata_by_name.get(compiled_tool["function"]["name"])) is None
        or metadata.retain_always
        or metadata.declared_domains is None
        or domain_hint in metadata.declared_domains
    )


def select_schema_for_domain(
    complete_schema: CompiledToolSchema,
    source_tools: list[llm.Tool],
    domain_hint: str | None,
) -> CompiledToolSchema:
    """Return the active schema for a domain hint, or the complete schema unchanged."""
    selected_tools = select_tools_for_domain(
        complete_schema.tools, source_tools, domain_hint
    )
    if selected_tools == complete_schema.tools:
        return complete_schema
    return CompiledToolSchema(
        tools=selected_tools,
        fingerprint=schema_fingerprint(list(selected_tools)),
    )


@callback
def build_routing_catalog(
    hass: HomeAssistant,
    *,
    assistant: str = CONVERSATION_DOMAIN,
) -> RoutingCatalog:
    """Build a routing catalog from exposed HA entities."""
    entity_reg = er.async_get(hass)
    entities: list[RoutingEntity] = []
    for entity_id, entry in entity_reg.entities.items():
        if not async_should_expose(hass, assistant, entity_id):
            continue
        state = hass.states.get(entity_id)
        if state is None:
            continue
        entities.append(
            RoutingEntity(
                entity_id=entity_id,
                domain=state.domain,
                name=str(state.name) if state.name else entity_id,
                aliases=tuple(str(alias) for alias in entry.aliases),
                area_id=entry.area_id,
                device_id=entry.device_id,
            )
        )
    return RoutingCatalog(entities=tuple(entities))


@callback
def build_routing_registries(hass: HomeAssistant) -> RoutingRegistries:
    """Build area, floor, and device registry snapshots for routing hints."""
    return RoutingRegistries(
        areas=tuple(
            RoutingArea(area.id, area.name, area.floor_id)
            for area in ar.async_get(hass).async_list_areas()
        ),
        floors=tuple(
            RoutingFloor(floor.floor_id, floor.name)
            for floor in fr.async_get(hass).async_list_floors()
        ),
        devices=tuple(
            RoutingDevice(device.id, device.area_id)
            for device in dr.async_get(hass).devices.values()
        ),
    )


@callback
def build_routing_preferences(
    hass: HomeAssistant,
    llm_context: llm.LLMContext,
    *,
    satellite_id: str | None = None,
) -> RoutingPreferences | None:
    """Return preferred area/floor from the requesting device or satellite."""
    device_id = llm_context.device_id
    if device_id is None and satellite_id is not None:
        satellite = er.async_get(hass).async_get(satellite_id)
        device_id = satellite.device_id if satellite is not None else None
    if device_id is None:
        return None

    device = dr.async_get(hass).async_get(device_id)
    if device is None or device.area_id is None:
        return None

    area = ar.async_get(hass).async_get_area(device.area_id)
    if area is None:
        return None
    return RoutingPreferences(area=area.id, floor=area.floor_id)

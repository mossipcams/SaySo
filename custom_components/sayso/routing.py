"""Conservative command-domain routing hints from HA registry metadata."""

from __future__ import annotations

import re
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
        "activate",
        "arm",
        "brighten",
        "close",
        "decrease",
        "dim",
        "disable",
        "disarm",
        "enable",
        "flip",
        "increase",
        "lock",
        "lower",
        "off",
        "on",
        "open",
        "pause",
        "play",
        "raise",
        "set",
        "start",
        "stop",
        "switch",
        "toggle",
        "turn",
        "unlock",
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


def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    return " ".join(_TOKEN_RE.findall(lowered))


def _tokenize(text: str) -> list[str]:
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
    if token == domain:
        return True
    if token in {f"{domain}s", f"{domain}es"}:
        return True
    return _singularize(token) == domain


def _has_control_verb(command_tokens: list[str]) -> bool:
    return any(token in _CONTROL_VERBS for token in command_tokens)


def _phrase_in_tokens(phrase_tokens: list[str], command_tokens: list[str]) -> bool:
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    for index in range(len(command_tokens) - width + 1):
        if command_tokens[index : index + width] == phrase_tokens:
            return True
    return False


def _entity_phrase_tokens(entity: RoutingEntity) -> list[list[str]]:
    phrases = (entity.name, *entity.aliases)
    return [_tokenize(_normalize_text(phrase)) for phrase in phrases if phrase.strip()]


def _domains_from_entity_matches(
    catalog: RoutingCatalog,
    command_tokens: list[str],
) -> set[str]:
    matched: set[str] = set()
    for entity in catalog.entities:
        for phrase_tokens in _entity_phrase_tokens(entity):
            if _phrase_in_tokens(phrase_tokens, command_tokens):
                matched.add(entity.domain)
                break
    return matched


def _domains_from_domain_terms(
    catalog: RoutingCatalog,
    command_tokens: list[str],
) -> set[str]:
    matched: set[str] = set()
    for domain in catalog.domains:
        for token in command_tokens:
            if _token_matches_domain(token, domain):
                matched.add(domain)
                break
    return matched


def _identify_from_entity_and_domain_terms(
    command: str,
    catalog: RoutingCatalog,
) -> str | None:
    normalized = _normalize_text(command)
    if not normalized or not catalog.entities:
        return None

    command_tokens = _tokenize(normalized)
    if not _has_control_verb(command_tokens):
        return None

    matched_domains = _domains_from_entity_matches(
        catalog,
        command_tokens,
    ) | _domains_from_domain_terms(catalog, command_tokens)

    if len(matched_domains) == 1:
        return next(iter(matched_domains))
    return None


def _devices_by_id(registries: RoutingRegistries) -> dict[str, RoutingDevice]:
    return {device.device_id: device for device in registries.devices}


def _entity_area_id(
    entity: RoutingEntity,
    devices: dict[str, RoutingDevice],
) -> str | None:
    if entity.area_id is not None:
        return entity.area_id
    if entity.device_id is None:
        return None
    device = devices.get(entity.device_id)
    if device is None:
        return None
    return device.area_id


def _name_phrase_tokens(name: str) -> list[str]:
    return _tokenize(_normalize_text(name))


def _areas_by_id(registries: RoutingRegistries) -> dict[str, RoutingArea]:
    return {area.area_id: area for area in registries.areas}


def _floors_by_id(registries: RoutingRegistries) -> dict[str, RoutingFloor]:
    return {floor.floor_id: floor for floor in registries.floors}


def _resolve_preferred_area_id(
    preferences: RoutingPreferences | None,
    areas_by_id: dict[str, RoutingArea],
) -> str | None:
    if preferences is None or preferences.area is None:
        return None
    if preferences.area in areas_by_id:
        return preferences.area
    preferred = preferences.area.casefold()
    for area in areas_by_id.values():
        if area.name.casefold() == preferred:
            return area.area_id
    return None


def _resolve_preferred_floor_id(
    preferences: RoutingPreferences | None,
    floors_by_id: dict[str, RoutingFloor],
) -> str | None:
    if preferences is None or preferences.floor is None:
        return None
    if preferences.floor in floors_by_id:
        return preferences.floor
    preferred = preferences.floor.casefold()
    for floor in floors_by_id.values():
        if floor.name.casefold() == preferred:
            return floor.floor_id
    return None


def _matching_areas_from_command(
    command_tokens: list[str],
    registries: RoutingRegistries,
    *,
    preferences: RoutingPreferences | None,
) -> list[RoutingArea]:
    areas_by_id = _areas_by_id(registries)
    matched: list[RoutingArea] = []
    for area in registries.areas:
        phrase_tokens = _name_phrase_tokens(area.name)
        if _phrase_in_tokens(phrase_tokens, command_tokens):
            matched.append(area)

    if not matched:
        return []

    if len(matched) == 1:
        return matched

    preferred_area_id = _resolve_preferred_area_id(preferences, areas_by_id)
    if preferred_area_id is None:
        return []

    return [area for area in matched if area.area_id == preferred_area_id]


def _matching_floors_from_command(
    command_tokens: list[str],
    registries: RoutingRegistries,
    *,
    preferences: RoutingPreferences | None,
) -> list[RoutingFloor]:
    floors_by_id = _floors_by_id(registries)
    matched: list[RoutingFloor] = []
    for floor in registries.floors:
        phrase_tokens = _name_phrase_tokens(floor.name)
        if _phrase_in_tokens(phrase_tokens, command_tokens):
            matched.append(floor)

    if not matched:
        return []

    if len(matched) == 1:
        return matched

    preferred_floor_id = _resolve_preferred_floor_id(preferences, floors_by_id)
    if preferred_floor_id is None:
        return []

    return [floor for floor in matched if floor.floor_id == preferred_floor_id]


def _area_ids_for_floor(floor_id: str, registries: RoutingRegistries) -> set[str]:
    return {
        area.area_id
        for area in registries.areas
        if area.floor_id == floor_id
    }


def _domains_for_area_ids(
    catalog: RoutingCatalog,
    area_ids: set[str],
    devices: dict[str, RoutingDevice],
) -> set[str]:
    matched: set[str] = set()
    for entity in catalog.entities:
        entity_area_id = _entity_area_id(entity, devices)
        if entity_area_id is not None and entity_area_id in area_ids:
            matched.add(entity.domain)
    return matched


def _identify_from_area_and_floor_evidence(
    command: str,
    catalog: RoutingCatalog,
    registries: RoutingRegistries,
    *,
    preferences: RoutingPreferences | None,
) -> str | None:
    normalized = _normalize_text(command)
    if not normalized or not catalog.entities:
        return None

    command_tokens = _tokenize(normalized)
    if not _has_control_verb(command_tokens):
        return None

    devices = _devices_by_id(registries)
    matched_areas = _matching_areas_from_command(
        command_tokens,
        registries,
        preferences=preferences,
    )
    matched_floors = _matching_floors_from_command(
        command_tokens,
        registries,
        preferences=preferences,
    )

    if not matched_areas and not matched_floors:
        return None

    area_ids: set[str] = set()
    if matched_areas:
        area_ids.update(area.area_id for area in matched_areas)
    if matched_floors:
        for floor in matched_floors:
            area_ids.update(_area_ids_for_floor(floor.floor_id, registries))

    if not area_ids:
        return None

    matched_domains = _domains_for_area_ids(catalog, area_ids, devices)
    if len(matched_domains) == 1:
        return next(iter(matched_domains))
    return None


def identify_command_domain(
    command: str,
    catalog: RoutingCatalog,
    *,
    registries: RoutingRegistries | None = None,
    preferences: RoutingPreferences | None = None,
) -> str | None:
    """Return a domain hint only for exact, unambiguous token or registry matches."""
    domain_hint = _identify_from_entity_and_domain_terms(command, catalog)
    if domain_hint is not None:
        return domain_hint

    if registries is None:
        return None

    return _identify_from_area_and_floor_evidence(
        command,
        catalog,
        registries,
        preferences=preferences,
    )


def _filter_compiled_tools(
    compiled_tools: tuple[dict[str, Any], ...],
    source_tools: list[llm.Tool],
    domain_hint: str,
) -> tuple[dict[str, Any], ...]:
    """Return compiled tools compatible with a confident domain hint."""
    metadata_by_name = {
        tool.name: extract_tool_routing_metadata(tool) for tool in source_tools
    }

    selected: list[dict[str, Any]] = []
    for compiled_tool in compiled_tools:
        name = compiled_tool["function"]["name"]
        metadata = metadata_by_name.get(name)
        if metadata is None:
            selected.append(compiled_tool)
            continue
        if (
            metadata.retain_always
            or metadata.declared_domains is None
            or domain_hint in metadata.declared_domains
        ):
            selected.append(compiled_tool)

    return tuple(selected)


def select_schema_for_domain(
    complete_schema: CompiledToolSchema,
    source_tools: list[llm.Tool],
    domain_hint: str | None,
) -> CompiledToolSchema:
    """Return the active schema for a domain hint, or the complete schema unchanged."""
    if domain_hint is None:
        return complete_schema

    selected_tools = _filter_compiled_tools(
        complete_schema.tools,
        source_tools,
        domain_hint,
    )
    if selected_tools == complete_schema.tools:
        return complete_schema

    return CompiledToolSchema(
        tools=selected_tools,
        fingerprint=schema_fingerprint(list(selected_tools)),
    )


def select_tools_for_domain(
    compiled_tools: tuple[dict[str, Any], ...],
    source_tools: list[llm.Tool],
    domain_hint: str | None,
) -> tuple[dict[str, Any], ...]:
    """Return a safe compiled subset for a confident domain hint."""
    if domain_hint is None:
        return compiled_tools

    selected_tools = _filter_compiled_tools(
        compiled_tools,
        source_tools,
        domain_hint,
    )
    if selected_tools == compiled_tools:
        return compiled_tools

    return selected_tools


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
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass)
    device_reg = dr.async_get(hass)
    return RoutingRegistries(
        areas=tuple(
            RoutingArea(area.id, area.name, area.floor_id)
            for area in area_reg.async_list_areas()
        ),
        floors=tuple(
            RoutingFloor(floor.floor_id, floor.name)
            for floor in floor_reg.async_list_floors()
        ),
        devices=tuple(
            RoutingDevice(device.id, device.area_id)
            for device in device_reg.devices.values()
        ),
    )


@callback
def build_routing_preferences(
    hass: HomeAssistant,
    llm_context: llm.LLMContext,
) -> RoutingPreferences | None:
    """Return preferred area/floor from the requesting satellite device."""
    if llm_context.device_id is None:
        return None

    device_reg = dr.async_get(hass)
    device = device_reg.async_get(llm_context.device_id)
    if device is None or device.area_id is None:
        return None

    area_reg = ar.async_get(hass)
    area = area_reg.async_get_area(device.area_id)
    if area is None:
        return None

    return RoutingPreferences(area=area.id, floor=area.floor_id)

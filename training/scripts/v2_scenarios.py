#!/usr/bin/env python3
"""The label-first scenario vocabulary of the v1/v2 synthetic corpus.

A row starts as a *spec*: a home, an expected outcome, and the metadata that
says what the row is teaching. Language is applied afterwards and may never
change any of it — which is why the vocabulary lives apart from the verbaliser
and the judge that consume it.
"""

from __future__ import annotations

import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.schema import (  # noqa: E402
    tool_schema_map,
    v2_openai_tools,
    validate_tool_arguments,
)
from generators.tools import script_tool_name  # noqa: E402

CATEGORY_WEIGHTS = {
    "clean_direct": 10,
    "conversational": 15,
    "entity_identity": 10,
    "multi_action_exclusion": 20,
    "stt_corrupted": 15,
    "status": 10,
    "ambiguity": 10,
    "unsupported_no_action": 10,
}

_AREAS = (
    "Kitchen",
    "Living Room",
    "Primary Bedroom",
    "Guest Room",
    "Office",
    "Garage",
    "Hallway",
    "Patio",
    "Laundry Room",
    "Workshop",
    "Nursery",
    "Basement",
)
_FLOORS = ("Upstairs", "Downstairs", "Main Floor", "Basement")
_PREFIXES = ("North", "South", "East", "West", "Main", "Side", "Corner", "Ceiling")
_DEVICES = (
    ("light", "Light", ("on", "off"), ("on", "off", "brightness", "color")),
    ("fan", "Fan", ("on", "off"), ("on", "off", "percentage")),
    ("switch", "Outlet", ("on", "off"), ("on", "off")),
    ("blinds", "Blinds", ("open", "closed"), ("open", "close")),
    ("garage_door", "Garage Door", ("open", "closed"), ("open", "close")),
    ("lock", "Door Lock", ("locked", "unlocked"), ("lock", "unlock")),
)
_SPECIAL_NAMES = {
    "light": ("Kids' Room Light", "O'Malley's Porch Light"),
    "fan": ("Joe's Workshop Fan", "Children's Bedroom Fan"),
    "switch": ("McKay's Office Outlet", "Joe's Desk Outlet"),
    "blinds": ("Children's Bedroom Blinds", "O'Malley's Study Blinds"),
    "garage_door": ("Joe's Garage Door", "O'Malley's Garage Door"),
    "lock": ("McKay's Front Door Lock", "Children's Door Lock"),
}
_KIND_TO_GENERIC_NOUN = {
    "light": "light",
    "fan": "fan",
    "switch": "outlet",
    "blinds": "blinds",
    "garage_door": "garage door",
    "lock": "door",
}
_GENERIC_NOUN_TO_KIND = {noun: kind for kind, noun in _KIND_TO_GENERIC_NOUN.items()}
_UNAVAILABLE_TYPE = {
    "light": "lights",
    "fan": "fans",
    "switch": "outlets",
    "blinds": "blinds",
    "garage_door": "garage doors",
    "lock": "doors",
}


def _entities_of_kind_in_area(home: dict[str, Any], kind: str, area: str) -> list[dict[str, Any]]:
    return [entity for entity in home["entities"] if entity["kind"] == kind and entity["area"] == area]


def _expected_generic_in_sayso_area(
    home: dict[str, Any],
    kind: str,
    rng: random.Random,
    *,
    turn_on: bool = True,
) -> dict[str, Any]:
    """Recipe 7: generic no-area utterances resolve only in sayso_entity_area."""
    area = home["sayso_entity_area"]
    matches = _entities_of_kind_in_area(home, kind, area)
    if not matches:
        return {
            "kind": "no_action",
            "response": "area_unavailable",
            "calls": [],
            "unavailable": {"area": area.casefold(), "type": _UNAVAILABLE_TYPE[kind]},
        }
    if len(matches) == 1:
        return {"kind": "action", "calls": [_control_call(matches[0], turn_on, rng)]}
    return {"kind": "no_action", "response": "clarify", "calls": []}


def _generic_no_area_hint(kind: str) -> str:
    noun = _KIND_TO_GENERIC_NOUN[kind]
    if kind == "blinds":
        return f"open the {noun}"
    if kind == "lock":
        return f"lock the {noun}"
    if kind == "fan":
        return f"turn off the {noun}"
    return f"turn on the {noun}"


def _is_generic_no_area_hint(hint: str) -> bool:
    if not hint:
        return False
    lowered = hint.casefold()
    if any(token in lowered for token in ("kitchen", "office", "living room", "hallway", "patio", "garage", "bedroom")):
        return False
    return any(hint.endswith(f"the {noun}") or f"the {noun}" in lowered for noun in _GENERIC_NOUN_TO_KIND)


def _entity(index: int, slot: int, rng: random.Random) -> dict[str, Any]:
    kind, noun, states, capabilities = _DEVICES[(index + slot) % len(_DEVICES)]
    domain = "cover" if kind in {"blinds", "garage_door"} else kind
    device_class = {"blinds": "blind", "garage_door": "garage", "lock": "door"}.get(kind)
    area = _AREAS[(index * 3 + slot * 5) % len(_AREAS)]
    floor = _FLOORS[(index + slot) % len(_FLOORS)]
    if slot == 0 and index % 7 == 0:
        names = _SPECIAL_NAMES[kind]
        name = names[(index // 7) % len(names)]
    else:
        prefix = _PREFIXES[(index * 5 + slot * 3) % len(_PREFIXES)]
        name = f"{area} {prefix} {noun}"
    alias = f"{area} {noun}"
    slug = "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")
    return {
        "entity_id": f"{domain}.{slug}",
        "name": name,
        "aliases": [alias, f"{prefix if 'prefix' in locals() else area} {noun}"],
        "domain": domain,
        "kind": kind,
        "device_class": device_class,
        "area": area,
        "floor": floor,
        "state": rng.choice(states),
        "capabilities": list(capabilities),
    }


def _home(index: int, rng: random.Random, *, sayso_entity_area: str | None = None) -> dict[str, Any]:
    entities = [_entity(index, slot, rng) for slot in range(6)]
    if not any("'" in entity["name"] for entity in entities):
        entity = entities[0]
        noun = next(noun for kind, noun, _states, _caps in _DEVICES if kind == entity["kind"])
        entity["name"] = f"Joe's {entity['area']} {noun}"
        entity["aliases"].append(entity["name"].replace("'", ""))
        slug = "".join(char.casefold() if char.isalnum() else "_" for char in entity["name"]).strip("_")
        entity["entity_id"] = f"{entity['domain']}.{slug}"
    area = sayso_entity_area or entities[index % len(entities)]["area"]
    return {
        "home_id": f"home_{index:06d}",
        "sayso_entity_area": area,
        "entities": entities,
    }


def _make_entity(
    *,
    name: str,
    kind: str,
    area: str,
    rng: random.Random,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    _kind, noun, states, capabilities = next(row for row in _DEVICES if row[0] == kind)
    domain = "cover" if kind in {"blinds", "garage_door"} else kind
    device_class = {"blinds": "blind", "garage_door": "garage", "lock": "door"}.get(kind)
    slug = "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")
    return {
        "entity_id": f"{domain}.{slug}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "device_class": device_class,
        "area": area,
        "floor": _FLOORS[0],
        "state": rng.choice(states),
        "capabilities": list(capabilities),
    }


def _stt_variant(entity: dict[str, Any]) -> tuple[str, str]:
    word, replacement, kind = {
        "light": ("light", "lite", "homophone_light"),
        "fan": ("fan", "van", "consonant_fan"),
        "switch": ("outlet", "out let", "word_boundary_outlet"),
        "blinds": ("blinds", "blends", "vowel_blinds"),
        "garage_door": ("garage", "garaj", "phonetic_garage"),
        "lock": ("lock", "lok", "phonetic_lock"),
    }[entity["kind"]]
    return re.sub(rf"\b{word}\b", replacement, entity["aliases"][0], flags=re.I).casefold(), kind


def _control_call(entity: dict[str, Any], turn_on: bool, rng: random.Random) -> dict[str, Any]:
    domain = entity["domain"]
    kind = entity["kind"]
    args: dict[str, Any] = {"name": entity["name"]}
    if domain in {"light", "fan", "switch"}:
        args["domain"] = [domain]
    elif kind == "blinds":
        args["device_class"] = ["blind"]
    elif kind == "garage_door":
        args["device_class"] = ["garage"]
    elif kind == "lock":
        args["device_class"] = ["door"]
    return {"name": "HassTurnOn" if turn_on else "HassTurnOff", "arguments": args}


def _single_action(entity: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if entity["kind"] == "light" and rng.randrange(3) == 0:
        return {
            "kind": "action",
            "calls": [
                {
                    "name": "HassLightSet",
                    "arguments": {"name": entity["name"], "domain": ["light"], "brightness": rng.randrange(10, 101)},
                }
            ],
        }
    if entity["kind"] == "fan" and rng.randrange(3) == 0:
        return {
            "kind": "action",
            "calls": [
                {
                    "name": "HassFanSetSpeed",
                    "arguments": {"name": entity["name"], "domain": ["fan"], "percentage": rng.randrange(10, 101)},
                }
            ],
        }
    return {"kind": "action", "calls": [_control_call(entity, bool(rng.randrange(2)), rng)]}


_UNSUPPORTED_HINTS = {
    "refuse": "disable the smoke alarm safety system",
    "clarify": "set the light to",
    "unsupported": "play music in the garage",
}

_APOSTROPHE_NAMES = (
    "Joe's Kitchen Light",
    "O'Malley's Study Blinds",
    "Kids' Room Light",
    "Joe's Guest Room Door Lock",
)

def _status_call(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"]}
    if entity["domain"] in {"light", "fan", "switch"}:
        args["domain"] = [entity["domain"]]
    return {"name": "GetLiveContext", "arguments": args}


def _area_scenario(index: int, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Build locked recipe-7 homes: sayso area default, clarify, or zero-match."""
    scenario = (
        "one_default_light",
        "clarify_default_light",
        "named_office_light",
        "clarify_named_kitchen_light",
        "one_default_fan",
        "clarify_outlet",
        "one_default_blinds",
        "one_default_lock",
        "zero_lights",
    )[index % 9]
    sayso_area = {
        "one_default_light": "Kitchen",
        "clarify_default_light": "Kitchen",
        "named_office_light": "Kitchen",
        "clarify_named_kitchen_light": "Kitchen",
        "one_default_fan": "Living Room",
        "clarify_outlet": "Hallway",
        "one_default_blinds": "Patio",
        "one_default_lock": "Kitchen",
        "zero_lights": "Kitchen",
    }[scenario]
    entities: list[dict[str, Any]] = []
    if scenario in {"one_default_light", "clarify_default_light", "zero_lights"}:
        if scenario != "zero_lights":
            entities.append(
                _make_entity(
                    name="Kitchen Sink Cool Light",
                    kind="light",
                    area="Kitchen",
                    rng=rng,
                    aliases=["light", "kitchen light"],
                )
            )
        if scenario == "clarify_default_light":
            entities.append(
                _make_entity(
                    name="Kitchen Ceiling Cool Light",
                    kind="light",
                    area="Kitchen",
                    rng=rng,
                    aliases=["light", "kitchen light"],
                )
            )
        entities.append(
            _make_entity(name="Office Main Light", kind="light", area="Office", rng=rng)
        )
        hint = "turn on the light"
        if scenario == "one_default_light":
            expected = {
                "kind": "action",
                "calls": [_control_call(entities[0], True, rng)],
            }
        elif scenario == "clarify_default_light":
            expected = {"kind": "no_action", "response": "clarify", "calls": []}
        else:
            expected = {
                "kind": "no_action",
                "response": "area_unavailable",
                "calls": [],
                "unavailable": {"area": "kitchen", "type": "lights"},
            }
    elif scenario == "named_office_light":
        entities.append(
            _make_entity(name="Office Main Light", kind="light", area="Office", rng=rng)
        )
        hint = "turn on the office light"
        expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
    elif scenario == "clarify_named_kitchen_light":
        entities.extend(
            [
                _make_entity(
                    name="Kitchen Sink Cool Light",
                    kind="light",
                    area="Kitchen",
                    rng=rng,
                    aliases=["kitchen light"],
                ),
                _make_entity(
                    name="Kitchen Ceiling Cool Light",
                    kind="light",
                    area="Kitchen",
                    rng=rng,
                    aliases=["kitchen light"],
                ),
            ]
        )
        hint = "turn on the kitchen light"
        expected = {"kind": "no_action", "response": "clarify", "calls": []}
    elif scenario == "one_default_fan":
        entities.extend(
            [
                _make_entity(
                    name="Living Room Ceiling Fan",
                    kind="fan",
                    area="Living Room",
                    rng=rng,
                    aliases=["fan"],
                ),
                _make_entity(name="Workshop West Fan", kind="fan", area="Workshop", rng=rng),
            ]
        )
        hint = "turn off the fan"
        expected = {"kind": "action", "calls": [_control_call(entities[0], False, rng)]}
    elif scenario == "clarify_outlet":
        entities.extend(
            [
                _make_entity(
                    name="Hallway East Outlet",
                    kind="switch",
                    area="Hallway",
                    rng=rng,
                    aliases=["outlet"],
                ),
                _make_entity(
                    name="Hallway West Outlet",
                    kind="switch",
                    area="Hallway",
                    rng=rng,
                    aliases=["outlet"],
                ),
                _make_entity(name="Nursery East Outlet", kind="switch", area="Nursery", rng=rng),
            ]
        )
        hint = "turn on the outlet"
        expected = {"kind": "no_action", "response": "clarify", "calls": []}
    elif scenario == "one_default_blinds":
        entities.extend(
            [
                _make_entity(
                    name="Patio South Blinds",
                    kind="blinds",
                    area="Patio",
                    rng=rng,
                    aliases=["blinds"],
                ),
                _make_entity(
                    name="Joe's Workshop Blinds",
                    kind="blinds",
                    area="Workshop",
                    rng=rng,
                ),
            ]
        )
        hint = "open the blinds"
        expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
    else:  # one_default_lock
        entities.extend(
            [
                _make_entity(
                    name="Kitchen Back Door Lock",
                    kind="lock",
                    area="Kitchen",
                    rng=rng,
                    aliases=["door"],
                ),
                _make_entity(
                    name="Patio Side Door Lock",
                    kind="lock",
                    area="Patio",
                    rng=rng,
                ),
            ]
        )
        hint = "lock the door"
        expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
    home = {
        "home_id": f"area_home_{index:06d}",
        "sayso_entity_area": sayso_area,
        "entities": entities,
    }
    return home, expected, hint, scenario


def _spec(category: str, index: int, seed: int) -> dict[str, Any]:
    rng = random.Random((seed << 24) ^ index)
    home = _home(index, rng)
    target = rng.choice(home["entities"])
    expected: dict[str, Any]
    excluded: list[str] = []
    spoken_targets: dict[str, str] = {}
    subtype = category
    request_hint = ""
    stt_corruption = None

    if category == "multi_action_exclusion":
        selected = home["entities"][: 2 + index % 2]
        expected = {
            "kind": "action",
            "calls": [_control_call(entity, bool((index + slot) % 2), rng) for slot, entity in enumerate(selected)],
        }
        excluded = [home["entities"][4]["name"]]
        subtype = "exclusion" if index % 3 else "exact_call_count"
    elif category == "status":
        target = home["entities"][index % len(home["entities"])]
        expected = {
            "kind": "status",
            "calls": [_status_call(target)],
            "state": target["state"],
        }
        subtype = ("named_device", "area_device", "state_query")[index % 3]
    elif category == "ambiguity":
        home, expected, request_hint, subtype = _area_scenario(index, rng)
        target_names = [
            call["arguments"]["name"] for call in expected.get("calls", []) if "name" in call["arguments"]
        ]
        return {
            "candidate_id": f"candidate_{index:06d}",
            "seed": seed,
            "category": category,
            "subcategory": subtype,
            "home": home,
            "expected": expected,
            "target_names": target_names,
            "spoken_targets": spoken_targets,
            "excluded_names": excluded,
            "contrastive_group": None,
            "request_hint": request_hint,
            "stt_corruption": stt_corruption,
            "utterance": None,
        }
    elif category == "unsupported_no_action":
        response = ("unsupported", "refuse", "clarify")[index % 3]
        expected = {"kind": "no_action", "response": response, "calls": []}
        subtype = response
        request_hint = _UNSUPPORTED_HINTS[response]
    elif category == "conversational":
        expected = _single_action(target, rng)
        subtype = ("deferral", "polite_brightness", "casual_fan")[index % 3]
        if subtype == "polite_brightness" and target["kind"] != "light":
            target = next(e for e in home["entities"] if e["kind"] == "light")
            expected = {
                "kind": "action",
                "calls": [
                    {
                        "name": "HassLightSet",
                        "arguments": {
                            "name": target["name"],
                            "domain": ["light"],
                            "brightness": 64,
                        },
                    }
                ],
            }
        elif subtype == "casual_fan" and target["kind"] != "fan":
            target = next((e for e in home["entities"] if e["kind"] == "fan"), target)
            expected = _single_action(target, rng)
    elif category == "clean_direct":
        expected = _single_action(target, rng)
        subtype = "catalog"
    elif category == "stt_corrupted":
        expected = _single_action(target, rng)
        spoken_targets[target["name"]], stt_corruption = _stt_variant(target)
        subtype = "canonical_resolution"
    elif category == "entity_identity":
        subtype = ("apostrophe", "casing", "alias", "similar_name")[index % 4]
        if subtype == "apostrophe":
            apostrophe_name = _APOSTROPHE_NAMES[index % len(_APOSTROPHE_NAMES)]
            kind = ("light", "blinds", "light", "lock")[index % 4]
            target = _make_entity(name=apostrophe_name, kind=kind, area=home["sayso_entity_area"], rng=rng)
            home["entities"][0] = target
            expected = _single_action(target, rng)
        elif subtype == "casing":
            spoken_targets[target["name"]] = target["name"].casefold()
            expected = _single_action(target, rng)
        elif subtype == "alias":
            spoken_targets[target["name"]] = target["aliases"][0]
            expected = _single_action(target, rng)
        else:
            spoken_targets[target["name"]] = target["name"].split()[-2] + " " + target["name"].split()[-1]
            expected = _single_action(target, rng)
    else:
        expected = _single_action(target, rng)

    return {
        "candidate_id": f"candidate_{index:06d}",
        "seed": seed,
        "category": category,
        "subcategory": subtype,
        "home": home,
        "expected": expected,
        "target_names": [
            call["arguments"]["name"] for call in expected.get("calls", []) if "name" in call["arguments"]
        ],
        "spoken_targets": spoken_targets,
        "excluded_names": excluded,
        "contrastive_group": None,
        "request_hint": request_hint,
        "stt_corruption": stt_corruption,
        "utterance": None,
    }


def build_specs(count: int, *, seed: int = 42) -> list[dict[str, Any]]:
    """Create deterministic synthetic homes and authoritative behavior labels."""
    if count <= 0 or count % 100:
        raise ValueError("count must be a positive multiple of 100")
    categories = [
        category
        for category, weight in CATEGORY_WEIGHTS.items()
        for _ in range(count * weight // 100)
    ]
    random.Random(seed).shuffle(categories)
    specs = [_spec(category, index, seed) for index, category in enumerate(categories)]
    direct_specs = [spec for spec in specs if spec["category"] == "clean_direct"]
    status_specs = [spec for spec in specs if spec["category"] == "status"]
    ambiguity_specs = [spec for spec in specs if spec["category"] == "ambiguity"]
    trio_count = min(
        len(direct_specs),
        len(status_specs),
        len(ambiguity_specs),
        count // 30,
    )
    grouped_ids: set[str] = set()
    for group_index in range(trio_count):
        action_spec, status_spec, ambiguity_spec = (
            direct_specs[group_index],
            status_specs[group_index],
            ambiguity_specs[group_index],
        )
        entities = deepcopy(action_spec["home"]["entities"][:6])
        sayso_area = action_spec["home"]["sayso_entity_area"]
        target = entities[0]
        kind = target["kind"]
        noun = _KIND_TO_GENERIC_NOUN[kind]
        group_rng = random.Random(seed + group_index)
        home_stub = {"sayso_entity_area": sayso_area, "entities": entities}
        in_sayso_area = _entities_of_kind_in_area(home_stub, kind, sayso_area)
        if len(in_sayso_area) == 1:
            template = in_sayso_area[0]
            duplicate = deepcopy(template)
            duplicate["name"] = f"Second {template['name']}"
            duplicate["entity_id"] += "_second"
            duplicate["aliases"] = [noun]
            entities.append(duplicate)
        elif len(in_sayso_area) >= 2:
            for entity in in_sayso_area:
                if noun not in entity["aliases"]:
                    entity["aliases"].append(noun)
        ambiguity_home = {"sayso_entity_area": sayso_area, "entities": entities}
        ambiguity_expected = _expected_generic_in_sayso_area(
            ambiguity_home,
            kind,
            group_rng,
            turn_on=kind != "fan",
        )
        for spec in (action_spec, status_spec, ambiguity_spec):
            spec["home"]["entities"] = deepcopy(entities)
            spec["home"]["sayso_entity_area"] = sayso_area
            spec["excluded_names"] = []
            spec["spoken_targets"] = {}
            spec["contrastive_group"] = f"contrast_{group_index:06d}"
            grouped_ids.add(spec["candidate_id"])
        action_spec["subcategory"] = "action_contrast"
        action_spec["expected"] = {
            "kind": "action",
            "calls": [_control_call(target, True, group_rng)],
        }
        action_spec["target_names"] = [target["name"]]
        action_spec["request_hint"] = ""
        status_spec["subcategory"] = "status_contrast"
        status_spec["expected"] = {
            "kind": "status",
            "calls": [_status_call(target)],
            "state": target["state"],
        }
        status_spec["target_names"] = [target["name"]]
        status_spec["request_hint"] = ""
        ambiguity_spec["subcategory"] = "ambiguity_contrast"
        ambiguity_spec["expected"] = ambiguity_expected
        ambiguity_spec["target_names"] = [
            call["arguments"]["name"]
            for call in ambiguity_expected.get("calls", [])
            if "name" in call.get("arguments", {})
        ]
        ambiguity_spec["request_hint"] = _generic_no_area_hint(kind)
    hard = [spec for spec in specs if spec["candidate_id"] in grouped_ids]
    return [spec for spec in hard if spec["candidate_id"] in grouped_ids] + [
        spec for spec in specs if spec["candidate_id"] not in grouped_ids
    ]


def validate_spec(spec: dict[str, Any]) -> str | None:
    """Validate authoritative behavior against its synthetic HA environment and v2."""
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    entities = {entity["name"]: entity for entity in spec["home"]["entities"]}
    schemas = tool_schema_map(v2_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    # Per-script tools are named after the script and are not in the pinned catalog.
    script_names = {
        script_tool_name(entity)
        for entity in spec["home"]["entities"]
        if entity.get("domain") == "script"
    }
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
        if name in script_names:
            if arguments:
                return "script_tool_takes_no_arguments"
            continue
        reason = validate_tool_arguments(name, arguments, schemas)
        if reason:
            return reason
        target = arguments.get("name")
        if target is not None and target not in entities:
            return "unknown_canonical_entity"
        if target in excluded:
            return "excluded_entity_called"
    for canonical, spoken in (spec.get("spoken_targets") or {}).items():
        if canonical not in entities or not spoken.strip():
            return "invalid_spoken_target"
    return None

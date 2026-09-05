#!/usr/bin/env python3
"""Build an eval-directed, label-first SaySo synthetic training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.schema import tool_schema_map, v1_openai_tools, validate_tool_arguments  # noqa: E402

DEFAULT_TRAIN_COUNT = 10_000

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

_BANNED_UTTERANCE = re.compile(
    r"<tool_call>|evals/cases/|tool_call_start",
    re.I,
)
_CLEAN_DIRECT_START = re.compile(
    r"^(turn|set|open|close|lock|unlock|switch)\b",
    re.I,
)

_FRAMING_PREFIXES = ("", "Please", "Could you please", "Hey, could you please", "Uh", "Okay, can you", "Would you please")
_FRAMING_SUFFIXES = ("", "please.", "for me?", "right now?", "thanks.")
_UNNATURAL_FRAMING = re.compile(
    r"\bplease\s+what\b|\b(?:could|can|would)\s+you\s+what\b",
    re.I,
)

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

_CONVERSATIONAL_TEMPLATES = (
    "Hey, when you get a chance, {action}.",
    "Could you {action} for me?",
    "Uh, I was wondering if you could {action}.",
    "Before I forget, {action}.",
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
    """Validate authoritative behavior against its synthetic HA environment and v1."""
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    entities = {entity["name"]: entity for entity in spec["home"]["entities"]}
    schemas = tool_schema_map(v1_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
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


def _system_prompt(home: dict[str, Any]) -> str:
    context = [
        {
            "name": entity["name"],
            "aliases": entity["aliases"],
            "domain": entity["domain"],
            "device_class": entity["device_class"],
            "area": entity["area"],
            "floor": entity["floor"],
            "state": entity["state"],
            "capabilities": entity["capabilities"],
        }
        for entity in home["entities"]
    ]
    sayso_area = home.get("sayso_entity_area", "")
    return (
        "You are SaySo, a concise Home Assistant conversation agent. Use only the supplied "
        "Home Assistant tools and preserve canonical entity names exactly. "
        f"This SaySo conversation entity area is {sayso_area!r}. "
        "Current exposed context: "
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )


def _call_id(candidate_id: str, index: int, call: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}:{index}:{json.dumps(call, sort_keys=True)}".encode()
    ).hexdigest()[:12]
    return f"call_{digest}"


def _final_text(spec: dict[str, Any]) -> str:
    expected = spec["expected"]
    if expected["kind"] == "status":
        return f"{spec['target_names'][0]} is {expected['state']}."
    if expected["kind"] == "action":
        return "Done."
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "this area")
        device_type = unavailable.get("type", "devices")
        return f"The {area} has no {device_type} available."
    return {
        "clarify": "Which device did you mean?",
        "unsupported": "I can't do that with the available Home Assistant tools.",
        "refuse": "I can't help with that request.",
    }[expected["response"]]


def render_example(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a validated, verbalized specification as canonical SaySo JSONL data."""
    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    calls = spec["expected"].get("calls") or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(spec["home"]), "train_on_turn": False},
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
    ]
    if calls:
        rendered_calls = []
        call_ids = []
        for index, call in enumerate(calls):
            call_id = _call_id(spec["candidate_id"], index, call)
            call_ids.append(call_id)
            rendered_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": True,
                "tool_calls": rendered_calls,
            }
        )
        for call_id, call in zip(call_ids, calls):
            result: dict[str, Any] = {"result": "Success"}
            if call["name"] == "GetLiveContext":
                result = {
                    "entities": [
                        {"name": spec["target_names"][0], "state": spec["expected"]["state"]}
                    ]
                }
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "train_on_turn": False,
                    "tool_call_id": call_id,
                }
            )
    messages.append(
        {"role": "assistant", "content": _final_text(spec), "train_on_turn": True}
    )
    family = spec["contrastive_group"] or spec["candidate_id"]
    metadata = {
        "candidate_id": spec["candidate_id"],
        "template_family": spec["contrastive_group"] or spec["category"],
        "phrasing_family": spec["contrastive_group"] or spec["subcategory"],
        "seed": family,
        "generation_seed": spec["seed"],
        "category": spec["category"],
        "subcategory": spec["subcategory"],
        "home_id": spec["home"]["home_id"],
        "contrastive_group": spec["contrastive_group"],
    }
    if "quality" in spec:
        metadata["quality"] = spec["quality"]
    return {"messages": messages, "tools": v1_openai_tools(), "metadata": metadata}


def request_seed(spec: dict[str, Any]) -> str:
    """Derive a compact semantic seed from authoritative labels, never model output."""
    expected = spec["expected"]
    if expected["kind"] == "no_action":
        return spec["request_hint"]
    targets = [spec["spoken_targets"].get(name, name) for name in spec["target_names"]]
    if expected["kind"] == "status":
        return f"what is the status of {targets[0]}"
    phrases: list[str] = []
    for target, call in zip(targets, expected["calls"]):
        name, arguments = call["name"], call["arguments"]
        device_class = set(arguments.get("device_class") or [])
        if name == "HassTurnOn":
            verb = "lock" if "door" in device_class else "open" if device_class else "turn on"
            phrases.append(f"{verb} {target}")
        elif name == "HassTurnOff":
            verb = "unlock" if "door" in device_class else "close" if device_class else "turn off"
            phrases.append(f"{verb} {target}")
        elif name == "HassLightSet":
            if "brightness" in arguments:
                phrases.append(f"set {target} brightness to {arguments['brightness']} percent")
            else:
                phrases.append(f"set {target} color to {arguments['color']}")
        elif name == "HassFanSetSpeed":
            phrases.append(f"set {target} speed to {arguments['percentage']} percent")
    seed = " and ".join(phrases)
    if spec["excluded_names"]:
        seed += ", but leave " + " and ".join(spec["excluded_names"]) + " alone"
    return seed


def expand_utterance(spec: dict[str, Any]) -> str:
    """Render a deterministic utterance from authoritative labels (recipe-first)."""
    category = spec["category"]
    expected = spec["expected"]
    if expected["kind"] == "no_action":
        return spec["request_hint"]
    if category == "ambiguity" and spec.get("request_hint"):
        return spec["request_hint"]
    targets = [spec["spoken_targets"].get(name, name) for name in spec["target_names"]]
    if expected["kind"] == "status":
        return f"what is the status of {targets[0]}"
    if category == "conversational":
        action_seed = request_seed(spec)
        template = _CONVERSATIONAL_TEMPLATES[
            hash(spec["candidate_id"]) % len(_CONVERSATIONAL_TEMPLATES)
        ]
        return template.format(action=action_seed)
    if category == "clean_direct":
        target, call = targets[0], expected["calls"][0]
        device_class = set(call["arguments"].get("device_class") or [])
        if call["name"] == "HassTurnOn":
            if "door" in device_class:
                return f"Lock {target}"
            if device_class:
                return f"Open {target}"
            return f"Turn on {target}"
        if call["name"] == "HassTurnOff":
            if "garage" in device_class:
                return f"Close {target}"
            if "door" in device_class:
                return f"Unlock {target}"
            if device_class:
                return f"Close {target}"
            return f"Turn off {target}"
        if call["name"] == "HassLightSet":
            brightness = call["arguments"].get("brightness")
            return f"Set {target} to {brightness} percent"
        if call["name"] == "HassFanSetSpeed":
            return f"Set {target} speed to {call['arguments']['percentage']} percent"
    return request_seed(spec)


def _protected_slots(spec: dict[str, Any]) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for index, name in enumerate(spec["target_names"], start=1):
        slots.append((f"<TARGET_{index}>", spec["spoken_targets"].get(name, name)))
    for index, name in enumerate(spec["excluded_names"], start=1):
        slots.append((f"<EXCLUDED_{index}>", name))
    values: list[str] = []
    for call in spec["expected"].get("calls", []):
        for value in call["arguments"].values():
            if isinstance(value, (int, float)) and str(value) not in values:
                values.append(str(value))
    if spec["expected"]["kind"] == "no_action":
        values.extend(value for value in re.findall(r"\d+(?:\.\d+)?", spec["request_hint"]) if value not in values)
    for index, value in enumerate(values, start=1):
        slots.append((f"<VALUE_{index}>", value))
    return slots


def template_seed(spec: dict[str, Any]) -> str:
    """Replace authoritative identity/value slots before LLM verbalization."""
    seed = request_seed(spec)
    for placeholder, value in sorted(_protected_slots(spec), key=lambda item: -len(item[1])):
        seed = seed.replace(value, placeholder)
    return seed


def _expand_template(spec: dict[str, Any], utterance: str) -> str | None:
    expanded = utterance.strip()
    slots = _protected_slots(spec)
    known = {placeholder for placeholder, _value in slots}
    if set(re.findall(r"<(?:TARGET|EXCLUDED|VALUE)_\d+>", expanded)) - known:
        return None
    for placeholder, value in slots:
        if expanded.count(placeholder) != 1:
            return None
        expanded = expanded.replace(placeholder, value)
    if re.search(r"<(?:TARGET|EXCLUDED|VALUE)_\d+>", expanded):
        return None
    return expanded


def _framed_utterance(spec: dict[str, Any], framing: Any) -> str | None:
    if (
        not isinstance(framing, list)
        or len(framing) < 2
        or not all(isinstance(part, str) for part in framing)
    ):
        return None
    prefix, suffix = framing[0].strip(), framing[-1].strip()
    framing_text = f"{prefix} {suffix}".strip()
    if len(framing_text.split()) > 12 or re.search(r"\d", framing_text):
        return None
    if re.search(r"\b(?:not|don['’]?t|dont|instead|except|without|leave|ignore|cancel|stop)\b", framing_text, re.I):
        return None
    allowed_prefix = {
        "hey", "hi", "okay", "ok", "uh", "um", "could", "can", "would", "you", "please",
        "just", "kindly", "maybe", "when", "get", "a", "chance", "before", "i", "forget",
        "id", "d", "like", "to", "go", "ahead", "also", "so", "well", "alright",
    }
    allowed_suffix = {"for", "me", "please", "thanks", "thank", "you", "right", "now", "if", "can"}

    def safe_words(text: str, allowed: set[str]) -> str:
        end = 0
        for match in re.finditer(r"[A-Za-z]+", text):
            if match.group().casefold() not in allowed:
                break
            end = match.end()
        return text[:end].strip()

    prefix = safe_words(prefix, allowed_prefix)
    suffix_words = safe_words(suffix, allowed_suffix)
    punctuation = suffix[-1] if suffix and suffix[-1] in ".?!" else ""
    suffix = suffix_words + punctuation
    seed = request_seed(spec)
    utterance = f"{prefix} {seed}".strip()
    if suffix:
        utterance += suffix if suffix[0] in ".,?!" else f" {suffix}"
    if _UNNATURAL_FRAMING.search(utterance):
        return None
    return utterance[:1].upper() + utterance[1:]


def _verbalizer_prompt(specs: list[dict[str, Any]]) -> str:
    payload = [
        {
            "candidate_id": spec["candidate_id"],
            "category": spec["category"],
            "subcategory": spec["subcategory"],
            "template_seed": template_seed(spec),
            "contrastive_group": spec["contrastive_group"],
        }
        for spec in specs
    ]
    return (
        "Generate natural conversational framing for each authoritative template_seed without rewriting or "
        "answering it. Each framing is [prefix,suffix]; examples are [\"Hey, could you please\",\"for me?\"], "
        "[\"Uh\",\"please.\"], or [\"\",\"\"]. Do not put actions, device names, negation, exclusions, or "
        "numbers in framing. Use sparse framing for clean_direct, varied speech for conversational, and mild "
        "disfluency for stt_corrupted. Return JSON only as {\"framings\":[[\"prefix\",\"suffix\"],...]}, in "
        "exact input order with exactly one pair per input.\nITEMS:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def framing_response_format(count: int) -> dict[str, Any]:
    """Constrain llama.cpp to one safe framing pair per authoritative row."""
    framing = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "prefixItems": [
            {"type": "string", "enum": list(_FRAMING_PREFIXES)},
            {"type": "string", "enum": list(_FRAMING_SUFFIXES)},
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "framings": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": framing,
            }
        },
        "required": ["framings"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "framings", "strict": True, "schema": schema}}


def judge_response_format(count: int) -> dict[str, Any]:
    """Constrain llama.cpp to three 1-5 quality scores per row."""
    score = {"type": "string", "pattern": "^[1-5]{3}$"}
    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": score,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "scores", "strict": True, "schema": schema}}


def verbalize_batch(
    specs: list[dict[str, Any]],
    complete: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fill utterances without allowing the language model to alter labels."""
    response = complete(_verbalizer_prompt(specs))
    framings = response.get("framings") if isinstance(response, dict) else None
    if len(specs) == 1 and isinstance(framings, list) and framings:
        framings = framings[:1]
    if isinstance(framings, list) and len(framings) == len(specs):
        verbalized: list[dict[str, Any]] = []
        rejected: dict[str, str] = {}
        for original, framing in zip(specs, framings):
            utterance = _framed_utterance(original, framing)
            if not utterance:
                rejected[original["candidate_id"]] = "verbalizer_missing_item"
                continue
            spec = deepcopy(original)
            spec["utterance"] = utterance
            verbalized.append(spec)
        return verbalized, rejected
    utterances = response.get("utterances") if isinstance(response, dict) else None
    if isinstance(utterances, list) and len(utterances) == len(specs):
        verbalized: list[dict[str, Any]] = []
        rejected: dict[str, str] = {}
        for original, utterance in zip(specs, utterances):
            expanded = _expand_template(original, utterance) if isinstance(utterance, str) else None
            if not expanded:
                rejected[original["candidate_id"]] = "verbalizer_missing_item"
                continue
            spec = deepcopy(original)
            spec["utterance"] = expanded
            verbalized.append(spec)
        return verbalized, rejected
    items = response.get("items") if isinstance(response, dict) else None
    by_id: dict[str, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate_id = item.get("candidate_id")
            utterance = item.get("utterance")
            if isinstance(candidate_id, str) and isinstance(utterance, str) and utterance.strip():
                if candidate_id in by_id:
                    by_id.pop(candidate_id, None)
                    continue
                by_id[candidate_id] = utterance.strip()
    verbalized: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for original in specs:
        candidate_id = original["candidate_id"]
        if candidate_id not in by_id:
            rejected[candidate_id] = "verbalizer_missing_item"
            continue
        expanded = _expand_template(original, by_id[candidate_id])
        if not expanded:
            rejected[candidate_id] = "verbalizer_missing_item"
            continue
        spec = deepcopy(original)
        spec["utterance"] = expanded
        verbalized.append(spec)
    return verbalized, rejected


def verbalize_resilient(
    specs: list[dict[str, Any]],
    complete: Any,
    *,
    attempts: int = 3,
) -> list[dict[str, Any]]:
    """Retry missing rows, splitting malformed batches down to single requests."""
    completed: dict[str, dict[str, Any]] = {}
    remaining = list(specs)
    retry_count = max(attempts, 10) if len(specs) == 1 else attempts
    for attempt in range(retry_count):
        retrying_complete = lambda prompt, n=attempt: complete(
            prompt.replace("ITEMS:\n", f"RETRY_VARIANT:{n}\nITEMS:\n", 1)
        )
        try:
            verbalized, rejected = verbalize_batch(remaining, retrying_complete)
        except Exception:
            verbalized = []
            rejected = {spec["candidate_id"]: "verbalizer_completion_error" for spec in remaining}
        completed.update({spec["candidate_id"]: spec for spec in verbalized})
        remaining = [spec for spec in remaining if spec["candidate_id"] in rejected]
        if not remaining:
            break
    if remaining:
        if len(remaining) == 1:
            spec = deepcopy(remaining[0])
            spec["utterance"] = _framed_utterance(spec, ["", ""]) or request_seed(spec)
            completed[spec["candidate_id"]] = spec
        else:
            midpoint = len(remaining) // 2
            for spec in verbalize_resilient(remaining[:midpoint], complete, attempts=attempts):
                completed[spec["candidate_id"]] = spec
            for spec in verbalize_resilient(remaining[midpoint:], complete, attempts=attempts):
                completed[spec["candidate_id"]] = spec
    return [completed[spec["candidate_id"]] for spec in specs]


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")))


def validate_utterance(spec: dict[str, Any]) -> str | None:
    """Apply deterministic label/context checks before the independent judge."""
    reason = validate_spec(spec)
    if reason:
        return reason
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not (2 <= len(utterance.split()) <= 60):
        return "invalid_utterance_length"
    if _UNNATURAL_FRAMING.search(utterance):
        return "unnatural_framing"
    if _BANNED_UTTERANCE.search(utterance):
        return "banned_content"
    text = _normalized(utterance)
    expected = spec["expected"]
    if spec["category"] == "conversational" and _CLEAN_DIRECT_START.match(utterance.strip()):
        return "conversational_not_voice"
    if expected["kind"] == "no_action":
        hint_terms = set(_normalized(spec["request_hint"]).split()) - {"the", "to", "a", "an"}
        if not hint_terms.intersection(text.split()):
            return "no_action_intent_missing"
        if "thermostat" in text and spec["category"] == "unsupported_no_action":
            return "banned_thermostat"
        return None

    if spec["category"] == "ambiguity" and spec.get("request_hint"):
        hint_terms = set(_normalized(spec["request_hint"]).split()) - {"the", "to", "a", "an"}
        if not hint_terms.intersection(text.split()):
            return "missing_expected_target"
        if expected["kind"] in {"action", "no_action"}:
            return None

    for canonical in spec["target_names"]:
        spoken = spec["spoken_targets"].get(canonical, canonical)
        if _normalized(spoken) not in text:
            return "missing_expected_target"

    if expected["kind"] == "status":
        if not any(cue in text for cue in ("status", "is ", "are ", "what ", "check", "doing")):
            return "status_not_query"
        if any(cue in text for cue in ("turn on", "turn off", "switch on", "switch off", "set ", "open ", "close ")):
            return "status_not_query"
        if any(call["name"] != "GetLiveContext" for call in expected.get("calls", [])):
            return "status_wrong_tool"
        return None

    action_cues = {
        "HassTurnOn": ("turn on", "switch on", "activate", "start", "open", "lock"),
        "HassTurnOff": ("turn off", "switch off", "deactivate", "stop", "close", "shut", "unlock"),
        "HassLightSet": ("brightness", "percent", "color", "dim", "bright"),
        "HassFanSetSpeed": ("speed", "percent", "faster", "slower"),
    }
    for call in expected["calls"]:
        if not any(cue in text for cue in action_cues.get(call["name"], ())):
            return "action_intent_missing"
    if spec["excluded_names"]:
        if any(_normalized(name) not in text for name in spec["excluded_names"]):
            return "missing_exclusion"
        if not any(cue in text for cue in ("leave", "except", "not", "dont", "alone", "exclude")):
            return "missing_exclusion"
    return None


def _judge_prompt(specs: list[dict[str, Any]]) -> str:
    payload = [
        {
            "candidate_id": spec["candidate_id"],
            "utterance": spec["utterance"],
            "authoritative_seed": request_seed(spec),
            "category": spec["category"],
            "subcategory": spec["subcategory"],
            "expected": spec["expected"],
            "excluded_names": spec["excluded_names"],
        }
        for spec in specs
    ]
    return (
        f"Independently judge all {len(specs)} utterance/authoritative-seed pairs. "
        "Score correctness, then clarity, then naturalness. Each digit is 1-5. "
        "correctness: 5 matches action vs status vs no-action, entities, exclusions, and exact_call_count; "
        "1 contradicts the label. "
        "clarity: 5 a listener knows exactly what to do; 1 garbled or ambiguous. "
        "naturalness: 5 a person would say this to a home voice assistant; 4 a normal terse command; "
        "3 robotic template-speak; 1-2 ungrammatical, including stacked framing like "
        "'Could you please what is the status...'. "
        "Do not default to 555. Use the full 1-5 range. Direct commands may still be 5/5/4. "
        f"Return JSON only with exactly {len(specs)} ordered score strings: "
        "{\"scores\":[\"543\"]}. No explanation.\nITEMS:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def judge_batch(
    specs: list[dict[str, Any]],
    complete: Any,
    *,
    generator_model: str,
    judge_model: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Use a distinct model to score label agreement and language quality."""
    if generator_model.strip().casefold() == judge_model.strip().casefold():
        raise ValueError("judge model must differ from generator model")
    eligible: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for spec in specs:
        reason = validate_utterance(spec)
        if reason:
            rejected[spec["candidate_id"]] = reason
        else:
            eligible.append(spec)
    if not eligible:
        return [], rejected
    response = complete(_judge_prompt(eligible))
    compact_scores = response.get("scores") if isinstance(response, dict) else None
    if len(eligible) == 1 and isinstance(compact_scores, dict):
        compact_scores = [compact_scores]
    if isinstance(compact_scores, list) and len(compact_scores) == len(eligible):
        difficulty = {
            "clean_direct": 2,
            "conversational": 3,
            "multi_action_exclusion": 5,
            "stt_corrupted": 5,
            "status": 4,
            "ambiguity": 5,
            "unsupported_no_action": 5,
            "entity_identity": 4,
        }
        accepted: list[dict[str, Any]] = []
        for spec, score in zip(eligible, compact_scores):
            if isinstance(score, str) and re.fullmatch(r"[1-5]{3}", score):
                correctness, clarity, naturalness = map(int, score)
            elif isinstance(score, dict) and all(
                isinstance(score.get(key), int) and 1 <= score[key] <= 5
                for key in ("correctness", "clarity", "naturalness")
            ):
                correctness, clarity, naturalness = (
                    score["correctness"], score["clarity"], score["naturalness"]
                )
            elif isinstance(score, list) and len(score) == 3 and all(
                isinstance(value, int) and 1 <= value <= 5 for value in score
            ):
                correctness, clarity, naturalness = score
            else:
                rejected[spec["candidate_id"]] = "judge_invalid_item"
                continue
            if min(correctness, clarity, naturalness) < 4:
                rejected[spec["candidate_id"]] = "judge_below_threshold"
                continue
            judged = deepcopy(spec)
            judged["quality"] = {
                "correctness": correctness,
                "clarity": clarity,
                "naturalness": naturalness,
                "difficulty": difficulty[spec["category"]],
                "semantic_key": request_seed(spec),
            }
            accepted.append(judged)
        return accepted, rejected
    items = response.get("items") if isinstance(response, dict) else None
    by_id = {
        item.get("candidate_id"): item
        for item in items or []
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    accepted: list[dict[str, Any]] = []
    for spec in eligible:
        candidate_id = spec["candidate_id"]
        item = by_id.get(candidate_id)
        if item is None:
            rejected[candidate_id] = "judge_missing_item"
            continue
        scores = {key: item.get(key) for key in ("correctness", "clarity", "naturalness", "difficulty")}
        semantic_key = item.get("semantic_key")
        if (
            not all(isinstance(value, int) and 1 <= value <= 5 for value in scores.values())
            or not isinstance(semantic_key, str)
            or not semantic_key.strip()
            or not isinstance(item.get("accept"), bool)
        ):
            rejected[candidate_id] = "judge_invalid_item"
            continue
        if not item["accept"] or min(scores["correctness"], scores["clarity"], scores["naturalness"]) < 4:
            rejected[candidate_id] = "judge_below_threshold"
            continue
        judged = deepcopy(spec)
        judged["quality"] = {**scores, "semantic_key": semantic_key.strip()}
        accepted.append(judged)
    return accepted, rejected


def judge_resilient(
    specs: list[dict[str, Any]],
    complete: Any,
    *,
    generator_model: str,
    judge_model: str,
    attempts: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Retry malformed judge output and split batches without retrying real quality failures."""
    accepted: dict[str, dict[str, Any]] = {}
    final_rejected: dict[str, str] = {}
    remaining = list(specs)
    retry_count = max(attempts, 3) if len(specs) == 1 else attempts
    for _attempt in range(retry_count):
        try:
            passed, rejected = judge_batch(
                remaining,
                complete,
                generator_model=generator_model,
                judge_model=judge_model,
            )
        except Exception:
            passed = []
            rejected = {spec["candidate_id"]: "judge_invalid_item" for spec in remaining}
        accepted.update({spec["candidate_id"]: spec for spec in passed})
        retry_ids = {
            candidate_id
            for candidate_id, reason in rejected.items()
            if reason in {"judge_missing_item", "judge_invalid_item"}
        }
        final_rejected.update(
            (candidate_id, reason)
            for candidate_id, reason in rejected.items()
            if candidate_id not in retry_ids
        )
        remaining = [spec for spec in remaining if spec["candidate_id"] in retry_ids]
        if not remaining:
            break
    if remaining and len(remaining) > 1:
        midpoint = len(remaining) // 2
        for half in (remaining[:midpoint], remaining[midpoint:]):
            passed, rejected = judge_resilient(
                half,
                complete,
                generator_model=generator_model,
                judge_model=judge_model,
                attempts=attempts,
            )
            accepted.update({spec["candidate_id"]: spec for spec in passed})
            final_rejected.update(rejected)
        remaining = []
    for spec in remaining:
        final_rejected[spec["candidate_id"]] = "judge_invalid_item"
    return [accepted[spec["candidate_id"]] for spec in specs if spec["candidate_id"] in accepted], final_rejected


def _behavior_key(spec: dict[str, Any]) -> str:
    behavior = {
        "expected": spec["expected"],
        "excluded": spec["excluded_names"],
        "home_id": spec["home"]["home_id"],
    }
    return json.dumps(behavior, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ranked(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantic_counts = Counter(_normalized(spec["quality"]["semantic_key"]) for spec in specs)
    category_counts = Counter(spec["category"] for spec in specs)
    ranked: list[dict[str, Any]] = []
    for original in specs:
        spec = deepcopy(original)
        quality = spec["quality"]
        semantic = _normalized(quality["semantic_key"])
        uniqueness = 5 / semantic_counts[semantic]
        coverage = 5 * len(specs) / (len(CATEGORY_WEIGHTS) * category_counts[spec["category"]])
        quality["rank_score"] = round(
            quality["correctness"] * 8
            + quality["clarity"] * 5
            + quality["naturalness"] * 4
            + quality["difficulty"] * 3
            + min(uniqueness, 5) * 2
            + min(coverage, 5),
            4,
        )
        ranked.append(spec)
    return sorted(ranked, key=lambda spec: (-spec["quality"]["rank_score"], spec["candidate_id"]))


def curate(
    specs: list[dict[str, Any]],
    *,
    min_count: int = DEFAULT_TRAIN_COUNT,
    max_count: int = DEFAULT_TRAIN_COUNT,
    excluded_utterances: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hard-filter duplicates, then retain the highest-quality balanced rows."""
    if min_count <= 0 or max_count < min_count:
        raise ValueError("invalid curation bounds")
    drops: Counter[str] = Counter()
    exact_seen: set[tuple[str, str]] = set()
    semantic_seen: set[tuple[str, str]] = set()
    excluded = {_normalized(text) for text in excluded_utterances or set()}
    unique: list[dict[str, Any]] = []
    for spec in _ranked(specs):
        if _normalized(spec["utterance"]) in excluded:
            drops["heldout_overlap"] += 1
            continue
        behavior = _behavior_key(spec)
        exact = (_normalized(spec["utterance"]), behavior)
        if exact in exact_seen:
            drops["exact_duplicate"] += 1
            continue
        exact_seen.add(exact)
        semantic = (_normalized(spec["quality"]["semantic_key"]), behavior)
        if semantic in semantic_seen:
            drops["semantic_duplicate"] += 1
            continue
        semantic_seen.add(semantic)
        unique.append(spec)
    if len(unique) < min_count:
        raise ValueError(
            f"quality floor left {len(unique)} rows, below required minimum {min_count}; thresholds unchanged"
        )
    if min_count >= 100:
        available = Counter(spec["category"] for spec in unique)
        missing = {
            category: math.ceil(min_count * weight / 200)
            for category, weight in CATEGORY_WEIGHTS.items()
            if available[category] < math.ceil(min_count * weight / 200)
        }
        if missing:
            raise ValueError(f"quality coverage floor not met: {missing}; thresholds unchanged")
    if len(unique) <= max_count:
        return unique, dict(drops)

    quotas = {category: max_count * weight // 100 for category, weight in CATEGORY_WEIGHTS.items()}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category, quota in quotas.items():
        rows = [spec for spec in unique if spec["category"] == category][:quota]
        selected.extend(rows)
        selected_ids.update(spec["candidate_id"] for spec in rows)
    direct_cap = max_count * CATEGORY_WEIGHTS["clean_direct"] // 100
    for spec in unique:
        if len(selected) >= max_count:
            break
        if spec["candidate_id"] in selected_ids:
            continue
        if spec["category"] == "clean_direct" and sum(
            row["category"] == "clean_direct" for row in selected
        ) >= direct_cap:
            continue
        selected.append(spec)
        selected_ids.add(spec["candidate_id"])
    if len(selected) < min_count:
        raise ValueError(
            f"coverage constraints left {len(selected)} rows, below required minimum {min_count}; thresholds unchanged"
        )
    drops["ranked_below_cut"] += len(unique) - len(selected)
    return sorted(
        selected, key=lambda spec: (-spec["quality"]["rank_score"], spec["candidate_id"])
    ), dict(drops)


def _json_content(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model response did not contain a JSON object")
    parsed, _end = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    return parsed


def openai_complete(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 600,
    post: Any = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat endpoint and parse its JSON response."""
    if post is None:
        import requests

        post = requests.post
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format or {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("model response content must be text")
            return _json_content(content)
        except Exception as error:  # endpoint and model formatting failures share retry policy
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI-compatible completion failed after 3 attempts: {last_error}")


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ValueError(f"{path}:{line_no} lacks candidate_id")
            rows[candidate_id] = row
    return rows


def load_user_utterances(path: Path) -> set[str]:
    """Read user prompts from canonical SaySo JSONL for leakage exclusion."""
    utterances: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for message in json.loads(line).get("messages", []):
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    utterances.add(content)
                elif isinstance(content, list):
                    text = " ".join(
                        part["text"] for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
                    if text:
                        utterances.add(text)
    return utterances


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["category"] for row in rows).items()))


def _audit_selected(rows: list[dict[str, Any]]) -> dict[str, int]:
    schemas = tool_schema_map(v1_openai_tools())
    calls = [call for row in rows for call in row["expected"].get("calls", [])]
    exact = [(_normalized(row["utterance"]), _behavior_key(row)) for row in rows]
    semantic = [(_normalized(row["quality"]["semantic_key"]), _behavior_key(row)) for row in rows]
    return {
        "deterministically_valid_rows": sum(validate_utterance(row) is None for row in rows),
        "schema_valid_tool_calls": sum(
            validate_tool_arguments(call["name"], call["arguments"], schemas) is None for call in calls
        ),
        "tool_calls": len(calls),
        "no_action_rows": sum(not row["expected"].get("calls") for row in rows),
        "status_rows": sum(row["expected"]["kind"] == "status" for row in rows),
        "multi_call_rows": sum(len(row["expected"].get("calls", [])) > 1 for row in rows),
        "exclusion_rows": sum(bool(row["excluded_names"]) for row in rows),
        "stt_resolution_rows": sum(row["category"] == "stt_corrupted" for row in rows),
        "apostrophe_argument_rows": sum(
            any("'" in call["arguments"].get("name", "") for call in row["expected"].get("calls", []))
            for row in rows
        ),
        "contrastive_rows": sum(bool(row["contrastive_group"]) for row in rows),
        "contrastive_groups": len({row["contrastive_group"] for row in rows if row["contrastive_group"]}),
        "excluded_entity_call_violations": sum(
            call["arguments"].get("name") in set(row["excluded_names"])
            for row in rows
            for call in row["expected"].get("calls", [])
        ),
        "exact_duplicate_keys": len(exact) - len(set(exact)),
        "semantic_duplicate_keys": len(semantic) - len(set(semantic)),
    }


def _run_batches(rows: list[dict[str, Any]], batch_size: int, workers: int, process: Any):
    batches = [rows[offset : offset + batch_size] for offset in range(0, len(rows), batch_size)]
    if workers == 1:
        for batch in batches:
            yield process(batch)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(process, batches)


def run_pipeline(
    *,
    out_dir: Path,
    count: int,
    seed: int,
    batch_size: int,
    generator_complete: Any = None,
    judge_complete: Any = None,
    generator_model: str,
    judge_model: str,
    min_count: int = DEFAULT_TRAIN_COUNT,
    max_count: int = DEFAULT_TRAIN_COUNT,
    stage: str = "all",
    excluded_utterances: set[str] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run resumable generation, judging, and curation stages."""
    if stage not in {"all", "generate", "judge", "curate"}:
        raise ValueError("unknown stage")
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    if generator_model.strip().casefold() == judge_model.strip().casefold():
        raise ValueError("judge model must differ from generator model")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / f"sayso_candidates_{count}.jsonl"
    judged_path = out_dir / f"sayso_judged_{count}.jsonl"
    rejected_path = out_dir / f"sayso_rejected_{count}.jsonl"
    report_path = out_dir / f"sayso_build_report_{count}.json"
    if report_path.exists() and stage in {"all", "curate"}:
        return json.loads(report_path.read_text(encoding="utf-8"))

    specs = build_specs(count, seed=seed)
    candidates = _read_jsonl(candidate_path)
    for row in candidates.values():
        if row.get("seed") != seed:
            raise ValueError("candidate checkpoint seed mismatch")
        if str(row.get("generator_model", "")).strip().casefold() == judge_model.strip().casefold():
            raise ValueError("judge model must differ from every candidate generator model")
    if stage in {"all", "generate"}:
        if generator_complete is None:
            raise ValueError("generator completion is required")
        missing = [spec for spec in specs if spec["candidate_id"] not in candidates]
        process = lambda batch: verbalize_resilient(batch, generator_complete)
        for completed in _run_batches(missing, batch_size, workers, process):
            for row in completed:
                row["generator_model"] = generator_model
            _append_jsonl(candidate_path, completed)
            candidates.update({row["candidate_id"]: row for row in completed})
            print(f"generated {len(candidates)}/{count}", flush=True)
    if len(candidates) != count:
        raise ValueError(f"candidate pool incomplete: {len(candidates)}/{count}")
    if stage == "generate":
        return {
            "candidate_count": len(candidates),
            "candidate_categories": _counts(list(candidates.values())),
            "generator_models": sorted({row["generator_model"] for row in candidates.values()}),
        }

    judged = _read_jsonl(judged_path)
    rejection_rows = _read_jsonl(rejected_path)
    for row in judged.values():
        if row.get("judge_model") != judge_model:
            raise ValueError("judge checkpoint model mismatch")
    if stage in {"all", "judge"}:
        if judge_complete is None:
            raise ValueError("judge completion is required")
        pending = [
            candidates[spec["candidate_id"]]
            for spec in specs
            if spec["candidate_id"] not in judged and spec["candidate_id"] not in rejection_rows
        ]
        process = lambda batch: judge_resilient(
                batch,
                judge_complete,
                generator_model=generator_model,
                judge_model=judge_model,
            )
        for accepted, rejected in _run_batches(pending, batch_size, workers, process):
            for row in accepted:
                row["judge_model"] = judge_model
            rejected_data = [
                {"candidate_id": candidate_id, "reason": reason, "stage": "judge"}
                for candidate_id, reason in rejected.items()
            ]
            _append_jsonl(judged_path, accepted)
            _append_jsonl(rejected_path, rejected_data)
            judged.update({row["candidate_id"]: row for row in accepted})
            rejection_rows.update({row["candidate_id"]: row for row in rejected_data})
            print(
                f"judged {len(judged) + len(rejection_rows)}/{count} "
                f"(accepted {len(judged)}, rejected {len(rejection_rows)})",
                flush=True,
            )
    if len(judged) + len(rejection_rows) != count:
        raise ValueError(
            f"judge pool incomplete: {len(judged) + len(rejection_rows)}/{count}"
        )
    if stage == "judge":
        return {
            "candidate_count": count,
            "judge_accepted": len(judged),
            "judge_rejected": len(rejection_rows),
        }

    selected, curation_drops = curate(
        list(judged.values()),
        min_count=min_count,
        max_count=max_count,
        excluded_utterances=excluded_utterances,
    )
    curated_path = out_dir / f"sayso_curated_{len(selected)}.jsonl"
    _write_jsonl(curated_path, [render_example(spec) for spec in selected])
    rejection_reasons = Counter(row["reason"] for row in rejection_rows.values())
    report = {
        "seed": seed,
        "generator_model": generator_model,
        "generator_models": sorted({row["generator_model"] for row in candidates.values()}),
        "judge_model": judge_model,
        "candidate_count": count,
        "candidate_categories": _counts(list(candidates.values())),
        "judge_accepted": len(judged),
        "judge_rejected": len(rejection_rows),
        "judge_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "curation_drops": dict(sorted(curation_drops.items())),
        "curated_count": len(selected),
        "curated_categories": _counts(selected),
        "quality_thresholds": {"correctness": 4, "clarity": 4, "naturalness": 4},
        "excluded_prompt_count": len(excluded_utterances or set()),
        "audit": _audit_selected(selected),
        "files": {
            "candidates": candidate_path.name,
            "candidates_sha256": _sha256(candidate_path),
            "curated": curated_path.name,
            "curated_sha256": _sha256(curated_path),
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def render_for_trl(example: dict[str, Any]) -> dict[str, Any]:
    """TRL view: dict tool arguments and plain-string message content."""
    from adapters.schema import extract_text_content, normalize_tool_arguments

    rendered = deepcopy(example)
    messages: list[dict[str, Any]] = []
    for message in rendered.get("messages") or []:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = extract_text_content(content)
        elif content is None and msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg["content"] = ""
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls: list[dict[str, Any]] = []
            for call in msg["tool_calls"]:
                rendered_call = dict(call)
                fn = dict(rendered_call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    parsed = normalize_tool_arguments(args)
                    if parsed is not None:
                        fn["arguments"] = parsed
                rendered_call["function"] = fn
                calls.append(rendered_call)
            msg["tool_calls"] = calls
        messages.append(msg)
    rendered["messages"] = messages
    return rendered


def _deterministic_train_utterance(spec: dict[str, Any], excluded: set[str]) -> str | None:
    """Return a deterministic utterance that avoids quality-eval prompt overlap."""
    base = expand_utterance(spec)
    if _normalized(base) not in excluded:
        return base
    prefixes = ("please ", "hey, ", "could you ", "okay, ")
    suffixes = (" please", " for me", " right now", " thanks")
    candidates = [f"{prefix}{base}" for prefix in prefixes] + [f"{base}{suffix}" for suffix in suffixes]
    for candidate in candidates:
        if _normalized(candidate) in excluded:
            continue
        trial = deepcopy(spec)
        trial["utterance"] = candidate
        if validate_utterance(trial) is None:
            return candidate
    return None


def build_deterministic_train_examples(
    count: int = DEFAULT_TRAIN_COUNT,
    *,
    seed: int = 20260904,
    excluded_utterances: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build label-first train rows via expand_utterance, excluding quality-eval prompts."""
    if count <= 0 or count % 100:
        raise ValueError("count must be a positive multiple of 100")
    excluded = {_normalized(text) for text in excluded_utterances or set()}
    quotas = {category: count * weight // 100 for category, weight in CATEGORY_WEIGHTS.items()}
    selected_by_category: dict[str, list[dict[str, Any]]] = {
        category: [] for category in quotas
    }
    pool_size = count
    seed_offset = 0
    while any(len(selected_by_category[category]) < quota for category, quota in quotas.items()):
        specs = build_specs(pool_size, seed=seed + seed_offset)
        for spec in specs:
            category = spec["category"]
            if len(selected_by_category[category]) >= quotas[category]:
                continue
            row = deepcopy(spec)
            utterance = _deterministic_train_utterance(row, excluded)
            if utterance is None:
                continue
            row["utterance"] = utterance
            if validate_utterance(row) is not None:
                continue
            selected_by_category[category].append(render_example(row))
        pool_size += 100
        seed_offset += 1
        if seed_offset > 50:
            raise ValueError(
                f"unable to collect balanced train rows without quality-eval overlap after {seed_offset} pools"
            )
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_WEIGHTS:
        rows.extend(selected_by_category[category][: quotas[category]])
    if len(rows) != count:
        raise ValueError(f"expected {count} train rows, built {len(rows)}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("legacy", "v3"), default="legacy")
    parser.add_argument("--stage", choices=("all", "generate", "judge", "curate"), default="all")
    parser.add_argument("--count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "synthetic_v2")
    parser.add_argument("--generator-url", default="http://192.168.1.140:8080/v1")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--judge-url", default="http://192.168.1.140:8080/v1")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--min-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--max-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--exclude-prompts", type=Path)
    parser.add_argument("--stt-rate", type=float, default=0.15)
    parser.add_argument("--paraphrase", action="store_true", default=False)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.pipeline == "v3":
        from generators.config import GeneratorConfig
        from generators.pipeline import run_generation, write_jsonl, write_manifest

        out_path = args.out_dir / "synthetic_v3_train.jsonl" if args.out_dir.is_dir() else args.out_dir
        config = GeneratorConfig(
            count=args.count,
            seed=args.seed,
            output_path=out_path,
            manifest_path=args.manifest or out_path.with_suffix(".manifest.json"),
            stt_noise_rate=args.stt_rate,
            paraphrase_enabled=args.paraphrase,
            token_budget=args.token_budget,
            exclude_prompts_path=args.exclude_prompts,
        )
        result = run_generation(config)
        write_jsonl(config.output_path, result["rows"])
        write_manifest(config.manifest_path, result["stats"])
        print(json.dumps(result["stats"], indent=2, default=str))
        return 0

    if not args.generator_model or not args.judge_model:
        parser.error("legacy pipeline requires --generator-model and --judge-model")

    generator_key = os.environ.get("SAYSO_GENERATOR_API_KEY", "")
    judge_key = os.environ.get("SAYSO_JUDGE_API_KEY", generator_key)
    generator_complete = None
    judge_complete = None
    if args.stage in {"all", "generate"}:
        generator_complete = lambda prompt: openai_complete(
            prompt,
            base_url=args.generator_url,
            model=args.generator_model,
            api_key=generator_key,
            temperature=0.7,
            max_tokens=max(256, len(json.loads(prompt.split("ITEMS:\n", 1)[1])) * 12),
            response_format=framing_response_format(len(json.loads(prompt.split("ITEMS:\n", 1)[1]))),
        )
    if args.stage in {"all", "judge"}:
        judge_complete = lambda prompt: openai_complete(
            prompt,
            base_url=args.judge_url,
            model=args.judge_model,
            api_key=judge_key,
            temperature=0.0,
            max_tokens=max(64, len(json.loads(prompt.split("ITEMS:\n", 1)[1])) * 8),
            response_format=judge_response_format(len(json.loads(prompt.split("ITEMS:\n", 1)[1]))),
        )
    report = run_pipeline(
        out_dir=args.out_dir,
        count=args.count,
        seed=args.seed,
        batch_size=args.batch_size,
        generator_complete=generator_complete,
        judge_complete=judge_complete,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        min_count=args.min_count,
        max_count=args.max_count,
        stage=args.stage,
        excluded_utterances=load_user_utterances(args.exclude_prompts) if args.exclude_prompts else None,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

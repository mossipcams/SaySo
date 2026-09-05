#!/usr/bin/env python3
"""Generate corrective SFT supplement and shadow eval gold rows for epoch-2 training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import (  # noqa: E402
    _control_call,
    _make_entity,
    _normalized,
    _status_call,
    expand_utterance,
    load_user_utterances,
    render_example,
    validate_spec,
    validate_utterance,
    write_jsonl,
)
from evals.recipe_lock import locked_specs, quality_eval_user_prompts  # noqa: E402

DEFAULT_CORRECTIVE_OUT = ROOT / "datasets" / "sayso_train_supplement.jsonl"
DEFAULT_SHADOW_OUT = ROOT / "datasets" / "sayso_shadow_eval.jsonl"
DEFAULT_HELDOUT = ROOT / "datasets" / "sayso_test_balanced.jsonl"
DEFAULT_BASE_TRAIN = ROOT / "datasets" / "sayso_train_first_10000.jsonl"

CORRECTIVE_TARGETS = {
    "light_fan_contrast": 200,
    "lock_polarity": 150,
    "apostrophe_names": 125,
    "multi_action": 100,
}
CORRECTIVE_MIN = 500
CORRECTIVE_MAX = 800
SHADOW_MIN = 100
SHADOW_MAX = 150
DEFAULT_SHADOW_COUNT = 125

_FRESH_AREAS = (
    "Den",
    "Sunroom",
    "Mudroom",
    "Loft",
    "Foyer",
    "Game Room",
    "Breakfast Nook",
    "Craft Room",
    "Reading Nook",
    "Utility Room",
    "Guest Suite",
    "Media Room",
)
_FRESH_APOSTROPHE_NAMES = (
    "Mary's Bedside Lamp",
    "Liam's Desk Lamp",
    "Ana's Sunroom Light",
    "O'Brien's Hall Light",
    "Girls' Playroom Light",
    "Boys' Bunk Light",
    "Chris's Patio Lamp",
    "Eva's Reading Light",
    "Nora's Craft Light",
    "Pat's Mudroom Lamp",
    "Kim's Loft Fan",
    "Alex's Foyer Light",
    "Rosa's Den Lamp",
    "Finn's Media Light",
    "Zoe's Craft Fan",
)
_FRESH_REGULAR_NAMES = (
    "Guest Suite Lamp",
    "Den West Fan",
    "Mudroom Entry Lock",
    "Sun Porch Blinds",
    "Loft Corner Outlet",
    "Game Room Ceiling Fan",
    "Breakfast Nook Lamp",
    "Media Room Side Blinds",
    "Reading Nook Lamp",
    "Utility Room Outlet",
    "Foyer Main Light",
    "Craft Room Desk Lamp",
)
_SHADOW_CONTRAST_AREAS = (
    "Annex",
    "Atrium",
    "Conservatory",
    "Solarium",
    "Studio",
    "Workshop",
    "Library",
    "Terrace",
    "Balcony",
    "Cellar",
    "Attic",
    "Porch",
)
_STT_ACTION_CUES = {
    "HassTurnOn": (
        "turn on",
        "trn on",
        "tern on",
        "switch on",
        "activate",
        "start",
        "open",
        "lock",
        "lok",
    ),
    "HassTurnOff": (
        "turn off",
        "turn of",
        "trn off",
        "trn of",
        "tern off",
        "tern of",
        "switch off",
        "deactivate",
        "stop",
        "close",
        "shut",
        "unlock",
        "unlok",
    ),
    "HassLightSet": ("brightness", "percent", "color", "dim", "bright", "prcent", "briteness"),
    "HassFanSetSpeed": ("speed", "percent", "faster", "slower", "prcent"),
}


def _validate_assigned_utterance(spec: dict[str, Any]) -> str | None:
    reason = validate_utterance(spec)
    if reason != "action_intent_missing" or spec["category"] != "stt_corrupted":
        return reason
    utterance = spec.get("utterance")
    if not isinstance(utterance, str):
        return reason
    text = _normalized(utterance)
    for call in spec["expected"]["calls"]:
        if not any(cue in text for cue in _STT_ACTION_CUES.get(call["name"], ())):
            return "action_intent_missing"
    return None


def _utterance_ambiguity(spec: dict[str, Any], *, attempt: int = 0) -> str:
    hint = spec["request_hint"]
    index = int(spec["candidate_id"].rsplit("_", 1)[-1])
    area = spec["home"]["sayso_entity_area"].casefold()
    prefixes = ("", "please ", "hey ", "okay ", "could you ", "uh ", "would you ", "can you ", "go ahead and ")
    suffixes = (
        "",
        " please",
        " now",
        f" in the {area}",
        f" in {area}",
        " right now",
        " for me",
        " when you get a chance",
        " if you can",
        " thanks",
        f" over in {area}",
        f" here in {area}",
        " real quick",
    )
    prefix = prefixes[(index * 17 + attempt * 7 + spec["seed"]) % len(prefixes)]
    suffix = suffixes[(index * 11 + attempt * 13 + spec["seed"] // 5) % len(suffixes)]
    return f"{prefix}{hint}{suffix}".strip()


_STT_SUBSTITUTIONS = (
    ("turn on", "trn on"),
    ("turn on", "tern on"),
    ("turn off", "turn of"),
    ("turn off", "trn off"),
    ("light", "lite"),
    ("fan", "van"),
    ("lock", "lok"),
    ("unlock", "unlok"),
    ("blinds", "blends"),
    ("percent", "prcent"),
    ("brightness", "briteness"),
)


def _recipe_lock_entity_names() -> set[str]:
    names: set[str] = set()
    for spec in locked_specs():
        for entity in spec["home"]["entities"]:
            names.add(entity["name"])
            names.update(entity.get("aliases") or [])
    return names


def _spec_shell(
    *,
    candidate_id: str,
    seed: int,
    category: str,
    subcategory: str,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str],
    spoken_targets: dict[str, str] | None = None,
    excluded_names: list[str] | None = None,
    request_hint: str = "",
    stt_corruption: str | None = None,
    dataset: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "category": category,
        "subcategory": subcategory,
        "home": home,
        "expected": expected,
        "target_names": target_names,
        "spoken_targets": spoken_targets or {},
        "excluded_names": excluded_names or [],
        "contrastive_group": None,
        "request_hint": request_hint,
        "stt_corruption": stt_corruption,
        "utterance": None,
        "dataset": dataset,
    }


def _fresh_home(index: int, rng: random.Random, *, sayso_entity_area: str | None = None) -> dict[str, Any]:
    area = sayso_entity_area or _FRESH_AREAS[index % len(_FRESH_AREAS)]
    entities = [
        _make_entity(
            name=f"{area} {_kind_noun(kind)}",
            kind=kind,
            area=area,
            rng=rng,
            aliases=[_kind_noun(kind)],
        )
        for kind in ("light", "fan", "switch", "lock")
    ]
    return {
        "home_id": f"corrective_home_{index:06d}",
        "sayso_entity_area": area,
        "entities": entities,
    }


def _kind_noun(kind: str) -> str:
    return {
        "light": "Main Lamp",
        "fan": "Ceiling Fan",
        "switch": "Outlet",
        "lock": "Door Lock",
        "blinds": "Blinds",
        "garage_door": "Garage Door",
    }[kind]


def _light_set_call(entity: dict[str, Any], brightness: int) -> dict[str, Any]:
    return {
        "name": "HassLightSet",
        "arguments": {"name": entity["name"], "domain": ["light"], "brightness": brightness},
    }


def _fan_speed_call(entity: dict[str, Any], percentage: int) -> dict[str, Any]:
    return {
        "name": "HassFanSetSpeed",
        "arguments": {"name": entity["name"], "domain": ["fan"], "percentage": percentage},
    }


def _lock_call(entity: dict[str, Any], lock: bool) -> dict[str, Any]:
    return _control_call(entity, lock, random.Random(0))


def _apply_stt(
    text: str,
    rng: random.Random,
    *,
    min_edits: int = 1,
    max_edits: int = 2,
) -> tuple[str, str]:
    lowered = text.casefold()
    applied: list[str] = []
    for source, replacement in _STT_SUBSTITUTIONS:
        if source in lowered and rng.random() < 0.45:
            lowered = lowered.replace(source, replacement, 1)
            applied.append(f"{source}->{replacement}")
            if len(applied) >= max_edits:
                break
    if len(applied) < min_edits:
        for source, replacement in _STT_SUBSTITUTIONS:
            if source in lowered:
                lowered = lowered.replace(source, replacement, 1)
                applied.append(f"{source}->{replacement}")
                if len(applied) >= min_edits:
                    break
    return lowered, "+".join(applied[:max_edits]) or "stt_mild"


def _utterance_light_fan(spec: dict[str, Any], rng: random.Random, *, attempt: int = 0) -> str:
    target = spec["target_names"][0]
    call = spec["expected"]["calls"][0]
    value = call["arguments"].get("brightness") or call["arguments"].get("percentage")
    templates = (
        "set {target} to {value} percent",
        "could you set {target} to {value} percent please",
        "please set {target} brightness to {value} percent",
        "make {target} {value} percent",
        "set {target} speed to {value} percent",
        "turn {target} to {value} percent",
        "uh set {target} to {value} percent",
        "would you set {target} to {value} percent",
        "set {target} at {value} percent",
        "bring {target} to {value} percent",
        "adjust {target} to {value} percent",
        "change {target} to {value} percent",
        "set the {target} to {value} percent",
        "please make {target} {value} percent",
        "can you set {target} to {value} percent",
        "go ahead and set {target} to {value} percent",
    )
    index = int(spec["candidate_id"].rsplit("_", 1)[-1])
    template = templates[(index * 17 + attempt * 3 + spec["seed"]) % len(templates)]
    return template.format(target=target, value=value)


def _utterance_lock_polarity(spec: dict[str, Any], rng: random.Random, *, attempt: int = 0) -> str:
    target = spec["target_names"][0]
    call = spec["expected"]["calls"][0]
    lock = call["name"] == "HassTurnOn"
    area = spec["home"]["entities"][0]["area"].casefold()
    index = int(spec["candidate_id"].rsplit("_", 1)[-1])
    if lock:
        cores = (
            f"lock {target}",
            f"lock the {target}",
            f"please lock {target}",
            f"could you lock {target}",
            f"go ahead and lock {target}",
            f"lock {target} in the {area}",
            f"please lock the {target} in the {area}",
            f"hey lock {target}",
            f"okay lock {target} now",
            f"would you lock {target} please",
            f"can you lock {target}",
            f"i need you to lock {target}",
            f"secure {target}",
            f"secure the {target}",
            f"make sure {target} is locked",
        )
    else:
        cores = (
            f"unlock {target}",
            f"unlock the {target}",
            f"please unlock {target}",
            f"could you unlock {target}",
            f"go ahead and unlock {target}",
            f"unlock {target} in the {area}",
            f"please unlock the {target} in the {area}",
            f"hey unlock {target}",
            f"okay unlock {target} now",
            f"would you unlock {target} please",
            f"can you unlock {target}",
            f"open up {target}",
            f"unlok {target}",
            f"unlok the {target}",
            f"release {target}",
        )
    prefixes = ("", "uh ", "so ", "well ", "maybe ", "just ", "real quick ")
    suffixes = ("", " please", " now", " thanks", " for me", " when you get a chance")
    core = cores[(index * 17 + attempt * 3 + spec["seed"]) % len(cores)]
    prefix = prefixes[(index * 11 + attempt * 5) % len(prefixes)]
    suffix = suffixes[(index * 13 + attempt * 7) % len(suffixes)]
    return f"{prefix}{core}{suffix}".strip()


def _utterance_apostrophe(spec: dict[str, Any], rng: random.Random, *, attempt: int = 0) -> str:
    target = spec["target_names"][0]
    call = spec["expected"]["calls"][0]
    device_class = set(call["arguments"].get("device_class") or [])
    index = int(spec["candidate_id"].rsplit("_", 1)[-1])
    turn_on = call["name"] == "HassTurnOn"
    if call["name"] == "HassTurnOn" and "door" in device_class:
        cores = (f"lock {target}", f"lock the {target}", f"please lock {target}")
    elif call["name"] == "HassTurnOff" and "door" in device_class:
        cores = (f"unlock {target}", f"unlock the {target}", f"please unlock {target}")
    elif turn_on:
        cores = (
            f"turn on {target}",
            f"turn on the {target}",
            f"please turn on {target}",
            f"could you turn on {target}",
            f"switch on {target}",
            f"hey turn on {target}",
            f"okay turn on {target}",
            f"activate {target}",
        )
    else:
        cores = (
            f"turn off {target}",
            f"turn off the {target}",
            f"please turn off {target}",
            f"could you turn off {target}",
            f"switch off {target}",
            f"hey turn off {target}",
            f"shut off {target}",
            f"deactivate {target}",
        )
    prefixes = ("", "uh ", "please ", "could you ", "hey ", "okay ")
    suffixes = ("", " please", " now", " thanks", " for me")
    core = cores[(index * 19 + attempt * 7 + spec["seed"]) % len(cores)]
    prefix = prefixes[(index * 11 + attempt * 5) % len(prefixes)]
    suffix = suffixes[(index * 13 + attempt * 3) % len(suffixes)]
    return f"{prefix}{core}{suffix}".strip()


def _utterance_multi_action(spec: dict[str, Any], rng: random.Random, *, attempt: int = 0) -> str:
    parts: list[str] = []
    for target, call in zip(spec["target_names"], spec["expected"]["calls"]):
        device_class = set(call["arguments"].get("device_class") or [])
        if call["name"] == "HassLightSet":
            parts.append(f"set {target} to {call['arguments']['brightness']} percent")
        elif call["name"] == "HassTurnOn" and "door" in device_class:
            parts.append(f"lock {target}")
        elif call["name"] == "HassTurnOff" and "door" in device_class:
            parts.append(f"unlock {target}")
        elif call["name"] == "HassTurnOn":
            parts.append(f"turn on {target}")
        else:
            parts.append(f"turn off {target}")
    utterance = " and ".join(parts)
    if spec["excluded_names"]:
        utterance += ", but leave " + " and ".join(spec["excluded_names"]) + " alone"
    prefixes = ("", "please ", "could you ", "hey ")
    suffixes = ("", " please", " now", " thanks")
    index = int(spec["candidate_id"].rsplit("_", 1)[-1])
    prefix = prefixes[(index * 17 + attempt * 7) % len(prefixes)]
    suffix = suffixes[(index * 11 + attempt * 13) % len(suffixes)]
    return f"{prefix}{utterance}{suffix}"


def _utterance_stt(spec: dict[str, Any], rng: random.Random, *, attempt: int = 0) -> str:
    canonical = spec["target_names"][0]
    spoken = spec["spoken_targets"].get(canonical, canonical)
    index = int(spec["candidate_id"].rsplit("_", 1)[-1])
    templates = (
        "turn on {spoken}",
        "turn on the {spoken}",
        "please turn on {spoken}",
        "could you turn on {spoken}",
        "hey turn on {spoken}",
        "pls turn on {spoken}",
        "uh turn on {spoken} please",
        "go ahead and turn on {spoken}",
        "switch on {spoken}",
        "activate {spoken}",
    )
    prefixes = ("", "uh ", "please ", "hey ", "okay ")
    suffixes = ("", " please", " now", " thanks")
    base = templates[(index * 19 + attempt * 7 + spec["seed"]) % len(templates)].format(spoken=spoken)
    prefix = prefixes[(index * 11 + attempt * 5) % len(prefixes)]
    suffix = suffixes[(index * 13 + attempt * 3) % len(suffixes)]
    clean = f"{prefix}{base}{suffix}".strip()
    asr_rng = random.Random((spec["seed"] << 8) ^ index ^ (attempt * 131))
    corrupted, corruption = _apply_stt(clean, asr_rng, min_edits=1, max_edits=2)
    spec["stt_corruption"] = corruption
    return corrupted


def _utterance_for_spec(spec: dict[str, Any], rng: random.Random, *, attempt: int = 0) -> str:
    category = spec["category"]
    if category == "light_fan_contrast":
        return _utterance_light_fan(spec, rng, attempt=attempt)
    if category == "lock_polarity":
        return _utterance_lock_polarity(spec, rng, attempt=attempt)
    if category == "apostrophe_names":
        return _utterance_apostrophe(spec, rng, attempt=attempt)
    if category == "multi_action":
        return _utterance_multi_action(spec, rng, attempt=attempt)
    if category == "ambiguity" and spec.get("request_hint"):
        return _utterance_ambiguity(spec, attempt=attempt)
    if category == "stt_corrupted":
        return _utterance_stt(spec, rng, attempt=attempt)
    if spec.get("utterance"):
        return spec["utterance"]
    return expand_utterance(spec)


def _assign_utterances(
    specs: list[dict[str, Any]],
    rng: random.Random,
    *,
    excluded_utterances: set[str],
) -> None:
    used: set[str] = set()
    for spec in specs:
        chosen: str | None = None
        for attempt in range(256):
            trial = dict(spec)
            trial["utterance"] = _utterance_for_spec(spec, rng, attempt=attempt)
            norm = _normalized(trial["utterance"])
            if norm in used or norm in excluded_utterances:
                continue
            if _validate_assigned_utterance(trial):
                continue
            chosen = trial["utterance"]
            if spec["category"] == "stt_corrupted" and trial.get("stt_corruption"):
                spec["stt_corruption"] = trial["stt_corruption"]
            used.add(norm)
            break
        if chosen is None:
            raise ValueError(f"{spec['candidate_id']}: unable to assign unique valid utterance")
        spec["utterance"] = chosen


def _build_light_fan_contrast(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random((seed << 20) ^ index)
    use_light = index % 2 == 0
    area = _FRESH_AREAS[index % len(_FRESH_AREAS)]
    light_name = _FRESH_REGULAR_NAMES[index % len(_FRESH_REGULAR_NAMES)]
    fan_name = f"{area} {_kind_noun('fan')}"
    if use_light:
        entity = _make_entity(name=light_name, kind="light", area=area, rng=rng)
        brightness = 20 + (index * 7) % 71
        expected = {"kind": "action", "calls": [_light_set_call(entity, brightness)]}
        subcategory = "brightness_not_fan_speed"
    else:
        entity = _make_entity(name=fan_name, kind="fan", area=area, rng=rng)
        percentage = 20 + (index * 11) % 71
        expected = {"kind": "action", "calls": [_fan_speed_call(entity, percentage)]}
        subcategory = "fan_speed_not_brightness"
    home = _fresh_home(index, rng, sayso_entity_area=area)
    home["entities"][0] = entity
    return _spec_shell(
        candidate_id=f"corrective_light_fan_{index:05d}",
        seed=seed,
        category="light_fan_contrast",
        subcategory=subcategory,
        home=home,
        expected=expected,
        target_names=[entity["name"]],
        dataset="corrective",
    )


def _build_lock_polarity(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random((seed << 18) ^ (index + 10_000))
    area = _FRESH_AREAS[(index + 2) % len(_FRESH_AREAS)]
    qualifiers = ("Front", "Back", "Side", "Entry", "Patio", "Garage", "Hall", "Rear")
    qualifier = qualifiers[(index // len(_FRESH_AREAS)) % len(qualifiers)]
    lock_name = f"{area} {qualifier} Door Lock"
    entity = _make_entity(name=lock_name, kind="lock", area=area, rng=rng, aliases=["door", f"{qualifier.lower()} door"])
    lock = index % 2 == 0
    expected = {"kind": "action", "calls": [_lock_call(entity, lock)]}
    home = _fresh_home(index + 10_000, rng, sayso_entity_area=area)
    home["entities"][0] = entity
    return _spec_shell(
        candidate_id=f"corrective_lock_{index:05d}",
        seed=seed,
        category="lock_polarity",
        subcategory="lock_on_unlock_off" if lock else "unlock_off_lock_on",
        home=home,
        expected=expected,
        target_names=[entity["name"]],
        dataset="corrective",
    )


def _build_apostrophe_names(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random((seed << 16) ^ (index + 20_000))
    name = _FRESH_APOSTROPHE_NAMES[index % len(_FRESH_APOSTROPHE_NAMES)]
    kind = "fan" if "fan" in name.casefold() else "light"
    area = _FRESH_AREAS[index % len(_FRESH_AREAS)]
    entity = _make_entity(name=name, kind=kind, area=area, rng=rng, aliases=[name])
    turn_on = index % 2 == 0
    expected = {"kind": "action", "calls": [_control_call(entity, turn_on, rng)]}
    home = _fresh_home(index + 20_000, rng, sayso_entity_area=area)
    home["entities"][0] = entity
    return _spec_shell(
        candidate_id=f"corrective_apostrophe_{index:05d}",
        seed=seed,
        category="apostrophe_names",
        subcategory="apostrophe",
        home=home,
        expected=expected,
        target_names=[entity["name"]],
        dataset="corrective",
    )


def _build_multi_action(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random((seed << 14) ^ (index + 30_000))
    area = _FRESH_AREAS[(index + 4) % len(_FRESH_AREAS)]
    light = _make_entity(name=f"{area} Desk Lamp", kind="light", area=area, rng=rng)
    outlet = _make_entity(name=f"{area} Wall Outlet", kind="switch", area=area, rng=rng)
    excluded = _make_entity(name=f"{area} Corner Fan", kind="fan", area=area, rng=rng)
    brightness = 25 + (index * 9) % 60
    expected = {
        "kind": "action",
        "calls": [_light_set_call(light, brightness), _control_call(outlet, False, rng)],
    }
    home = {
        "home_id": f"corrective_multi_{index:06d}",
        "sayso_entity_area": area,
        "entities": [light, outlet, excluded],
    }
    return _spec_shell(
        candidate_id=f"corrective_multi_{index:05d}",
        seed=seed,
        category="multi_action",
        subcategory="retention",
        home=home,
        expected=expected,
        target_names=[light["name"], outlet["name"]],
        excluded_names=[excluded["name"]],
        dataset="corrective",
    )


def build_corrective_specs(seed: int = 20260905) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.extend(_build_light_fan_contrast(index, seed) for index in range(CORRECTIVE_TARGETS["light_fan_contrast"]))
    specs.extend(_build_lock_polarity(index, seed) for index in range(CORRECTIVE_TARGETS["lock_polarity"]))
    specs.extend(_build_apostrophe_names(index, seed) for index in range(CORRECTIVE_TARGETS["apostrophe_names"]))
    specs.extend(_build_multi_action(index, seed) for index in range(CORRECTIVE_TARGETS["multi_action"]))
    return specs


def _shadow_ambiguity_scenario(index: int, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    scenario = (
        "one_default_light",
        "clarify_default_light",
        "named_area_light",
        "one_default_fan",
        "clarify_outlet",
        "one_default_blinds",
        "one_default_lock",
        "zero_lights",
    )[index % 8]
    sayso_area = {
        "one_default_light": "Den",
        "clarify_default_light": "Sunroom",
        "named_area_light": "Mudroom",
        "one_default_fan": "Game Room",
        "clarify_outlet": "Foyer",
        "one_default_blinds": "Craft Room",
        "one_default_lock": "Breakfast Nook",
        "zero_lights": "Utility Room",
    }[scenario]
    entities: list[dict[str, Any]] = []
    if scenario in {"one_default_light", "clarify_default_light", "zero_lights"}:
        if scenario != "zero_lights":
            entities.append(
                _make_entity(
                    name=f"{sayso_area} Sink Lamp",
                    kind="light",
                    area=sayso_area,
                    rng=rng,
                    aliases=["light"],
                )
            )
        if scenario == "clarify_default_light":
            entities.append(
                _make_entity(
                    name=f"{sayso_area} Ceiling Lamp",
                    kind="light",
                    area=sayso_area,
                    rng=rng,
                    aliases=["light"],
                )
            )
        entities.append(_make_entity(name="Media Room Main Lamp", kind="light", area="Media Room", rng=rng))
        hint = "turn on the light"
        if scenario == "one_default_light":
            expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
        elif scenario == "clarify_default_light":
            expected = {"kind": "no_action", "response": "clarify", "calls": []}
        else:
            expected = {
                "kind": "no_action",
                "response": "area_unavailable",
                "calls": [],
                "unavailable": {"area": sayso_area.casefold(), "type": "lights"},
            }
    elif scenario == "named_area_light":
        entities.append(_make_entity(name="Guest Suite Corner Lamp", kind="light", area="Guest Suite", rng=rng))
        hint = "turn on the guest suite light"
        expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
    elif scenario == "one_default_fan":
        entities.extend(
            [
                _make_entity(name="Game Room Ceiling Fan", kind="fan", area="Game Room", rng=rng, aliases=["fan"]),
                _make_entity(name="Reading Nook Side Fan", kind="fan", area="Reading Nook", rng=rng),
            ]
        )
        hint = "turn off the fan"
        expected = {"kind": "action", "calls": [_control_call(entities[0], False, rng)]}
    elif scenario == "clarify_outlet":
        entities.extend(
            [
                _make_entity(name="Foyer East Outlet", kind="switch", area="Foyer", rng=rng, aliases=["outlet"]),
                _make_entity(name="Foyer West Outlet", kind="switch", area="Foyer", rng=rng, aliases=["outlet"]),
            ]
        )
        hint = "turn on the outlet"
        expected = {"kind": "no_action", "response": "clarify", "calls": []}
    elif scenario == "one_default_blinds":
        entities.extend(
            [
                _make_entity(name="Craft Room South Blinds", kind="blinds", area="Craft Room", rng=rng, aliases=["blinds"]),
                _make_entity(name="Sun Porch Blinds", kind="blinds", area="Sunroom", rng=rng),
            ]
        )
        hint = "open the blinds"
        expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
    else:
        entities.extend(
            [
                _make_entity(name="Breakfast Nook Back Door Lock", kind="lock", area="Breakfast Nook", rng=rng, aliases=["door"]),
                _make_entity(name="Mudroom Entry Lock", kind="lock", area="Mudroom", rng=rng),
            ]
        )
        hint = "lock the door"
        expected = {"kind": "action", "calls": [_control_call(entities[0], True, rng)]}
    home = {
        "home_id": f"shadow_ambiguity_{index:06d}",
        "sayso_entity_area": sayso_area,
        "entities": entities,
    }
    return home, expected, hint, scenario


def build_shadow_specs(seed: int = 20260905, count: int = DEFAULT_SHADOW_COUNT) -> list[dict[str, Any]]:
    if not SHADOW_MIN <= count <= SHADOW_MAX:
        raise ValueError(f"shadow count must be {SHADOW_MIN}-{SHADOW_MAX}, got {count}")
    recipe_slots = {
        "clean_direct": max(1, count // 10),
        "conversational": max(1, count // 10),
        "entity_identity": max(1, count // 8),
        "multi_action_exclusion": max(1, count // 10),
        "stt_corrupted": max(1, count // 10),
        "status": max(1, count // 10),
        "ambiguity": max(1, count // 6),
        "unsupported_no_action": max(1, count // 12),
        "light_fan_contrast": max(1, count // 10),
        "lock_polarity": max(1, count // 10),
    }
    while sum(recipe_slots.values()) > count:
        key = max(recipe_slots, key=lambda name: recipe_slots[name])
        recipe_slots[key] -= 1
    while sum(recipe_slots.values()) < count:
        recipe_slots["ambiguity"] += 1

    specs: list[dict[str, Any]] = []

    for slot in range(recipe_slots["clean_direct"]):
        rng = random.Random((seed << 24) ^ slot)
        area = _FRESH_AREAS[slot % len(_FRESH_AREAS)]
        light = _make_entity(name=f"{area} Task Lamp", kind="light", area=area, rng=rng)
        home = {"home_id": f"shadow_clean_{slot:04d}", "sayso_entity_area": area, "entities": [light]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_clean_{slot:04d}",
                seed=seed,
                category="clean_direct",
                subcategory="named_device",
                home=home,
                expected={"kind": "action", "calls": [_control_call(light, True, rng)]},
                target_names=[light["name"]],
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["conversational"]):
        rng = random.Random((seed << 22) ^ (slot + 100))
        area = _FRESH_AREAS[(slot + 3) % len(_FRESH_AREAS)]
        light = _make_entity(name=f"{area} Mood Lamp", kind="light", area=area, rng=rng)
        brightness = 30 + slot * 5
        home = {"home_id": f"shadow_conv_{slot:04d}", "sayso_entity_area": area, "entities": [light]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_conv_{slot:04d}",
                seed=seed,
                category="conversational",
                subcategory="brightness",
                home=home,
                expected={"kind": "action", "calls": [_light_set_call(light, brightness)]},
                target_names=[light["name"]],
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["entity_identity"]):
        rng = random.Random((seed << 20) ^ (slot + 200))
        name = _FRESH_APOSTROPHE_NAMES[(slot + 5) % len(_FRESH_APOSTROPHE_NAMES)]
        kind = "fan" if "fan" in name.casefold() else "light"
        area = _FRESH_AREAS[(slot + 1) % len(_FRESH_AREAS)]
        entity = _make_entity(name=name, kind=kind, area=area, rng=rng, aliases=[name])
        home = {"home_id": f"shadow_identity_{slot:04d}", "sayso_entity_area": area, "entities": [entity]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_identity_{slot:04d}",
                seed=seed,
                category="entity_identity",
                subcategory="apostrophe",
                home=home,
                expected={"kind": "action", "calls": [_control_call(entity, slot % 2 == 0, rng)]},
                target_names=[entity["name"]],
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["multi_action_exclusion"]):
        spec = _build_multi_action(slot + 40_000, seed)
        spec["candidate_id"] = f"shadow_multi_{slot:04d}"
        spec["dataset"] = "shadow"
        spec["category"] = "multi_action_exclusion"
        specs.append(spec)

    for slot in range(recipe_slots["stt_corrupted"]):
        rng = random.Random((seed << 16) ^ (slot + 400))
        area = _FRESH_AREAS[(slot + 6) % len(_FRESH_AREAS)]
        target = _make_entity(name=f"{area} Hall Lamp", kind="light", area=area, rng=rng)
        spoken, corruption = _apply_stt(f"turn on {target['name']}", rng)
        home = {"home_id": f"shadow_stt_{slot:04d}", "sayso_entity_area": area, "entities": [target]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_stt_{slot:04d}",
                seed=seed,
                category="stt_corrupted",
                subcategory="canonical_resolution",
                home=home,
                expected={"kind": "action", "calls": [_control_call(target, True, rng)]},
                target_names=[target["name"]],
                spoken_targets={target["name"]: spoken},
                stt_corruption=corruption,
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["status"]):
        rng = random.Random((seed << 14) ^ (slot + 500))
        area = _FRESH_AREAS[(slot + 2) % len(_FRESH_AREAS)]
        target = _make_entity(name=f"{area} Panel Fan", kind="fan", area=area, rng=rng)
        home = {"home_id": f"shadow_status_{slot:04d}", "sayso_entity_area": area, "entities": [target]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_status_{slot:04d}",
                seed=seed,
                category="status",
                subcategory="named_device",
                home=home,
                expected={"kind": "status", "calls": [_status_call(target)], "state": target["state"]},
                target_names=[target["name"]],
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["ambiguity"]):
        rng = random.Random((seed << 12) ^ (slot + 600))
        home, expected, hint, scenario = _shadow_ambiguity_scenario(slot, rng)
        target_names = [
            call["arguments"]["name"]
            for call in expected.get("calls", [])
            if isinstance(call.get("arguments"), dict) and call["arguments"].get("name")
        ]
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_ambiguity_{slot:04d}",
                seed=seed,
                category="ambiguity",
                subcategory=scenario,
                home=home,
                expected=expected,
                target_names=target_names,
                request_hint=hint,
                dataset="shadow",
            )
        )

    unsupported_hints = (
        ("refuse", "disable the pantry alarm safety system"),
        ("clarify", "set the lamp to"),
        ("unsupported", "play music in the den"),
        ("refuse", "disable the nursery alarm safety system"),
        ("clarify", "set the fan to"),
        ("unsupported", "play music in the loft"),
        ("refuse", "disable the garage alarm safety system"),
        ("clarify", "set the blinds to"),
        ("unsupported", "play music in the sunroom"),
        ("refuse", "disable the foyer alarm safety system"),
        ("clarify", "set the outlet to"),
        ("unsupported", "play music in the craft room"),
    )
    for slot in range(recipe_slots["light_fan_contrast"]):
        rng = random.Random((seed << 8) ^ (slot + 800))
        area = _SHADOW_CONTRAST_AREAS[slot % len(_SHADOW_CONTRAST_AREAS)]
        use_light = slot % 2 == 0
        if use_light:
            entity = _make_entity(name=f"{area} Cove Light", kind="light", area=area, rng=rng)
            expected = {"kind": "action", "calls": [_light_set_call(entity, 20 + slot * 5)]}
            subcategory = "brightness_not_fan_speed"
        else:
            entity = _make_entity(name=f"{area} Exhaust Fan", kind="fan", area=area, rng=rng)
            expected = {"kind": "action", "calls": [_fan_speed_call(entity, 25 + slot * 4)]}
            subcategory = "fan_speed_not_brightness"
        home = {"home_id": f"shadow_light_fan_{slot:04d}", "sayso_entity_area": area, "entities": [entity]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_light_fan_{slot:04d}",
                seed=seed,
                category="light_fan_contrast",
                subcategory=subcategory,
                home=home,
                expected=expected,
                target_names=[entity["name"]],
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["lock_polarity"]):
        rng = random.Random((seed << 6) ^ (slot + 900))
        area = _SHADOW_CONTRAST_AREAS[(slot + 3) % len(_SHADOW_CONTRAST_AREAS)]
        lock = slot % 2 == 0
        entity = _make_entity(
            name=f"{area} Service Door Latch",
            kind="lock",
            area=area,
            rng=rng,
            aliases=["door", "service door"],
        )
        home = {"home_id": f"shadow_lock_{slot:04d}", "sayso_entity_area": area, "entities": [entity]}
        specs.append(
            _spec_shell(
                candidate_id=f"shadow_lock_{slot:04d}",
                seed=seed,
                category="lock_polarity",
                subcategory="lock_on_unlock_off" if lock else "unlock_off_lock_on",
                home=home,
                expected={"kind": "action", "calls": [_lock_call(entity, lock)]},
                target_names=[entity["name"]],
                dataset="shadow",
            )
        )

    for slot in range(recipe_slots["unsupported_no_action"]):
        response, hint = unsupported_hints[slot % len(unsupported_hints)]
        rng = random.Random((seed << 10) ^ (slot + 700))
        area = _FRESH_AREAS[slot % len(_FRESH_AREAS)]
        light = _make_entity(name=f"{area} Desk Lamp", kind="light", area=area, rng=rng)
        home = {"home_id": f"shadow_unsupported_{slot:04d}", "sayso_entity_area": area, "entities": [light]}
        spec = _spec_shell(
            candidate_id=f"shadow_unsupported_{slot:04d}",
            seed=seed,
            category="unsupported_no_action",
            subcategory=response,
            home=home,
            expected={"kind": "no_action", "response": response, "calls": []},
            target_names=[],
            request_hint=hint,
            dataset="shadow",
        )
        spec["utterance"] = hint
        specs.append(spec)

    return specs[:count]


def _load_excluded_utterances(
    *,
    heldout_path: Path | None,
    base_train_path: Path | None,
) -> set[str]:
    excluded: set[str] = set()
    for path in (heldout_path, base_train_path):
        if path and path.is_file():
            excluded.update(_normalized(text) for text in load_user_utterances(path))
    excluded.update(_normalized(text) for text in quality_eval_user_prompts())
    return excluded


def _validate_light_fan_tool(spec: dict[str, Any]) -> None:
    if spec["category"] != "light_fan_contrast":
        return
    call = spec["expected"]["calls"][0]
    entity = next(entity for entity in spec["home"]["entities"] if entity["name"] == spec["target_names"][0])
    if entity["kind"] == "light" and call["name"] != "HassLightSet":
        raise ValueError(f"{spec['candidate_id']}: light row must use HassLightSet")
    if entity["kind"] == "fan" and call["name"] != "HassFanSetSpeed":
        raise ValueError(f"{spec['candidate_id']}: fan row must use HassFanSetSpeed")


def _validate_lock_polarity(spec: dict[str, Any]) -> None:
    if spec["category"] != "lock_polarity":
        return
    call = spec["expected"]["calls"][0]
    lock = spec["subcategory"] == "lock_on_unlock_off"
    if lock and call["name"] != "HassTurnOn":
        raise ValueError(f"{spec['candidate_id']}: lock must use HassTurnOn")
    if not lock and call["name"] != "HassTurnOff":
        raise ValueError(f"{spec['candidate_id']}: unlock must use HassTurnOff")


def validate_corrective_specs(
    specs: list[dict[str, Any]],
    *,
    recipe_lock_utterances: set[str],
    recipe_lock_entities: set[str],
    heldout_utterances: set[str],
    check_counts: bool = True,
) -> None:
    count = len(specs)
    if check_counts and not CORRECTIVE_MIN <= count <= CORRECTIVE_MAX:
        raise ValueError(f"corrective row count must be {CORRECTIVE_MIN}-{CORRECTIVE_MAX}, got {count}")
    if check_counts:
        counts = Counter(spec["category"] for spec in specs)
        for category, target in CORRECTIVE_TARGETS.items():
            if counts.get(category, 0) != target:
                raise ValueError(f"corrective {category}: expected {target}, got {counts.get(category, 0)}")
    _validate_specs(
        specs,
        recipe_lock_utterances=recipe_lock_utterances,
        recipe_lock_entities=recipe_lock_entities,
        heldout_utterances=heldout_utterances,
    )


def validate_shadow_specs(
    specs: list[dict[str, Any]],
    *,
    recipe_lock_utterances: set[str],
    recipe_lock_entities: set[str],
    heldout_utterances: set[str],
    check_counts: bool = True,
) -> None:
    count = len(specs)
    if check_counts and not SHADOW_MIN <= count <= SHADOW_MAX:
        raise ValueError(f"shadow row count must be {SHADOW_MIN}-{SHADOW_MAX}, got {count}")
    _validate_specs(
        specs,
        recipe_lock_utterances=recipe_lock_utterances,
        recipe_lock_entities=recipe_lock_entities,
        heldout_utterances=heldout_utterances,
    )


def _validate_specs(
    specs: list[dict[str, Any]],
    *,
    recipe_lock_utterances: set[str],
    recipe_lock_entities: set[str],
    heldout_utterances: set[str],
) -> None:
    normalized_prompts: set[str] = set()
    for spec in specs:
        reason = validate_spec(spec)
        if reason:
            raise ValueError(f"{spec['candidate_id']}: validate_spec -> {reason}")
        for entity in spec["home"]["entities"]:
            if entity["name"] in recipe_lock_entities:
                raise ValueError(f"{spec['candidate_id']}: reuses recipe-lock entity {entity['name']!r}")
        utterance = spec.get("utterance")
        if not isinstance(utterance, str):
            raise ValueError(f"{spec['candidate_id']}: missing utterance")
        norm = _normalized(utterance)
        if norm in recipe_lock_utterances:
            raise ValueError(f"{spec['candidate_id']}: overlaps recipe-lock utterance")
        if norm in heldout_utterances:
            raise ValueError(f"{spec['candidate_id']}: overlaps held-out prompt")
        if norm in normalized_prompts:
            raise ValueError(f"{spec['candidate_id']}: duplicate normalized utterance")
        normalized_prompts.add(norm)
        _validate_light_fan_tool(spec)
        _validate_lock_polarity(spec)
        if _validate_assigned_utterance(spec):
            raise ValueError(f"{spec['candidate_id']}: validate_utterance failed")
        if spec["category"] == "status":
            calls = spec["expected"]["calls"]
            if not calls or any(call["name"] != "GetLiveContext" for call in calls):
                raise ValueError(f"{spec['candidate_id']}: status must use GetLiveContext only")
        if spec["category"] == "apostrophe_names" or (
            spec["category"] == "entity_identity" and spec["subcategory"] == "apostrophe"
        ):
            if "'" not in spec["target_names"][0]:
                raise ValueError(f"{spec['candidate_id']}: apostrophe row missing apostrophe in name")


def build_corrective_examples(
    *,
    seed: int = 20260905,
    heldout_path: Path | None = DEFAULT_HELDOUT,
    base_train_path: Path | None = DEFAULT_BASE_TRAIN,
) -> list[dict[str, Any]]:
    specs = build_corrective_specs(seed)
    excluded = _load_excluded_utterances(heldout_path=heldout_path, base_train_path=base_train_path)
    heldout_utterances = (
        {_normalized(text) for text in load_user_utterances(heldout_path)}
        if heldout_path and heldout_path.is_file()
        else set()
    )
    _assign_utterances(specs, random.Random(seed), excluded_utterances=excluded)
    validate_corrective_specs(
        specs,
        recipe_lock_utterances={_normalized(text) for text in quality_eval_user_prompts()},
        recipe_lock_entities=_recipe_lock_entity_names(),
        heldout_utterances=heldout_utterances,
    )
    return [render_example(spec) for spec in specs]


def build_shadow_examples(
    *,
    seed: int = 20260905,
    count: int = DEFAULT_SHADOW_COUNT,
    heldout_path: Path | None = DEFAULT_HELDOUT,
    base_train_path: Path | None = DEFAULT_BASE_TRAIN,
) -> list[dict[str, Any]]:
    specs = build_shadow_specs(seed, count=count)
    excluded = _load_excluded_utterances(heldout_path=heldout_path, base_train_path=base_train_path)
    heldout_utterances = (
        {_normalized(text) for text in load_user_utterances(heldout_path)}
        if heldout_path and heldout_path.is_file()
        else set()
    )
    _assign_utterances(specs, random.Random(seed + 1), excluded_utterances=excluded)
    validate_shadow_specs(
        specs,
        recipe_lock_utterances={_normalized(text) for text in quality_eval_user_prompts()},
        recipe_lock_entities=_recipe_lock_entity_names(),
        heldout_utterances=heldout_utterances,
    )
    rows = [render_example(spec) for spec in specs]
    for row in rows:
        row["metadata"]["shadow_eval"] = True
    return rows


def corrective_category_counts(specs: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(spec["category"] for spec in specs).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrective-out", type=Path, default=DEFAULT_CORRECTIVE_OUT)
    parser.add_argument("--shadow-out", type=Path, default=DEFAULT_SHADOW_OUT)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--shadow-count", type=int, default=DEFAULT_SHADOW_COUNT)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    args = parser.parse_args()

    corrective = build_corrective_examples(
        seed=args.seed,
        heldout_path=args.heldout,
        base_train_path=args.base_train,
    )
    shadow = build_shadow_examples(
        seed=args.seed,
        count=args.shadow_count,
        heldout_path=args.heldout,
        base_train_path=args.base_train,
    )
    args.corrective_out.parent.mkdir(parents=True, exist_ok=True)
    args.shadow_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.corrective_out, corrective)
    write_jsonl(args.shadow_out, shadow)
    report = {
        "corrective": {
            "path": str(args.corrective_out),
            "rows": len(corrective),
            "counts": corrective_category_counts(build_corrective_specs(args.seed)),
        },
        "shadow": {
            "path": str(args.shadow_out),
            "rows": len(shadow),
            "seed": args.seed,
            "shadow_count": args.shadow_count,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
